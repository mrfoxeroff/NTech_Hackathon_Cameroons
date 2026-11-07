"""Repair reciprocal local ID switches before cross-camera association."""

import math
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass, replace
from statistics import median

import numpy as np
from lap import lapjv

from .associate import DEFAULTS, Observation, associate, split_tracklets, summarize


@dataclass(frozen=True)
class Switch:
    fragment: int
    camera: str
    old_sample: int
    new_sample: int
    brightness_delta: float

    @property
    def midpoint(self):
        return (self.old_sample + self.new_sample) / 2


def appearance_switches(fragments):
    candidates = []
    for index, fragment in enumerate(fragments):
        samples = [obs for obs in fragment.observations if obs.reid and obs.brightness is not None]
        for a, b, c, d in zip(samples, samples[1:], samples[2:], samples[3:]):
            if max(b.time - a.time, c.time - b.time, d.time - c.time) > 1000:
                continue
            previous, following = [a.brightness, b.brightness], [c.brightness, d.brightness]
            delta = median(following) - median(previous)
            if (
                abs(delta) < 0.35
                or abs(a.brightness - b.brightness) > 0.20
                or abs(c.brightness - d.brightness) > 0.20
            ):
                continue
            left, right = summarize([a, b], "reid"), summarize([c, d], "reid")
            if (
                left is not None
                and right is not None
                and sum(x * y for x, y in zip(left, right, strict=True)) < 0.65
            ):
                candidates.append(Switch(index, b.camera, b.time, c.time, delta))
    # A reciprocal change within one view distinguishes track swaps from a
    # single person changing pose or a lighting change across the whole scene.
    return [
        event
        for event in candidates
        if any(
            other.camera == event.camera
            and other.fragment != event.fragment
            and fragments[other.fragment].first.local_id != fragments[event.fragment].first.local_id
            and abs(other.midpoint - event.midpoint) <= 1000
            and other.brightness_delta * event.brightness_delta < 0
            for other in candidates
        )
    ]


def matching_point(row, matrix, size):
    """Use the raw lower box only as approximate evidence when feet are hidden."""
    if row.get("touches_frame_bottom"):
        return None
    point = row.get("footpoint_px")
    if point is None or row.get("footpoint_status") != "observed":
        x1, _, x2, y2 = row["bbox_xyxy"]
        point = ((x1 + x2) / 2, y2)
    x, y, scale = matrix @ [*point, 1]
    if not math.isfinite(scale) or abs(scale) < 1e-9:
        return None
    x, y = x / scale, y / scale
    if 0 <= x < size[0] and 0 <= y < size[1]:
        return float(x), float(y)
    return None


def nested_duplicates(fragments, borders):
    """Suppress a short inner box when a longer track covers the same person."""
    frames = defaultdict(list)
    for index, fragment in enumerate(fragments):
        for obs in fragment.observations:
            if obs.bbox:
                frames[obs.camera, obs.frame].append((index, obs))

    suppressed = set()
    for index, fragment in enumerate(fragments):
        detections = fragment.observations
        singleton = len(detections) == 1
        if not singleton and (len(detections) < 3 or fragment.last.time - fragment.first.time > 1200):
            continue
        owners = None
        for inner in detections:
            if not inner.bbox:
                owners = set()
                break
            x1, y1, x2, y2 = inner.bbox
            width, height = x2 - x1, y2 - y1
            matching = set()
            for outer_index, outer in frames[inner.camera, inner.frame]:
                if outer_index == index or (outer.confidence or 0) < (inner.confidence or 0):
                    continue
                owner = fragments[outer_index]
                if len(owner.observations) <= len(detections):
                    continue
                a, b, c, d = outer.bbox
                overlap = max(0, min(x2, c) - max(x1, a)) * max(0, min(y2, d) - max(y1, b))
                if overlap < 0.95 * width * height or abs(y1 - b) > 0.08 * (d - b):
                    continue
                if singleton:
                    edge = any(
                        borders.get(inner, {}).get(side) and borders.get(outer, {}).get(side)
                        for side in ("left", "right")
                    )
                    if edge and overlap >= 0.60 * (c - a) * (d - b) and abs(y2 - d) <= 0.08 * (d - b):
                        matching.add(outer_index)
                elif height <= 0.80 * (d - b) and abs((x1 + x2 - a - c) / 2) <= 0.12 * (c - a):
                    matching.add(outer_index)
            owners = matching if owners is None else owners & matching
            if not owners:
                break
        if owners:
            suppressed.update(detections)
    return suppressed


def nearest(series, timestamp, tolerance):
    times, observations = series
    index = bisect_left(times, timestamp)
    candidates = observations[max(0, index - 1) : index + 1]
    closest = min(candidates, key=lambda obs: abs(obs.time - timestamp), default=None)
    return closest if closest is not None and abs(closest.time - timestamp) <= tolerance else None


def time_series(observations):
    ordered = sorted(observations, key=lambda obs: (obs.time, obs.frame))
    return [obs.time for obs in ordered], ordered


def pair_cost(target, reference, target_feature, reference_feature, preferred):
    distance = math.dist(target.xy, reference.xy)
    if distance > 24:
        return math.inf
    cost = distance + (0 if preferred else 1.5)
    if target_feature and reference_feature:
        brightness = abs(target_feature.brightness - reference_feature.brightness)
        if brightness > 0.35:
            return math.inf
        similarity = sum(x * y for x, y in zip(target_feature.reid, reference_feature.reid, strict=True))
        cost += 8 * brightness + 3 * (1 - similarity)
    return cost


def geometry_votes(frame, references, features, base):
    usable = [obs for obs in frame if obs.xy is not None]
    if not usable or not references:
        return {}
    ids = list(references)
    costs = np.full((len(usable), len(ids) + len(usable)), 26.0)
    for i, target in enumerate(usable):
        feature = nearest(features[target.camera, target.local_id], target.time, 700)
        for j, ident in enumerate(ids):
            reference, reference_feature = references[ident]
            costs[i, j] = pair_cost(target, reference, feature, reference_feature, base[target] == ident)
    # An extra column per detection allows unmatched observations. The difference
    # to the best solution without an edge measures ambiguity in the whole frame.
    best, assignment, _ = lapjv(costs, extend_cost=True)
    votes = {}
    for i, j in enumerate(assignment):
        if not 0 <= j < len(ids) or costs[i, j] >= 26:
            continue
        old = costs[i, j]
        costs[i, j] = math.inf
        alternate, _, _ = lapjv(costs, extend_cost=True)
        costs[i, j] = old
        if alternate - best >= 3:
            votes[usable[i]] = ids[j]
    return votes


def stable_runs(observations, votes):
    runs = []
    for obs in observations:
        if obs not in votes:
            continue
        ident = votes[obs]
        if not runs or runs[-1][0] != ident or obs.time - runs[-1][1][-1] > 450:
            runs.append((ident, []))
        runs[-1][1].append(obs.time)
    return [(ident, times[0], times[-1]) for ident, times in runs if len(times) >= 3]


def repair_geometry(groups, fragments, reference_camera, config):
    base = {obs: ident for ident, group in enumerate(groups) for obs in group}
    anchor = {
        ident: time_series(o for o in group if o.camera == reference_camera and o.xy is not None)
        for ident, group in enumerate(groups)
    }
    anchor_features = {
        ident: time_series(
            o for o in group if o.camera == reference_camera and o.reid and o.brightness is not None
        )
        for ident, group in enumerate(groups)
    }
    by_track = defaultdict(list)
    frames = defaultdict(list)
    for obs in base:
        if obs.camera == reference_camera:
            continue
        frames[obs.camera, obs.frame].append(obs)
        if obs.reid and obs.brightness is not None:
            by_track[obs.camera, obs.local_id].append(obs)
    features = {key: time_series(by_track[key]) for key in {(obs.camera, obs.local_id) for obs in base}}
    votes = {}
    overlaps = {frozenset(pair) for pair in config["overlap_pairs"]}
    for (camera, _), frame in frames.items():
        if frozenset((camera, reference_camera)) not in overlaps:
            continue
        time = frame[0].time
        references = {
            ident: (point, nearest(anchor_features[ident], time, 700))
            for ident, series in anchor.items()
            if (point := nearest(series, time, config["sync_tolerance_ms"])) is not None
        }
        votes.update(geometry_votes(frame, references, features, base))

    assignment = dict(base)
    strong = set()
    for fragment in fragments:
        if fragment.first.camera == reference_camera:
            continue
        runs = stable_runs(fragment.observations, votes)
        if not runs:
            continue
        for obs in fragment.observations:
            before = next((run for run in reversed(runs) if run[1] <= obs.time), None)
            after = next((run for run in runs if run[1] > obs.time), None)
            if before is None:
                assignment[obs] = after[0]
            elif obs.time <= before[2] or after is None or before[0] == after[0]:
                assignment[obs] = before[0]
            else:
                assignment[obs] = None
            if assignment[obs] is not None and votes.get(obs) == assignment[obs]:
                strong.add(obs)

    # Continuity can extend a match past the last frame seen in the anchor view.
    # Do not let such extensions assign the same ID to two boxes in one frame.
    for frame in frames.values():
        owners = defaultdict(list)
        for obs in frame:
            if assignment[obs] is not None:
                owners[assignment[obs]].append(obs)
        for detections in owners.values():
            if len(detections) <= 1:
                continue
            confirmed = [obs for obs in detections if obs in strong]
            keep = confirmed[0] if len(confirmed) == 1 else None
            for obs in detections:
                if obs != keep:
                    assignment[obs] = None

    result = defaultdict(list)
    pending = []
    for obs, ident in assignment.items():
        if ident is None:
            pending.append(obs)
        else:
            result[ident].append(obs)
    changed = sum(ident is not None and ident != base[obs] for obs, ident in assignment.items())
    return [result[ident] for ident in sorted(result)], pending, changed


def associate_consistently(rows, settings, calibrations, plan_size):
    config = (
        DEFAULTS
        | settings
        | {
            "max_map_distance": 24.0,
            "max_brightness_difference": 0.25,
            "allow_geometry_only": False,
        }
    )
    observations = [
        replace(Observation.parse(row), xy=matching_point(row, calibrations[row["camera_id"]], plan_size))
        for row in rows
    ]
    fragments = split_tracklets(observations, config["max_local_gap_ms"])
    duplicates = nested_duplicates(
        fragments, {obs: row.get("border_flags") or {} for obs, row in zip(observations, rows, strict=True)}
    )
    fragments = [fragment for fragment in fragments if fragment.first not in duplicates]
    events = appearance_switches(fragments)
    cuts = defaultdict(list)
    for event in events:
        cuts[event.fragment].append((event.old_sample, event.new_sample))
    working, pending = [], [obs for obs in observations if obs in duplicates]
    originals = {}
    for index, fragment in enumerate(fragments):
        windows = sorted(cuts[index])
        for obs in fragment.observations:
            if any(start < obs.time < end for start, end in windows):
                pending.append(obs)
                continue
            part = sum(obs.time >= end for _, end in windows)
            local_id = f"{index}:{part}"
            working.append(replace(obs, local_id=local_id))
            originals[obs.camera, local_id, obs.frame] = obs
    groups = associate(working, config)
    reference = min(calibrations)
    groups, ambiguous, changed = repair_geometry(
        groups, split_tracklets(working, config["max_local_gap_ms"]), reference, config
    )
    groups = [[originals[obs.camera, obs.local_id, obs.frame] for obs in group] for group in groups]
    pending.extend(originals[obs.camera, obs.local_id, obs.frame] for obs in ambiguous)
    report = {
        "switches": [
            {
                "camera_id": event.camera,
                "local_id": fragments[event.fragment].first.local_id,
                "before_ms": event.old_sample,
                "after_ms": event.new_sample,
            }
            for event in events
        ],
        "pending_observations": len(pending),
        "duplicate_observations": len(duplicates),
        "reassigned_by_geometry": changed,
        "map_distance_px": config["max_map_distance"],
    }
    return groups, pending, report
