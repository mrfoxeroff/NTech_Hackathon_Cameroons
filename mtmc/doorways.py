"""Recover a dormant global identity at the same physical entrance."""

import math

from .associate import split_tracklets, upper_anchor
from .reconnect import recent_appearance, similarity


def in_portal(obs, zone):
    if obs.xy is None or obs.camera not in zone:
        return False
    left, top, right, bottom = zone[obs.camera]
    x, y = obs.xy
    return left <= x <= right and top <= y <= bottom


def doorway_score(before, after, zones, memory_ms, overlaps):
    a, b = before.last, after.first
    if a.exit_after or b.time <= a.time:
        return None
    if a.bbox is None or b.bbox is None:
        return None
    a_zones = {name for name, zone in zones.items() if in_portal(a, zone)}
    b_zones = {name for name, zone in zones.items() if in_portal(b, zone)}
    if a_zones and b_zones and not a_zones & b_zones:
        return None
    door = min(a_zones & b_zones, default=None)
    same_camera = a.camera == b.camera
    distance = math.dist(a.xy, b.xy) if a.xy is not None and b.xy is not None else None
    head_error = (
        math.dist(upper_anchor(a), upper_anchor(b)) / min(a.bbox[3] - a.bbox[1], b.bbox[3] - b.bbox[1])
        if same_camera
        else 0
    )
    if distance is None:
        if not same_camera or head_error > 0.25:
            return None
        look = similarity(before.reid, after.reid)
        gallery = recent_appearance(before, after, memory_ms)
        if look is not None and gallery is not None and look >= 0.88 and gallery >= 0.90:
            return (look + gallery) / 2 - 0.35 * head_error, None
        return None
    shared_view = frozenset((a.camera, b.camera)) in overlaps
    if door is None and not same_camera and (not shared_view or distance > 8):
        return None
    if door is None and same_camera and (distance > 5 or head_error > 0.19):
        return None
    if distance > 17 or head_error > 0.42:
        return None
    look = similarity(before.reid, after.reid)
    gallery = recent_appearance(before, after, memory_ms)
    if look is None or gallery is None:
        return None
    if door is None and (look < 0.85 or gallery < 0.82):
        return None
    if door is not None and same_camera and (look < 0.68 or gallery < 0.68 or (look + gallery) / 2 < 0.735):
        return None
    if not same_camera and (look < 0.60 or gallery < 0.67):
        return None
    return (look + gallery) / 2 - 0.21 * head_error - distance / 130, door


def best_evidence(departures, arrivals, zones, memory, overlaps):
    candidates = []
    for before in departures:
        for after in arrivals:
            match = doorway_score(before, after, zones, memory, overlaps)
            if match:
                candidates.append((match[0], match[1], before.last, after.first))
    return max(candidates, key=lambda item: item[0], default=None)


def restore_door_returns(groups, config):
    """Process arrivals in time order. Memory starts at the last sighting in any view."""
    zones = config.get("door_return_zones") or {}
    memory = config["reappearance_memory_ms"]
    overlaps = {frozenset(pair) for pair in config.get("overlap_pairs", [])} if not zones else set()
    if not memory:
        return groups, []
    tracks = {i: split_tracklets(group, config["visibility_gap_ms"]) for i, group in enumerate(groups)}
    starts = {i: runs[0].first.time for i, runs in tracks.items()}
    ends = {i: max(run.last.time for run in runs) for i, runs in tracks.items()}
    beginnings = {
        i: [run for run in runs if run.first.time - starts[i] <= 1100] for i, runs in tracks.items()
    }
    current = dict(enumerate(groups))
    links = []
    for arriving in sorted(tracks, key=lambda i: (starts[i], i)):
        start = starts[arriving]
        proposals = []
        for old in current:
            if old == arriving or not 0 < start - ends[old] <= memory:
                continue
            departures = [run for run in tracks[old] if ends[old] - run.last.time <= 1100]
            if any(run.last.exit_after for run in departures if ends[old] - run.last.time <= 200):
                continue
            match = best_evidence(departures, beginnings[arriving], zones, memory, overlaps)
            if match:
                proposals.append((match, old, departures))
        proposals.sort(key=lambda item: (-item[0][0], item[1]))
        if not proposals:
            continue
        (score, door, before, after), old, departures = proposals[0]
        if len(proposals) > 1 and score - proposals[1][0][0] < 0.07:
            continue
        rivals = [
            best_evidence(departures, beginnings[i], zones, memory, overlaps)
            for i in current
            if i not in (old, arriving)
            and abs(starts[i] - start) <= 1100
            and starts[i] <= ends[arriving]
            and start <= ends[i]
            and 0 < starts[i] - ends[old] <= memory
        ]
        if any(rival and score - rival[0] < 0.07 for rival in rivals):
            continue
        # A dormant ID can be claimed only once at this arrival time.
        # Its next recovery will be measured from the new last sighting.
        links.append(
            {
                "door": door,
                "gap_ms": start - ends[old],
                "last_seen_ms": ends[old],
                "returned_ms": start,
                "score": round(score, 3),
                "exit": [before.camera, before.local_id, before.time],
                "entry": [after.camera, after.local_id, after.time],
            }
        )
        current[old] = current[old] + current.pop(arriving)
        tracks[old].extend(tracks[arriving])
        ends[old] = ends[arriving]
    return list(current.values()), links
