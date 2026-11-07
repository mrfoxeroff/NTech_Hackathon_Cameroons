"""Offline cross-camera tracking for a single synchronized recording session."""

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median


def vector(value):
    if value is None:
        return None
    values = tuple(map(float, value))
    if not values or not all(map(math.isfinite, values)):
        raise ValueError("Expected a nonempty vector of finite numbers")
    return values


def unit(values):
    norm = math.hypot(*values)
    return tuple(value / norm for value in values) if norm else None


@dataclass(frozen=True)
class Observation:
    camera: str
    local_id: str | int
    frame: int
    time: int
    zone: str | None
    reid: tuple | None
    hsv: tuple | None
    xy: tuple | None
    exit_after: bool
    brightness: float | None = None
    bbox: tuple | None = None
    confidence: float | None = None
    appearance_weight: float = 1.0
    source_segment: int = 0

    @classmethod
    def parse(cls, row):
        frame, time = row["frame_index"], row["timestamp_ms"]
        if any(type(value) is not int or value < 0 for value in (frame, time)):
            raise ValueError("frame_index and timestamp_ms must be nonnegative integers")
        camera, local_id = row["camera_id"], row["local_id"]
        if not isinstance(camera, str) or not camera or type(local_id) not in (str, int):
            raise ValueError("camera_id must be a nonempty string and local_id a string or integer")
        xy = vector(row.get("map_xy"))
        if xy is not None and len(xy) != 2:
            raise ValueError("map_xy must contain two coordinates")
        exit_after = row.get("confirmed_global_exit", False)
        if type(exit_after) is not bool:
            raise ValueError("confirmed_global_exit must be a boolean")
        zone = row.get("zone_id")
        if zone is not None and not isinstance(zone, str):
            raise ValueError("zone_id must be a string")
        brightness = row.get("torso_brightness")
        if brightness is not None and not 0 <= float(brightness) <= 1:
            raise ValueError("torso_brightness must be between zero and one")
        bbox = vector(row.get("bbox_xyxy"))
        if bbox is not None and (len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]):
            raise ValueError("bbox_xyxy must be a nonempty rectangle")
        confidence = row.get("confidence")
        if confidence is not None and not 0 <= float(confidence) <= 1:
            raise ValueError("confidence must be between zero and one")
        weight = float(row.get("appearance_weight", 1.0))
        if not 0 <= weight <= 1:
            raise ValueError("appearance_weight must be between zero and one")
        segment = row.get("source_segment", 0)
        if type(segment) is not int or segment < 0:
            raise ValueError("source_segment must be a nonnegative integer")
        return cls(camera, local_id, frame, time, zone, vector(row.get("embedding")),
                   vector(row.get("hsv")), xy, exit_after, None if brightness is None else float(brightness),
                   bbox, None if confidence is None else float(confidence), weight, segment)


@dataclass(frozen=True)
class Tracklet:
    observations: list[Observation]
    reid: tuple | None
    hsv: tuple | None

    @property
    def first(self):
        return self.observations[0]

    @property
    def last(self):
        return self.observations[-1]


def summarize(observations, field):
    vectors = []
    for obs in observations:
        value = getattr(obs, field)
        normalized = unit(value) if value is not None else None
        weight = obs.appearance_weight if field == "reid" else 1.0
        if normalized is not None and weight > 0:
            vectors.append(tuple(x * weight for x in normalized))
    return unit(tuple(map(sum, zip(*vectors, strict=True)))) if vectors else None


def split_tracklets(observations, gap_ms):
    grouped = defaultdict(list)
    for obs in observations:
        grouped[obs.camera, obs.local_id].append(obs)
    tracklets = []
    for group in grouped.values():
        fragment = []
        for obs in sorted(group, key=lambda item: (item.time, item.frame)):
            if fragment and (fragment[-1].exit_after or obs.time - fragment[-1].time > gap_ms
                             or obs.source_segment != fragment[-1].source_segment):
                tracklets.append(Tracklet(fragment, summarize(fragment, "reid"), summarize(fragment, "hsv")))
                fragment = []
            fragment.append(obs)
        tracklets.append(Tracklet(fragment, summarize(fragment, "reid"), summarize(fragment, "hsv")))
    return sorted(tracklets, key=lambda t: (t.first.time, t.first.camera, str(t.first.local_id), t.first.frame))


def synchronized_pairs(a, b, tolerance):
    """Pair each observation at most once, using only the common time window."""
    i = j = 0
    while i < len(a) and j < len(b):
        delta = a[i].time - b[j].time
        if abs(delta) <= tolerance:
            yield a[i], b[j]
            i += 1
            j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1


def appearance(a, b, config):
    limit = config["max_brightness_difference"]
    if limit is not None:
        left = [obs.brightness for obs in a.observations if obs.brightness is not None]
        right = [obs.brightness for obs in b.observations if obs.brightness is not None]
        if left and right and abs(median(left) - median(right)) > limit:
            return -1.0
    if a.reid is not None and b.reid is not None:
        left, right, threshold = a.reid, b.reid, config["reid_min_similarity"]
    elif a.hsv is not None and b.hsv is not None:
        left, right, threshold = a.hsv, b.hsv, config["hsv_min_similarity"]
    else:
        return None
    score = sum(x * y for x, y in zip(left, right, strict=True))
    if score < threshold:
        return -1.0
    if config["contextual_appearance"] and a.reid is not None and b.reid is not None:
        lo, hi = max(a.first.time, b.first.time), min(a.last.time, b.last.time)
        nearby = [[o for o in t.observations if lo <= o.time <= hi and o.reid is not None]
                  for t in (a, b)]
        # One truncated crop must not outweigh the full trajectory's appearance.
        if all(len(rows) >= config["context_min_embeddings"] for rows in nearby):
            u, v = (summarize(rows, "reid") for rows in nearby)
            if u is not None and v is not None:
                score = (score + sum(x * y for x, y in zip(u, v, strict=True))) / 2
    # Context changes candidate ranking, not the calibrated full-track admission
    # threshold. A difficult shared pose must not erase otherwise valid evidence.
    return score


DEFAULTS = {
    "sync_tolerance_ms": 100,
    "min_overlap_pairs": 3,
    "max_local_gap_ms": 750,
    "max_handoff_gap_ms": 400,
    "visibility_gap_ms": 250,
    "reid_min_similarity": 0.75,
    "hsv_min_similarity": 0.90,
    "min_score_margin": 0.05,
    "max_map_distance": None,
    "allow_geometry_only": True,
    "max_brightness_difference": None,
    "overlap_pairs": [],
    "overlap_requires_map": [],
    "continuous_handoffs": [],
    "stitch_local_fragments": False,
    "contextual_appearance": False,
    "resolve_group_conflicts": False,
    "joint_overlap_matching": False,
    "joint_max_edges": 12,
    "context_min_embeddings": 3,
    "stitch_max_gap_ms": 1500,
    "stitch_max_overlap_ms": 100,
    "stitch_motion_window_ms": 600,
    "stitch_min_observations": 4,
    "stitch_min_similarity": 0.50,
    "stitch_max_motion_error": 0.45,
    "stitch_min_margin": 0.08,
    "temporal_evidence_ranking": False,
    "evidence_bin_ms": 200,
    "merge_local_duplicates": False,
    "duplicate_min_pairs": 2,
    "duplicate_min_iou": 0.50,
    "duplicate_max_anchor_error": 0.20,
    "duplicate_min_similarity": 0.65,
    "duplicate_min_margin": 0.08,
}


def upper_anchor(obs):
    """Upper box center is less affected by hidden feet than the box center."""
    x1, y1, x2, y2 = obs.bbox
    return (x1 + x2) / 2, y1 + 0.15 * (y2 - y1)


def predict_anchor(rows, target_time):
    """Robust velocity from sufficiently separated samples, in pixels/ms."""
    velocities = [tuple((q - p) / (b.time - a.time) for p, q in zip(upper_anchor(a), upper_anchor(b)))
                  for i, a in enumerate(rows) for b in rows[i + 1:] if b.time - a.time >= 150]
    if not velocities:
        return None
    velocity = tuple(median(v[k] for v in velocities) for k in (0, 1))
    return tuple(median(upper_anchor(o)[k] + velocity[k] * (target_time - o.time) for o in rows)
                 for k in (0, 1))


def local_candidate(a, b, config):
    if (a.first.camera != b.first.camera or a.last.exit_after
            or a.last.source_segment != b.first.source_segment):
        return None
    gap = b.first.time - a.last.time
    if not -config["stitch_max_overlap_ms"] <= gap <= config["stitch_max_gap_ms"]:
        return None
    if b.last.time <= a.last.time:
        return None
    window = config["stitch_motion_window_ms"]
    left = [o for o in a.observations if o.bbox is not None and o.time >= a.last.time - window]
    right = [o for o in b.observations if o.bbox is not None and o.time <= b.first.time + window]
    if min(len(left), len(right)) < config["stitch_min_observations"]:
        return None
    forward, backward = predict_anchor(left, b.first.time), predict_anchor(right, a.last.time)
    if forward is None or backward is None or a.last.bbox is None or b.first.bbox is None:
        return None
    scale = median(o.bbox[3] - o.bbox[1] for o in left + right)
    error = max(math.dist(forward, upper_anchor(b.first)), math.dist(backward, upper_anchor(a.last))) / scale
    if error > config["stitch_max_motion_error"]:
        return None
    # Very short overlaps are allowed only for duplicate, near-identical upper boxes.
    # Ordinary simultaneous people must remain separate even when dressed alike.
    if gap <= 0:
        pairs = list(synchronized_pairs([o for o in left if o.time >= b.first.time],
                                        [o for o in right if o.time <= a.last.time], 0))
        if not pairs or any(math.dist(upper_anchor(p), upper_anchor(q)) / scale > 0.15
                            or min(p.bbox[2], q.bbox[2]) <= max(p.bbox[0], q.bbox[0]) for p, q in pairs):
            return None
    # Use a small endpoint gallery so changes of pose elsewhere do not dominate.
    endpoint_left = [o for o in a.observations if o.time >= a.last.time - 1000]
    endpoint_right = [o for o in b.observations if o.time <= b.first.time + 1000]
    u, v = summarize(endpoint_left, "reid"), summarize(endpoint_right, "reid")
    if u is None or v is None:
        return None
    similarity = sum(x * y for x, y in zip(u, v, strict=True))
    if similarity < config["stitch_min_similarity"]:
        return None
    brightness = [[o.brightness for o in rows if o.brightness is not None]
                  for rows in (endpoint_left, endpoint_right)]
    limit = config["max_brightness_difference"]
    if limit is not None and all(brightness) and abs(median(brightness[0]) - median(brightness[1])) > limit:
        return None
    return 0.65 * similarity + 0.35 * (1 - error / config["stitch_max_motion_error"])


def stitch_local(tracklets, config):
    """Join mutual-best endpoints; keep ambiguity and confirmed exits as breaks."""
    current = list(tracklets)
    while True:
        edges, outgoing, incoming = [], defaultdict(list), defaultdict(list)
        for i, a in enumerate(current):
            for j in range(i + 1, len(current)):
                b = current[j]
                if b.first.time - a.last.time > config["stitch_max_gap_ms"]:
                    break
                score = local_candidate(a, b, config)
                if score is not None:
                    edge = (score, i, j)
                    edges.append(edge)
                    outgoing[i].append(edge)
                    incoming[j].append(edge)
        for entries in list(outgoing.values()) + list(incoming.values()):
            entries.sort(reverse=True)
        accepted = None
        for edge in sorted(edges, reverse=True):
            score, i, j = edge
            if all(entries[0] == edge and (len(entries) == 1 or score - entries[1][0] >= config["stitch_min_margin"])
                   for entries in (outgoing[i], incoming[j])):
                accepted = i, j
                break
        if accepted is None:
            return current
        i, j = accepted
        observations = sorted(current[i].observations + current[j].observations, key=lambda o: (o.time, o.frame))
        current[i] = Tracklet(observations, summarize(observations, "reid"), summarize(observations, "hsv"))
        current.pop(j)


def candidate(a, b, overlaps, handoffs, config):
    if a.first.camera == b.first.camera or a.last.exit_after and a.last.time < b.last.time:
        return None
    gap = b.first.time - a.last.time
    spatial_limit = config["max_map_distance"]
    distance = None
    if gap <= 0:
        camera_pair = frozenset((a.first.camera, b.first.camera))
        if camera_pair not in overlaps:
            return None
        pairs = list(synchronized_pairs(a.observations, b.observations, config["sync_tolerance_ms"]))
        if len(pairs) < config["min_overlap_pairs"]:
            return None
        distances = [math.dist(p.xy, q.xy) for p, q in pairs if p.xy is not None and q.xy is not None]
        if len(distances) >= config["min_overlap_pairs"]:
            distance = median(distances)
        elif any(camera_pair == frozenset(pair) for pair in config["overlap_requires_map"]):
            return None
        mode = "overlap"
    else:
        key = (a.last.camera, a.last.zone, b.first.camera, b.first.zone)
        if gap > config["max_handoff_gap_ms"] or key not in handoffs:
            return None
        if a.last.xy is not None and b.first.xy is not None:
            distance = math.dist(a.last.xy, b.first.xy)
        mode = "handoff"

    spatial = None
    if spatial_limit is not None and distance is not None:
        if distance > spatial_limit:
            return None
        spatial = 1 - distance / spatial_limit
    visual = appearance(a, b, config)
    if visual == -1.0 or (visual is None and (
        spatial is None or mode == "handoff" or not config["allow_geometry_only"]
    )):
        return None
    scores = [score for score in (visual, spatial) if score is not None]
    score = sum(scores) / len(scores)
    if mode == "overlap" and config["temporal_evidence_ranking"]:
        # Occupied time bins avoid rewarding higher FPS or duplicate boxes.
        # Thresholds still apply to appearance above; this affects ranking only.
        bins = {int((p.time + q.time) / 2 // config["evidence_bin_ms"]) for p, q in pairs}
        score *= math.sqrt(len(bins) * config["evidence_bin_ms"] / 1000)
    return score, mode


def can_merge(left, right, overlaps):
    for a in left:
        for b in right:
            if (a.last.exit_after and b.last.time > a.last.time or
                    b.last.exit_after and a.last.time > b.last.time):
                return False
            if max(a.first.time, b.first.time) <= min(a.last.time, b.last.time):
                cameras = frozenset((a.first.camera, b.first.camera))
                if len(cameras) == 1 or cameras not in overlaps:
                    return False
    return True


def alternatives(edge, entries, key, tracklets, mode):
    if mode == "handoff":
        return entries
    selected = tracklets[edge[2] if edge[3] == key else edge[1]]
    rivals = []
    for item in entries:
        other = tracklets[item[2] if item[3] == key else item[1]]
        if max(other.first.time, selected.first.time) <= min(other.last.time, selected.last.time):
            rivals.append(item)
    return rivals


def joint_overlap_choice(edges, owners, groups, tracklets, overlaps, config):
    """Bounded exact search over small ambiguous components, with abstention.

    An edge is accepted only if excluding it makes the best *whole assignment*
    worse by the configured margin. Tied clothing/trajectory assignments stay split.
    Large components fall back to conservative mutual-best matching.
    """
    unique = {}
    for edge in edges:
        key = tuple(sorted((owners[edge[1]], owners[edge[2]])))
        if key not in unique or edge > unique[key]:
            unique[key] = edge
    pending = list(unique.values())
    while pending:
        component = [pending.pop()]
        nodes = {owners[component[0][1]], owners[component[0][2]]}
        changed = True
        while changed:
            changed = False
            for edge in pending[:]:
                ends = {owners[edge[1]], owners[edge[2]]}
                if ends & nodes:
                    nodes.update(ends)
                    component.append(edge)
                    pending.remove(edge)
                    changed = True
        if len(component) > config["joint_max_edges"]:
            continue
        component.sort(reverse=True)

        def solve(forbidden=None):
            best_score, best_edges = 0.0, []

            def visit(index, partitions, score, selected):
                nonlocal best_score, best_edges
                if index == len(component):
                    if score > best_score:
                        best_score, best_edges = score, selected
                    return
                if score + sum(e[0] for e in component[index:]) < best_score:
                    return
                edge = component[index]
                if edge != forbidden:
                    a, b = owners[edge[1]], owners[edge[2]]
                    left = next(p for p in partitions if a in p)
                    right = next(p for p in partitions if b in p)
                    if left != right and can_merge(
                            [tracklets[k] for node in left for k in groups[node]],
                            [tracklets[k] for node in right for k in groups[node]], overlaps):
                        merged = [p for p in partitions if p != left and p != right] + [left | right]
                        visit(index + 1, merged, score + edge[0], selected + [edge])
                visit(index + 1, partitions, score, selected)

            visit(0, [{node} for node in nodes], 0.0, [])
            return best_score, best_edges

        best_score, selected = solve()
        for edge in selected:
            alternative_score, _ = solve(forbidden=edge)
            if best_score - alternative_score >= config["min_score_margin"]:
                return owners[edge[1]], owners[edge[2]]
    return None


def associate(observations, settings):
    config = DEFAULTS | settings
    if config["max_map_distance"] is not None and not config["max_map_distance"] > 0:
        raise ValueError("max_map_distance must be positive when enabled")
    if config["min_overlap_pairs"] < 1:
        raise ValueError("min_overlap_pairs must be at least one")
    for name in ("stitch_motion_window_ms", "stitch_max_motion_error", "stitch_min_observations", "context_min_embeddings", "evidence_bin_ms"):
        if not math.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if type(config["joint_max_edges"]) is not int or not 1 <= config["joint_max_edges"] <= 16:
        raise ValueError("joint_max_edges must be an integer between 1 and 16")
    overlaps = {frozenset(pair) for pair in config["overlap_pairs"]}
    handoffs = {tuple(route) for route in config["continuous_handoffs"]}
    tracklets = split_tracklets(observations, config["max_local_gap_ms"])
    if config["merge_local_duplicates"]:
        if __package__:
            from .stabilize_tracklets import merge_duplicates
        else:
            from stabilize_tracklets import merge_duplicates
        tracklets = merge_duplicates(tracklets, config)
    if config["stitch_local_fragments"]:
        tracklets = stitch_local(tracklets, config)
    owners = list(range(len(tracklets)))
    groups = {i: [i] for i in owners}
    for mode in ("overlap", "handoff"):
        edges = []
        handoff_edges = {}
        ranked = defaultdict(list)
        for i, a in enumerate(tracklets):
            for j in range(i + 1, len(tracklets)):
                b = tracklets[j]
                gap = b.first.time - a.last.time
                if gap > (0 if mode == "overlap" else config["max_handoff_gap_ms"]):
                    break
                if mode == "handoff" and gap <= 0:
                    continue
                match = candidate(a, b, overlaps, handoffs, config)
                if match is None:
                    continue
                score, _ = match
                key_a = (i, b.first.camera) if mode == "overlap" else (owners[i], "out")
                key_b = (j, a.first.camera) if mode == "overlap" else (owners[j], "in")
                edge = (score, i, j, key_a, key_b)
                if mode == "overlap":
                    edges.append(edge)
                elif owners[i] != owners[j]:
                    route = (owners[i], owners[j])
                    if route not in handoff_edges or edge > handoff_edges[route]:
                        handoff_edges[route] = edge
        edges.extend(handoff_edges.values())
        if config["resolve_group_conflicts"] and mode == "overlap":
            # Rebuild competition after each merge. An edge incompatible with an
            # already established group cannot keep blocking a valid fragment.
            while True:
                feasible = [edge for edge in edges if owners[edge[1]] != owners[edge[2]]
                            and can_merge([tracklets[k] for k in groups[owners[edge[1]]]],
                                          [tracklets[k] for k in groups[owners[edge[2]]]], overlaps)]
                ranked = defaultdict(list)
                for edge in sorted(feasible, reverse=True):
                    ranked[edge[3]].append(edge)
                    ranked[edge[4]].append(edge)
                accepted = None
                for edge in sorted(feasible, reverse=True):
                    score, i, j, key_a, key_b = edge
                    if all(entries[0] == edge and (len(entries) == 1 or
                           score - entries[1][0] >= config["min_score_margin"])
                           for entries in (alternatives(edge, ranked[key_a], key_a, tracklets, mode),
                                           alternatives(edge, ranked[key_b], key_b, tracklets, mode))):
                        accepted = owners[i], owners[j]
                        break
                if accepted is None:
                    if config["joint_overlap_matching"]:
                        accepted = joint_overlap_choice(feasible, owners, groups, tracklets, overlaps, config)
                    if accepted is None:
                        break
                a, b = accepted
                groups[a].extend(groups[b])
                for k in groups.pop(b):
                    owners[k] = a
            continue
        for edge in edges:
            ranked[edge[3]].append(edge)
            ranked[edge[4]].append(edge)
        for entries in ranked.values():
            entries.sort(reverse=True)
        used_in, used_out = set(), set()
        for edge in sorted(edges, reverse=True):
            score, i, j, key_a, key_b = edge
            if any(entries[0] != edge or (len(entries) > 1 and
                   score - entries[1][0] < config["min_score_margin"])
                   for entries in (alternatives(edge, ranked[key_a], key_a, tracklets, mode),
                                   alternatives(edge, ranked[key_b], key_b, tracklets, mode))):
                continue
            a, b = owners[i], owners[j]
            if a == b or (mode == "handoff" and (key_a in used_out or key_b in used_in)):
                continue
            if not can_merge([tracklets[k] for k in groups[a]], [tracklets[k] for k in groups[b]], overlaps):
                continue
            groups[a].extend(groups[b])
            for k in groups.pop(b):
                owners[k] = a
            if mode == "handoff":
                used_out.add(key_a)
                used_in.add(key_b)
    return [[obs for index in indices for obs in tracklets[index].observations]
            for indices in sorted(groups.values(), key=min)]


def export_tracks(groups, output, visibility_gap_ms):
    output.mkdir(parents=True, exist_ok=True)
    tracks = []
    with (output / "assignments.jsonl").open("w", encoding="utf-8") as file:
        for global_id, group in enumerate(groups, 1):
            cameras = defaultdict(list)
            segments = {}
            for obs in sorted(group, key=lambda row: (row.time, row.camera, row.frame)):
                intervals = cameras[obs.camera]
                if (not intervals or obs.time - intervals[-1][1] > visibility_gap_ms
                        or segments.get(obs.camera) != obs.source_segment):
                    intervals.append([obs.time, obs.time])
                else:
                    intervals[-1][1] = obs.time
                segments[obs.camera] = obs.source_segment
                file.write(json.dumps({"camera_id": obs.camera, "local_id": obs.local_id,
                                       "frame_index": obs.frame, "timestamp_ms": obs.time,
                                       "global_id": global_id}, ensure_ascii=False) + "\n")
            tracks.append({"global_id": global_id, "start_ms": min(o.time for o in group),
                           "end_ms": max(o.time for o in group), "cameras": dict(cameras)})
    (output / "tracks.json").write_text(json.dumps({"total_tracks": len(tracks), "tracks": tracks},
                                                    ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "tracks.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["global_id", "start_ms", "end_ms", "cameras"])
        writer.writeheader()
        for track in tracks:
            writer.writerow(track | {"cameras": json.dumps(track["cameras"], ensure_ascii=False)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest_path = args.observations.parent / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("stage") == "local_import" and manifest.get("status") != "complete":
                raise ValueError("Cannot associate an incomplete or failed local import")
        settings = json.loads(args.config.read_text(encoding="utf-8-sig"))
        observations = []
        seen = set()
        with args.observations.open(encoding="utf-8-sig") as file:
            for line_no, line in enumerate(file, 1):
                if not line.strip():
                    continue
                try:
                    obs = Observation.parse(json.loads(line))
                    key = (obs.camera, obs.local_id, obs.frame)
                    if key in seen:
                        raise ValueError("Duplicate camera_id/local_id/frame_index")
                    seen.add(key)
                    observations.append(obs)
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"{args.observations}:{line_no}: {error}") from error
        groups = associate(observations, settings)
        export_tracks(groups, args.output, (DEFAULTS | settings)["visibility_gap_ms"])
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    print(f"Total tracks: {len(groups)}")


if __name__ == "__main__":
    main()
