"""Offline two-camera refinement with independently validated image geometry.

The homography maps upper-body anchors between views, NOT floor/map positions.
It is fitted only from long mutual-best ReID matches and validated on held-out
times. Ambiguous short detections remain explicitly unconfirmed.
"""
import argparse
from collections import defaultdict
import json
import hashlib
import math
from pathlib import Path
from statistics import median

import cv2
import numpy as np
from lap import lapjv

from associate import (DEFAULTS, Observation, Tracklet, associate, export_tracks, appearance,
                       candidate, can_merge, split_tracklets, stitch_local, summarize,
                       synchronized_pairs, upper_anchor)
from dataclasses import replace
from stabilize_tracklets import merge_duplicates


def key(o):
    return o.camera, o.local_id, o.frame


def local_tracks(observations, cfg):
    ts = split_tracklets(observations, cfg["max_local_gap_ms"])
    if cfg["merge_local_duplicates"]:
        ts = merge_duplicates(ts, cfg)
    return stitch_local(ts, cfg) if cfg["stitch_local_fragments"] else ts


def fit_geometry(reference, target, cfg):
    edges = []
    for i, a in enumerate(reference):
        for j, b in enumerate(target):
            pairs = list(synchronized_pairs(a.observations, b.observations, 100))
            if len(pairs) < 40 or min(a.last.time, b.last.time) - max(a.first.time, b.first.time) < 8000:
                continue
            score = appearance(a, b, cfg)
            if score is not None and score >= .50:
                edges.append((score * math.sqrt(len(pairs)), i, j, pairs))
    seeds = []
    for score, i, j, pairs in sorted(edges, key=lambda e: e[0], reverse=True):
        rivals = [s for s, k, l, _ in edges if (k, l) != (i, j) and (k == i or l == j)]
        if not rivals or score - max(rivals) >= .3:
            seeds.append((i, j, pairs))
    if len(seeds) < 2:
        return None, {}, {"accepted": False, "reason": "fewer_than_two_long_independent_matches"}
    train, validation = [], []
    for i, j, pairs in seeds:
        # Whole one-second bins held out; neighboring duplicate detections do
        # not appear on opposite sides of the validation split.
        for p, q in pairs:
            if p.bbox and q.bbox:
                (validation if p.time // 1000 % 4 == 0 else train).append((p, q))
    if len(train) < 40 or len(validation) < 12:
        return None, {}, {"accepted": False, "reason": "insufficient_geometry_samples"}
    src = np.float32([upper_anchor(p) for p, q in train])
    dst = np.float32([upper_anchor(q) for p, q in train])
    scale = median(q.bbox[3] - q.bbox[1] for p, q in train)
    cv2.setRNGSeed(0)
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, .15 * scale)
    if homography is None or not np.isfinite(homography).all():
        return None, {}, {"accepted": False, "reason": "degenerate_fit"}
    src_test = np.float32([upper_anchor(p) for p, q in validation])
    predicted = cv2.perspectiveTransform(src_test[:, None], homography)[:, 0]
    actual = np.float32([upper_anchor(q) for p, q in validation])
    errors = np.linalg.norm(predicted - actual, axis=1)
    normalized = errors / np.array([q.bbox[3] - q.bbox[1] for p, q in validation])
    report = {"accepted": bool(np.median(normalized) < .15 and np.mean(normalized < .35) >= .65),
              "train_pairs": len(train), "validation_pairs": len(validation),
              "train_inlier_fraction": float(mask.mean()),
              "validation_median_box_heights": float(np.median(normalized)),
              "validation_fraction_within_035_box_heights": float(np.mean(normalized < .35)),
              "matrix": homography.tolist(),
              "seed_track_pairs": [[i, j] for i, j, _ in seeds]}
    return (homography, {j: i for i, j, _ in seeds}, report) if report["accepted"] else (None, {}, report)


def recent_appearance(track, time):
    rows = [o for o in track.observations if abs(o.time - time) <= 600 and o.reid]
    if not rows:
        return None, None
    values = [o.brightness for o in rows if o.brightness is not None]
    return summarize(rows, "reid"), median(values) if values else None


def geometry_assign(reference, target, h, seeds, cfg):
    frames = defaultdict(dict)
    for j, track in enumerate(target):
        for o in track.observations:
            # Duplicate boxes are retained for export, but contribute one vote.
            previous = frames[o.frame].get(j)
            if previous is None or (o.confidence or 0) > (previous.confidence or 0):
                frames[o.frame][j] = o
    histories = defaultdict(list)
    previous_state = {}
    reference_times = [np.array([o.time for o in t.observations]) for t in reference]
    for frame, current in sorted(frames.items()):
        time = min(o.time for o in current.values())
        locked = {seeds[j] for j in current if j in seeds}
        for j in current:
            if j in seeds:
                histories[j].append((frame, seeds[j], "long_reid_seed"))
                previous_state[j] = seeds[j]
        active = [j for j in current if j not in seeds and len(target[j].observations) >= 5
                  and target[j].last.time - target[j].first.time >= 350]
        available = [i for i in range(len(reference)) if i not in locked]
        if not active:
            continue
        costs = np.full((len(active), len(available) + len(active)), 1.25)
        for row, j in enumerate(active):
            obs = current[j]
            local_vector, local_brightness = recent_appearance(target[j], time)
            for col, i in enumerate(available):
                rt = reference[i]
                idx = int(np.argmin(abs(reference_times[i] - time)))
                nearest = rt.observations[idx]
                if abs(nearest.time - time) > cfg["sync_tolerance_ms"]:
                    # Local continuity covers a temporary loss or the end of
                    # reference-camera visibility. Never creates a new match.
                    if previous_state.get(j) == i:
                        costs[row, col] = .70
                    continue
                if not obs.bbox or not nearest.bbox:
                    continue
                prediction = cv2.perspectiveTransform(np.float32([[upper_anchor(nearest)]]), h)[0, 0]
                error = float(np.linalg.norm(prediction - upper_anchor(obs))) / (obs.bbox[3] - obs.bbox[1])
                if error > 1.0:
                    if previous_state.get(j) == i:
                        costs[row, col] = 1.10
                    continue
                rv, rb = recent_appearance(rt, time)
                if local_brightness is not None and rb is not None and abs(local_brightness - rb) > .32:
                    continue
                visual = sum(a * b for a, b in zip(local_vector, rv, strict=True)) if local_vector and rv else .3
                costs[row, col] = error + .25 * (1 - visual)
                if j in previous_state and previous_state[j] != i:
                    costs[row, col] += .18
        _, ci, _ = lapjv(costs, extend_cost=True)
        ri = range(len(active))
        for row, col in zip(ri, ci):
            j = active[row]
            state = available[col] if col < len(available) and costs[row, col] < 1.20 else None
            histories[j].append((frame, state, "geometry_and_continuity" if state is not None else "ambiguous"))
            if state is not None:
                previous_state[j] = state

    assignments, reasons = {}, {}
    # Offline interpolation only within a local track. A change to a different
    # person is left unresolved at the boundary rather than hidden by smoothing.
    for j, track in enumerate(target):
        history = histories[j]
        known = [(f, s) for f, s, _ in history if s is not None]
        frame_states = {f: s for f, s, _ in history}
        for o in track.observations:
            state = frame_states.get(o.frame)
            reason = "matched" if state is not None else "unconfirmed"
            if state is None and known and len(track.observations) >= 5:
                before = [(f, s) for f, s in known if f < o.frame]
                after = [(f, s) for f, s in known if f > o.frame]
                if not before:
                    state, reason = after[0][1], "backfilled_local_continuity"
                elif not after:
                    state, reason = before[-1][1], "forward_local_continuity"
                elif before[-1][1] == after[0][1]:
                    state, reason = before[-1][1], "interpolated_local_continuity"
            assignments[key(o)] = state
            reasons[key(o)] = reason
        if j not in seeds:
            states = [(o, assignments[key(o)]) for o in track.observations
                      if assignments[key(o)] is not None]
            boundaries = [(a.time + b.time) / 2 for (a, sa), (b, sb) in zip(states, states[1:]) if sa != sb]
            for o in track.observations:
                if any(abs(o.time - boundary) <= 180 for boundary in boundaries):
                    assignments[key(o)] = None
                    reasons[key(o)] = "identity_transition_uncertain"
    # Enforce frame exclusivity after interpolation, allowing only duplicate
    # detections already merged by the geometric/appearance duplicate check.
    for frame, current in frames.items():
        used = defaultdict(list)
        for j, o in current.items():
            state = assignments[key(o)]
            if state is not None:
                used[state].append(j)
        for state, tracks in used.items():
            if len(tracks) <= 1:
                continue
            keep = next((j for j in tracks if j in seeds), None)
            for j in tracks:
                if j != keep:
                    for o in target[j].observations:
                        if o.frame == frame:
                            assignments[key(o)] = None
                            reasons[key(o)] = "same_camera_conflict"
    return assignments, reasons


def reid_fallback(observations, rows, cfg):
    groups = associate(observations, cfg)
    lookup = {key(o): r for o, r in zip(observations, rows)}
    # OSNet is an independent rescue for entire still-unmatched single-camera
    # groups. It cannot override an established Intel match or local collision.
    remaining = [g for g in groups if len({o.camera for o in g}) == 1]
    matched = [g for g in groups if len({o.camera for o in g}) > 1]
    secondary = []
    for group in remaining:
        converted = []
        for o in sorted(group, key=lambda o: (o.time, o.frame)):
            row = lookup[key(o)]
            converted.append(replace(o, reid=tuple(row["embedding_osnet"]) if row.get("embedding_osnet") else None, hsv=None))
        secondary.append(Tracklet(converted, summarize(converted, "reid"), None))
    secondary_cfg = cfg | {"reid_min_similarity": .65, "stitch_min_similarity": .80,
                           "duplicate_min_similarity": .80}
    overlaps = {frozenset(p) for p in cfg["overlap_pairs"]}
    edges = []
    for i, a in enumerate(secondary):
        for j in range(i + 1, len(secondary)):
            b = secondary[j]
            if min(sum(o.reid is not None for o in t.observations) for t in (a, b)) < 3:
                continue
            left, right = sorted((a, b), key=lambda t: t.first.time)
            match = candidate(left, right, overlaps, set(), secondary_cfg)
            if match and can_merge([a], [b], overlaps):
                edges.append((match[0], i, j))
    used = set()
    for score, i, j in sorted(edges, reverse=True):
        rivals = [s for s, k, l in edges if (k, l) != (i, j) and ({i, j} & {k, l})]
        if not ({i, j} & used) and (not rivals or score - max(rivals) >= cfg["min_score_margin"]):
            matched.append(remaining[i] + remaining[j])
            used.update((i, j))
    return matched + [group for i, group in enumerate(remaining) if i not in used]


def refine(rows, cfg, reference_camera="cam_1"):
    obs = [Observation.parse(r) for r in rows]
    cameras = sorted({o.camera for o in obs})
    if any(o.exit_after for o in obs) or any(len({o.source_segment for o in obs if o.camera == c}) > 1 for c in cameras):
        return reid_fallback(obs, rows, cfg), [], {"method": "reid", "reason": "explicit_exit_or_source_discontinuity"}
    if len(cameras) != 2 or reference_camera not in cameras:
        return reid_fallback(obs, rows, cfg), [], {"method": "reid", "reason": "two_camera_geometry_not_available"}
    ts = local_tracks(obs, cfg)
    reference = [t for t in ts if t.first.camera == reference_camera]
    target = [t for t in ts if t.first.camera != reference_camera]
    h, seeds, report = fit_geometry(reference, target, cfg)
    if h is None:
        return reid_fallback(obs, rows, cfg), [], {"method": "reid_with_osnet_rescue", "geometry": report}
    assignment, reasons = geometry_assign(reference, target, h, seeds, cfg)
    groups = [list(t.observations) for t in reference]
    pending = []
    for t in target:
        if not any(assignment[key(o)] is not None for o in t.observations) and len(t.observations) >= 5:
            # A persistent person never matched to the reference camera remains
            # a real independent track; do not suppress it to improve the count.
            groups.append(list(t.observations))
            continue
        for o in t.observations:
            state = assignment[key(o)]
            if state is None:
                pending.append((o, reasons[key(o)]))
            else:
                groups[state].append(o)
    return groups, pending, {"method": "validated_geometry_with_reid_seeds", "geometry": report,
                             "unconfirmed_observations": len(pending)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--enriched", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output already exists")
    cfg = DEFAULTS | json.loads(args.config.read_text(encoding="utf-8"))
    demo = json.loads(args.demo.read_text(encoding="utf-8"))
    if demo.get("status") != "complete":
        raise ValueError("Cannot refine an incomplete demo")
    args.output.mkdir(parents=True)
    reports = []
    for case in demo["cases"]:
        source = Path(case["global_output"])
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        if manifest["status"] != "complete":
            raise ValueError("Incomplete local import")
        path = args.enriched / case["name"] / "observations.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        originals = [json.loads(line) for line in (source / "observations.jsonl").read_text(encoding="utf-8").splitlines()]
        if [{k: v for k, v in r.items() if k != "embedding_osnet"} for r in rows] != originals:
            raise ValueError("Enriched observations do not match the original import")
        groups, pending, report = refine(rows, cfg)
        all_keys = [key(o) for g in groups for o in g] + [key(o) for o, _ in pending]
        assert len(all_keys) == len(set(all_keys)) == len(rows), "Lost or duplicated observations"
        target = args.output / case["name"]
        export_tracks(groups, target, cfg["visibility_gap_ms"])
        with (target / "assignments.jsonl").open("a", encoding="utf-8") as out:
            for o, reason in pending:
                out.write(json.dumps({"camera_id": o.camera, "local_id": o.local_id, "frame_index": o.frame,
                                      "timestamp_ms": o.time, "global_id": None, "status": "unconfirmed",
                                      "reason": reason}) + "\n")
        tracks_path = target / "tracks.json"
        track_export = json.loads(tracks_path.read_text(encoding="utf-8"))
        track_export.update(unconfirmed_observations=len(pending), observations_total=len(rows))
        tracks_path.write_text(json.dumps(track_export, indent=2) + "\n", encoding="utf-8")
        (target / "observations.jsonl").write_bytes(path.read_bytes())
        manifest.update(settings=cfg, refinement=report,
                        input_observations_sha256=hashlib.sha256((source / "observations.jsonl").read_bytes()).hexdigest(),
                        enriched_observations_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        secondary_reid={"name": "osnet_x0_25_msmt17", "role": "unmatched_group_rescue",
                                        "threshold": .65, "minimum_embeddings_per_group": 3})
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        report.update(case=case["name"], total_tracks=len(groups), observations=len(rows), pending=len(pending))
        reports.append(report)
        case.update(global_output=str(target.resolve()), total_tracks=len(groups), pending_observations=len(pending))
        print(case["name"], len(groups), "tracks", len(pending), "pending", report["method"], flush=True)
    (args.output / "demo.json").write_text(json.dumps(demo, indent=2) + "\n", encoding="utf-8")
    (args.output / "report.json").write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
