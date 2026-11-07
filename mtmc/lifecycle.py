"""Confirm identities from observed runs and reconcile fragments in shared time."""

import math
from collections import defaultdict
from itertools import combinations
from statistics import median

import cv2
import numpy as np

from .associate import local_candidate, split_tracklets, synchronized_pairs, upper_anchor
from .doorways import restore_door_returns
from .reconnect import aligned_points, return_score, similarity


def visibility_runs(group, gap_ms):
    runs = []
    for obs in sorted(group, key=lambda o: (o.time, o.camera, o.frame)):
        if not runs or obs.time - runs[-1][-1].time > gap_ms:
            runs.append([])
        runs[-1].append(obs)
    return runs


def supported_contour(obs, row, distant):
    if obs.bbox is None or (obs.confidence or 0) < 0.60 or row.get("footpoint_status") != "observed":
        return False
    if any((row.get("border_flags") or {}).values()) or not row.get("silhouette_polygons"):
        return False
    x1, y1, x2, y2 = obs.bbox
    return 0.14 * (y2 - y1) <= x2 - x1 <= (0.85 if distant else 1.12) * (y2 - y1)


def mask_duplicate(a, b, details):
    if a.bbox is None or b.bbox is None:
        return False
    height = min(a.bbox[3] - a.bbox[1], b.bbox[3] - b.bbox[1])
    if math.dist(upper_anchor(a), upper_anchor(b)) > 0.14 * height:
        return False
    polygons = [details[o].get("silhouette_polygons") or [] for o in (a, b)]
    if not all(polygons):
        return False
    left, top = min(a.bbox[0], b.bbox[0]), min(a.bbox[1], b.bbox[1])
    width, height = max(a.bbox[2], b.bbox[2]) - left, max(a.bbox[3], b.bbox[3]) - top
    if width <= 0 or height <= 0:
        return False
    masks = []
    for shapes in polygons:
        mask = np.zeros((65, 65), np.uint8)
        outlines = [
            np.rint((np.asarray(poly) - (left, top)) * (64 / width, 64 / height)).astype(np.int32)
            for poly in shapes
        ]
        cv2.fillPoly(mask, outlines, 1)
        masks.append(mask)
    overlap = np.count_nonzero(masks[0] & masks[1])
    return overlap >= 0.62 * np.count_nonzero(masks[0] | masks[1]) and overlap >= 12


def world_overlap(a, b, tolerance):
    pairs = aligned_points(a, b, tolerance)
    look = similarity(a.reid, b.reid)
    if look is None or look < 0.46 or len(pairs) < 5 or pairs[-1][0].time - pairs[0][0].time < 440:
        return None
    distances = [d for _, _, d in pairs]
    typical = median(distances)
    if typical > 13 or sum(d <= 19 for d in distances) < 0.8 * len(distances):
        return None
    p, q, _ = pairs[0]
    r, s, _ = pairs[-1]
    u, v = np.subtract(r.xy, p.xy), np.subtract(s.xy, q.xy)
    lengths = [np.linalg.norm(w) for w in (u, v)]
    if min(lengths) >= 7 and np.dot(u, v) < 0.5 * math.prod(lengths):
        return None
    if min(lengths) < 7 and (look < 0.60 or typical > 9):
        return None
    brightness = [[o.brightness for o in t.observations if o.brightness is not None] for t in (a, b)]
    if all(brightness) and abs(median(brightness[0]) - median(brightness[1])) > 0.28:
        return None
    return 1 + look - typical / 35, "synchronized"


def corridor_motion(a, b, tolerance, corridor):
    if not corridor or {a.first.camera, b.first.camera} - set(corridor["motion_cameras"]):
        return None
    left, top, right, bottom = corridor["motion_rect"]
    pairs = [
        (p, q, d)
        for p, q, d in aligned_points(a, b, tolerance)
        if all(left <= o.xy[0] <= right and top <= o.xy[1] <= bottom for o in (p, q))
    ]
    if len(pairs) < 9 or pairs[-1][0].time - pairs[0][0].time < 950:
        return None
    distances = [d for _, _, d in pairs]
    lateral = median(abs(p.xy[0] - q.xy[0]) for p, q, _ in pairs)
    if lateral > 8:
        return None
    first_a, first_b, _ = pairs[0]
    last_a, last_b, _ = pairs[-1]
    u, v = np.subtract(last_a.xy, first_a.xy), np.subtract(last_b.xy, first_b.xy)
    lengths = [np.linalg.norm(w) for w in (u, v)]
    if min(lengths) < 14 or min(lengths) < 0.40 * max(lengths) or np.dot(u, v) < 0.95 * math.prod(lengths):
        return None
    typical = median(distances)
    look = similarity(a.reid, b.reid)
    if look is None or look < 0.14:
        return None
    if typical <= 8 and len(pairs) >= 12 and look >= 0.26:
        return 1.26 + look / 4 - typical / 90, "corridor_motion", pairs[0][0].time, pairs[-1][0].time
    if typical <= 17 and min(lengths) >= 20:
        return 1.05 + look / 4 - typical / 100, "corridor_witness", pairs[0][0].time, pairs[-1][0].time
    if typical > 34 or look < 0.34 or lateral > 5 or min(distances) > 14 or min(lengths) < 60:
        return None
    # Opposite fisheye views have a longitudinal bias. Require the tracks
    # to agree through several independent portions of the actual walk.
    direction = np.sign(u[1])
    chunks = np.array_split(np.arange(len(pairs)), 4)
    agrees = 0
    for chunk in chunks:
        p, q, _ = pairs[chunk[0]]
        r, s, _ = pairs[chunk[-1]]
        first = direction * (r.xy[1] - p.xy[1])
        second = direction * (s.xy[1] - q.xy[1])
        if min(first, second) >= 8 and min(first, second) >= 0.45 * max(first, second):
            agrees += 1
    if agrees >= 3:
        return 1.16 + look / 3 - typical / 160, "corridor_motion", pairs[0][0].time, pairs[-1][0].time
    return None


def cross_handoff(a, b, details):
    gap = b.first.time - a.last.time
    if not 0 < gap <= 800 or a.last.exit_after:
        return None
    look = similarity(a.reid, b.reid)
    if look is None or look < 0.54:
        return None
    samples = []
    for track, end in ((a, True), (b, False)):
        boundary = track.last if end else track.first
        points = [
            o
            for o in track.observations
            if abs(o.time - boundary.time) <= 500
            and o.xy is not None
            and details[o].get("footpoint_status") == "observed"
        ]
        if len(points) < 3:
            return None
        samples.append(points)
    left, right = samples
    u = np.subtract(left[-1].xy, left[0].xy)
    v = np.subtract(right[-1].xy, right[0].xy)
    lengths = [np.linalg.norm(w) for w in (u, v)]
    distance = math.dist(left[-1].xy, right[0].xy)
    if min(lengths) < 4 or np.dot(u, v) < 0.7 * math.prod(lengths) or distance > 10:
        return None
    displacement = np.subtract(right[0].xy, left[-1].xy)
    if distance > 4 and np.dot(u, displacement) < -0.1 * lengths[0] * distance:
        return None
    return 0.85 + look - distance / 35, "handoff"


def evidence(a, b, overlaps, details, config):
    if b.first.time < a.first.time:
        a, b = b, a
    if a.first.camera == b.first.camera:
        gap = b.first.time - a.last.time
        if (
            -150 <= gap <= 0
            and not a.last.exit_after
            and a.last.source_segment == b.first.source_segment
            and a.last.time < b.last.time
            and min(len(a.observations), len(b.observations)) >= 4
        ):
            shared = list(synchronized_pairs(a.observations, b.observations, 0))
            look = similarity(a.reid, b.reid)
            if (
                1 <= len(shared) <= 2
                and look is not None
                and look >= 0.68
                and all(p.frame == q.frame and mask_duplicate(p, q, details) for p, q in shared)
            ):
                return 1.35 + 0.3 * look, "same_mask_boundary"
        score = local_candidate(a, b, config)
        if score is not None and score >= 0.59:
            return 0.8 + score, "local_motion"
        score = return_score(a, b, config["reappearance_memory_ms"])
        if score is not None and score >= 0.79:
            return 0.64 + score, "local_return"
        return None
    if frozenset((a.first.camera, b.first.camera)) not in overlaps:
        return None
    if b.first.time <= a.last.time:
        return world_overlap(a, b, config["sync_tolerance_ms"])
    return cross_handoff(a, b, details)


def conflicts(left, right, overlaps, details, tolerance):
    duplicates = set()
    for a in left:
        for b in right:
            if max(a.first.time, b.first.time) > min(a.last.time, b.last.time):
                continue
            if a.first.camera == b.first.camera:
                pairs = list(synchronized_pairs(a.observations, b.observations, 0))
                if not pairs:
                    return None
                for p, q in pairs:
                    if p.frame != q.frame or not mask_duplicate(p, q, details):
                        return None

                    def rank(o):
                        x1, y1, x2, y2 = o.bbox
                        return (o.confidence or 0) * (x2 - x1) * (y2 - y1)

                    duplicates.add(min((p, q), key=rank))
            elif frozenset((a.first.camera, b.first.camera)) not in overlaps:
                return None
            else:
                ds = [
                    d
                    for p, q, d in aligned_points(a, b, tolerance)
                    if all(details[o].get("footpoint_status") == "observed" for o in (p, q))
                ]
                if len(ds) >= 4 and median(ds) > 35 and min(ds) > 23:
                    return None
    return duplicates


def review_lifecycle(groups, details, config):
    policy = config.get("lifecycle")
    if not policy:
        return groups, [], {}
    motion_config = config | {
        "stitch_min_observations": 3,
        "stitch_max_overlap_ms": 150,
        "stitch_max_motion_error": 0.34,
        "stitch_min_similarity": 0.57,
        "stitch_max_gap_ms": 1600,
    }
    runs = {i: split_tracklets(group, 480) for i, group in enumerate(groups)}
    bounds = {i: (parts[0].first.time, max(t.last.time for t in parts)) for i, parts in runs.items()}
    overlaps = {frozenset(pair) for pair in config["overlap_pairs"]}
    tolerance = config["sync_tolerance_ms"]
    edges = []
    corridor = config.get("distant_corridor")
    for i, j in combinations(runs, 2):
        start = max(bounds[i][0], bounds[j][0])
        end = min(bounds[i][1], bounds[j][1])
        if start - end > config["reappearance_memory_ms"]:
            continue
        best = None
        witnesses = []
        for a in runs[i]:
            for b in runs[j]:
                match = evidence(a, b, overlaps, details, motion_config)
                if match is not None and (best is None or match[0] > best[0]):
                    best = (*match, a.first, b.first)
                if frozenset((a.first.camera, b.first.camera)) not in overlaps:
                    continue
                movement = corridor_motion(a, b, tolerance, corridor)
                if movement is None:
                    continue
                score, mode, begin, finish = movement
                if mode == "corridor_witness":
                    witnesses.append((score, a.first, b.first, begin, finish))
                elif best is None or score > best[0]:
                    best = score, mode, a.first, b.first
        for a, b in combinations(witnesses, 2):
            if len({a[1].camera, a[2].camera, b[1].camera, b[2].camera}) < 3:
                continue
            if min(a[4], b[4]) - max(a[3], b[3]) < 700:
                continue
            score = 0.35 + (a[0] + b[0]) / 2
            if best is None or score > best[0]:
                best = score, "two_view_witness", a[1], a[2]
        if best is not None:
            edges.append((best[0], i, j, *best[1:]))

    owners = list(range(len(groups)))
    current = dict(enumerate(groups))
    pending = []
    links = []
    while edges:
        ranked = defaultdict(list)
        viable = {}
        for edge in sorted(edges, key=lambda e: (-e[0], e[1], e[2])):
            _, i, j, *_ = edge
            a, b = sorted((owners[i], owners[j]))
            if a == b or (a, b) in viable:
                continue
            excluded = conflicts(runs[a], runs[b], overlaps, details, tolerance)
            if excluded is None:
                continue
            viable[a, b] = edge, excluded
            ranked[a].append((a, b))
            ranked[b].append((a, b))
        joined = False
        for (a, b), (edge, excluded) in viable.items():
            score, _, _, mode, first, second = edge
            if any(
                ranked[k][0] != (a, b) or (len(ranked[k]) > 1 and score - viable[ranked[k][1]][0][0] < 0.10)
                for k in (a, b)
            ):
                continue
            current[a] = [o for o in current[a] + current.pop(b) if o not in excluded]
            runs[a] = split_tracklets(current[a], 480)
            owners = [a if owner == b else owner for owner in owners]
            pending.extend(sorted(excluded, key=lambda o: (o.time, o.camera, o.frame)))
            links.append(
                {
                    "mode": mode,
                    "first": [first.camera, first.local_id, first.time],
                    "second": [second.camera, second.local_id, second.time],
                    "duplicate_observations": len(excluded),
                }
            )
            joined = True
        if not joined:
            break

    restored, door_links = restore_door_returns(list(current.values()), config)
    confirmed, tentative, isolated = [], [], []
    for group in restored:
        cameras = {o.camera for o in group}
        periods = visibility_runs(group, config["visibility_gap_ms"])
        span = max(run[-1].time - run[0].time for run in periods)
        heights = [o.bbox[3] - o.bbox[1] for o in group if o.bbox]
        single = len(cameras) == 1
        distant = bool(heights and median(heights) < policy["far_height_px"])
        minimum = policy["far_ms"] if distant and single else policy["min_ms"]
        support = sum(supported_contour(o, details[o], distant) for o in group) if single else None
        required = policy["far_support"] if distant else policy["near_support"]
        reason = None
        if span < minimum:
            reason = "short_distant" if distant and single else "short_unconfirmed"
        elif support is not None and support < required:
            reason = "weak_distant_contour" if distant else "weak_single_view_contour"
        if reason:
            pending.extend(group)
            tentative.append(
                {
                    "first_ms": min(o.time for o in group),
                    "last_ms": max(o.time for o in group),
                    "cameras": sorted(cameras),
                    "observations": len(group),
                    "visible_ms": span,
                    "support_observations": support,
                    "reason": reason,
                }
            )
        else:
            kept = []
            for run in periods:
                threshold = policy.get("isolated_ms", 0)
                if (
                    threshold
                    and run[-1].time - run[0].time <= threshold
                    and len({o.camera for o in run}) == 1
                ):
                    pending.extend(run)
                    isolated.append(
                        {
                            "camera": run[0].camera,
                            "first_ms": run[0].time,
                            "last_ms": run[-1].time,
                            "observations": len(run),
                        }
                    )
                else:
                    kept.extend(run)
            confirmed.append(kept)
    return (
        confirmed,
        pending,
        {"links": links, "door_returns": door_links, "tentative": tentative, "isolated": isolated},
    )
