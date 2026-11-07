"""Recover fragmented identities using synchronized floor trajectories and recent appearance."""

import math
from collections import defaultdict
from itertools import combinations
from statistics import median

from .associate import split_tracklets, synchronized_pairs


def similarity(left, right):
    if left is None or right is None:
        return None
    return sum(a * b for a, b in zip(left, right, strict=True))


def aligned_points(left, right, tolerance):
    start = max(left.first.time, right.first.time)
    end = min(left.last.time, right.last.time)
    return [
        (a, b, math.dist(a.xy, b.xy))
        for a, b in synchronized_pairs(left.observations, right.observations, tolerance)
        if a.xy is not None and b.xy is not None and start <= a.time <= end and start <= b.time <= end
    ]


def trajectory_score(left, right, pairs):
    look = similarity(left.reid, right.reid)
    if look is None or look < 0.34 or len(pairs) < 8:
        return None
    distances = [distance for _, _, distance in pairs]
    typical = median(distances)
    close = [a.time for a, _, distance in pairs if distance <= 16]
    if len(close) < 7 or close[-1] - close[0] < 650:
        return None
    if typical <= 12:
        return 1 + look - typical / 40
    if typical > 32 or look < 0.45:
        return None
    # Opposite views can disagree on depth. Require sustained motion in the
    # same direction as well as a section where the two projections agree.
    agrees = 0
    for i in range(0, len(pairs) - 5, 6):
        a, b, _ = pairs[i]
        c, d, _ = pairs[i + 5]
        if c.time - a.time > 1200:
            continue
        u = tuple(y - x for x, y in zip(a.xy, c.xy, strict=True))
        v = tuple(y - x for x, y in zip(b.xy, d.xy, strict=True))
        lengths = math.hypot(*u), math.hypot(*v)
        if min(lengths) >= 6 and sum(x * y for x, y in zip(u, v, strict=True)) >= 0.8 * math.prod(lengths):
            agrees += 1
    if agrees >= 3:
        return 1 + look - typical / 60
    return None


def uncertain_tail(target, reference, pairs):
    """Abstain when one view switches appearance and moves away from the match."""
    samples = [o for o in target.observations if o.brightness is not None]
    if len(samples) < 3:
        return []
    previous, before, after = samples[-3:]
    if (
        target.last.time - after.time > 700
        or after.time - before.time > 900
        or before.time - previous.time > 1000
        or abs(before.brightness - previous.brightness) > 0.15
        or abs(after.brightness - before.brightness) < 0.45
    ):
        return []
    anchors = [o for o in reference.observations if o.brightness is not None]
    if not anchors:
        return []
    old, new = [min(anchors, key=lambda o: abs(o.time - t.time)) for t in (before, after)]
    if (
        max(abs(old.time - before.time), abs(new.time - after.time)) > 400
        or abs(old.brightness - new.brightness) > 0.16
        or abs(before.brightness - old.brightness) > 0.28
        or abs(after.brightness - new.brightness) < 0.42
    ):
        return []
    a, b, _ = min(pairs, key=lambda pair: abs(pair[0].time - before.time))
    c, d, _ = min(pairs, key=lambda pair: abs(pair[0].time - after.time))
    if max(abs(a.time - before.time), abs(c.time - after.time)) > 240:
        return []
    u = tuple(y - x for x, y in zip(a.xy, c.xy, strict=True))
    v = tuple(y - x for x, y in zip(b.xy, d.xy, strict=True))
    if min(math.hypot(*u), math.hypot(*v)) < 6 or sum(x * y for x, y in zip(u, v, strict=True)) >= 0:
        return []
    boundary = (before.time + after.time) / 2
    return [o for o in target.observations if o.time > boundary]


def trajectory_link(left, right, tolerance):
    points = aligned_points(left, right, tolerance)
    score = trajectory_score(left, right, points)
    if score is None:
        return None
    uncertain = set(
        uncertain_tail(left, right, points)
        + uncertain_tail(right, left, [(b, a, distance) for a, b, distance in points])
    )
    if uncertain:
        points = [(a, b, d) for a, b, d in points if a not in uncertain and b not in uncertain]
        score = trajectory_score(left, right, points)
        if score is None:
            return None
    a, b, _ = min(points, key=lambda point: point[2])
    return score, a, b, uncertain


def recent_appearance(left, right, memory_ms):
    # Keep views separately. A front crop and a back crop should not average
    # into one template. Require repeated evidence for the long return path.
    before = [o.reid for o in left.observations if o.reid and left.last.time - o.time <= memory_ms][-12:]
    after = [o.reid for o in right.observations if o.reid and o.time - right.first.time <= 3500][:8]
    if min(len(before), len(after)) < 2:
        return None
    strengths = sorted((max(similarity(a, b) for a in before) for b in after), reverse=True)
    return strengths[1]


def return_score(left, right, memory_ms):
    if left.first.camera != right.first.camera:
        return None
    if right.first.time < left.first.time:
        left, right = right, left
    gap = right.first.time - left.last.time
    if not 0 < gap <= memory_ms or left.last.exit_after:
        return None
    if left.last.xy is None or right.first.xy is None:
        return None
    distance = math.dist(left.last.xy, right.first.xy)
    look = similarity(left.reid, right.reid)
    if look is None:
        return None
    if gap <= 5000:
        if distance <= 8 and look >= 0.68:
            return look - distance / 80
        repeated = all(sum(o.reid is not None for o in t.observations) >= 2 for t in (left, right))
        same_local_id = (
            left.first.local_id == right.first.local_id
            and left.last.source_segment == right.first.source_segment
        )
        if gap <= 2500 and distance <= 22 and look >= 0.72 and (repeated or same_local_id):
            return look - distance / 80
        if gap <= 200 and distance <= 15 and look >= 0.76:
            return look - distance / 80
        return None
    if distance > 6 or look < 0.82:
        return None
    gallery = recent_appearance(left, right, memory_ms)
    if gallery is not None and gallery >= 0.88:
        return (look + gallery) / 2 - distance / 60
    return None


def overlap_conflicts(left, right, overlaps, tolerance, corridor):
    uncertain = set()
    for a in left:
        for b in right:
            if max(a.first.time, b.first.time) > min(a.last.time, b.last.time):
                continue
            pair = frozenset((a.first.camera, b.first.camera))
            if pair not in overlaps:
                return None
            points = aligned_points(a, b, tolerance)
            distances = [d for _, _, d in points]
            if len(distances) >= 5 and min(distances) > 30 and median(distances) > 45:
                if not corridor or pair != frozenset(corridor["cameras"]):
                    return None
                detail = a if a.first.camera == corridor["cameras"][1] else b
                if (
                    detail.last.time - detail.first.time > 1500
                    or any(
                        o.bbox is None or o.bbox[3] > corridor["image_bottom_px"] for o in detail.observations
                    )
                    or median(abs(p.xy[0] - q.xy[0]) for p, q, _ in points) > 8
                ):
                    return None
                uncertain.update(detail.observations)
    return uncertain


def best_link(left, right, mode, overlaps, tolerance, memory):
    best = None
    directions = set()
    for a in left:
        for b in right:
            if mode == "trajectory":
                if frozenset((a.first.camera, b.first.camera)) not in overlaps:
                    continue
                link = trajectory_link(a, b, tolerance)
                if link is None:
                    continue
            else:
                score = return_score(a, b, memory)
                if score is None:
                    continue
                earlier, later = sorted((a, b), key=lambda fragment: fragment.first.time)
                directions.add(earlier is a)
                link = score, earlier.last, later.first, set()
            if best is None or link[0] > best[0]:
                best = link
    if best is not None and len(directions) == 2:
        # A fragment that fits both sides of a visibility gap is less
        # ambiguous than a single nearby endpoint.
        score, first, second, uncertain = best
        return score + 0.15, first, second, uncertain
    return best


def reconnect(groups, config):
    memory = config.get("reappearance_memory_ms", 0)
    if not memory:
        return groups, [], []
    # A disconnected island attached through a switched local ID must earn
    # a link again before it can block another, simultaneously visible group.
    islands = []
    for group in groups:
        part = []
        for obs in sorted(group, key=lambda o: (o.time, o.camera, o.frame)):
            if part and obs.time - part[-1].time >= 2000:
                islands.append(part)
                part = []
            part.append(obs)
        islands.append(part)

    overlaps = {frozenset(pair) for pair in config["overlap_pairs"]}
    tolerance = min(160, config["sync_tolerance_ms"])
    links = []
    pending = []
    for mode in ("trajectory", "return"):
        fragments = {
            i: split_tracklets(group, config["visibility_gap_ms"]) for i, group in enumerate(islands) if group
        }
        bounds = {
            i: (items[0].first.time, max(item.last.time for item in items)) for i, items in fragments.items()
        }
        ranked = defaultdict(list)
        for (i, left), (j, right) in combinations(fragments.items(), 2):
            start = max(bounds[i][0], bounds[j][0])
            end = min(bounds[i][1], bounds[j][1])
            if start - end > (memory if mode == "return" else 0):
                continue
            best = best_link(left, right, mode, overlaps, tolerance, memory)
            if best is None:
                continue
            corridor = config.get("distant_corridor") if mode == "trajectory" else None
            conflicts = overlap_conflicts(left, right, overlaps, tolerance, corridor)
            if conflicts is None or best[1] in conflicts or best[2] in conflicts:
                continue
            edge = (best[0], i, j)
            evidence = best[1], best[2], best[3] | conflicts
            ranked[i].append((edge, evidence))
            ranked[j].append((edge, evidence))
        for candidates in ranked.values():
            candidates.sort(key=lambda entry: entry[0], reverse=True)
        for i, candidates in ranked.items():
            edge, (first, second, uncertain) = candidates[0]
            score, a, b = edge
            if i != a or ranked[b][0][0] != edge:
                continue
            if any(len(ranked[k]) > 1 and score - ranked[k][1][0][0] < 0.12 for k in (a, b)):
                continue
            islands[a] = [o for o in islands[a] + islands[b] if o not in uncertain]
            islands[b] = []
            pending.extend(sorted(uncertain, key=lambda o: (o.time, o.camera, o.frame)))
            links.append(
                {
                    "mode": mode,
                    "score": round(score, 3),
                    "evidence": [
                        {"camera": o.camera, "local_id": o.local_id, "time_ms": o.time}
                        for o in (first, second)
                    ],
                }
            )
    return [group for group in islands if group], links, pending
