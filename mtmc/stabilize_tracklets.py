"""Conservative repair of duplicate local detections.

All observations are retained. No person labels are used.
"""

import math

if __package__:
    from .associate import Tracklet, summarize, synchronized_pairs, upper_anchor
else:
    from associate import Tracklet, summarize, synchronized_pairs, upper_anchor


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def make_tracklet(rows):
    rows = sorted(rows, key=lambda o: (o.time, o.frame, str(o.local_id)))
    return Tracklet(rows, summarize(rows, "reid"), summarize(rows, "hsv"))


def intersection_over_union(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0


def duplicate_score(a, b, config):
    if a.first.camera != b.first.camera or a.first.source_segment != b.first.source_segment:
        return None
    if any(o.exit_after for o in a.observations + b.observations):
        return None
    pairs = list(synchronized_pairs(a.observations, b.observations, 0))
    if len(pairs) < config["duplicate_min_pairs"] or any(not p.bbox or not q.bbox for p, q in pairs):
        return None
    for p, q in pairs:
        scale = min(p.bbox[3] - p.bbox[1], q.bbox[3] - q.bbox[1])
        if (intersection_over_union(p.bbox, q.bbox) < config["duplicate_min_iou"]
                or math.dist(upper_anchor(p), upper_anchor(q)) / scale > config["duplicate_max_anchor_error"]):
            return None
    lo, hi = max(a.first.time, b.first.time), min(a.last.time, b.last.time)
    features = [summarize([o for o in t.observations if lo - 400 <= o.time <= hi + 400], "reid")
                for t in (a, b)]
    if any(v is None for v in features):
        return None
    score = cosine(*features)
    return score if score >= config["duplicate_min_similarity"] else None


def merge_duplicates(tracklets, config):
    current = list(tracklets)
    while True:
        edges = []
        for i, a in enumerate(current):
            for j in range(i + 1, len(current)):
                score = duplicate_score(a, current[j], config)
                if score is not None:
                    edges.append((score, i, j))
        accepted = None
        for score, i, j in sorted(edges, reverse=True):
            rivals = [s for s, k, l in edges if (k, l) != (i, j) and ({i, j} & {k, l})]
            if not rivals or score - max(rivals) >= config["duplicate_min_margin"]:
                accepted = i, j
                break
        if accepted is None:
            return sorted(current, key=lambda t: (t.first.time, t.first.camera, str(t.first.local_id)))
        i, j = accepted
        current[i] = make_tracklet(current[i].observations + current[j].observations)
        current.pop(j)
