"""Reconcile distant fragments against the other end of a calibrated corridor."""

from bisect import bisect_left
from collections import defaultdict
from statistics import median

from .associate import split_tracklets, summarize, synchronized_pairs
from .reconnect import similarity


def reconcile_corridor(groups, config):
    zone = config.get("distant_corridor")
    if not zone:
        return groups, [], []
    anchor_camera, detail_camera = zone["cameras"]
    left, top, right, bottom = zone["map_rect"]

    def in_zone(obs):
        return obs.xy is not None and left <= obs.xy[0] <= right and top <= obs.xy[1] <= bottom

    def distant(obs):
        return in_zone(obs) and obs.bbox is not None and obs.bbox[3] <= zone["image_bottom_px"]

    def deltas(detail, anchor):
        return [(b.xy[0] - a.xy[0], b.xy[1] - a.xy[1]) for a, b in synchronized_pairs(detail, anchor, 140)]

    anchors = {}
    offsets = []
    for i, group in enumerate(groups):
        anchor = sorted((o for o in group if o.camera == anchor_camera and in_zone(o)), key=lambda o: o.time)
        detail = sorted((o for o in group if o.camera == detail_camera and distant(o)), key=lambda o: o.time)
        pairs = deltas(detail, anchor)
        if len(pairs) >= 8:
            anchors[i] = anchor, detail, [o.time for o in anchor]
            offsets.append(median(x for x, _ in pairs))
    if len(offsets) < 2:
        return groups, [], []
    offset = median(offsets)

    proposals = defaultdict(list)
    moved = set()
    pending = []
    report = []
    for group in groups:
        if any(o.camera != detail_camera for o in group):
            continue
        for fragment in split_tracklets(group, config["max_local_gap_ms"]):
            observations = fragment.observations
            detail = [o for o in observations if distant(o)]
            if len(detail) < 8 or len(detail) < 0.9 * len(observations):
                continue
            ranked = []
            for ident, (anchor, gallery, _) in anchors.items():
                pairs = deltas(detail, anchor)
                if len(pairs) < max(8, 0.65 * len(detail)):
                    continue
                x_error = median(abs(x - offset) for x, _ in pairs)
                y_error = median(abs(y) for _, y in pairs)
                appearance = summarize(
                    [
                        o
                        for o in gallery
                        if fragment.first.time - 60000 <= o.time <= fragment.last.time + 60000
                    ],
                    "reid",
                )
                look = similarity(fragment.reid, appearance)
                if look is None or look < 0.28 or x_error > 4.5 or y_error > 42:
                    continue
                ranked.append((look - x_error / 25 - y_error / 150, ident))
            ranked.sort(reverse=True)
            if not ranked or (len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 0.12):
                continue
            score, ident = ranked[0]
            anchor, _, times = anchors[ident]
            accepted = 0
            for obs in detail:
                i = bisect_left(times, obs.time)
                reference = min(anchor[max(0, i - 1) : i + 1], key=lambda o: abs(o.time - obs.time))
                if (
                    abs(reference.time - obs.time) <= 180
                    and abs(reference.xy[0] - obs.xy[0] - offset) <= 8
                    and abs(reference.xy[1] - obs.xy[1]) <= 60
                ):
                    proposals[ident, obs.frame].append((score, obs))
                    accepted += 1
                else:
                    pending.append(obs)
            moved.update(detail)
            report.append(
                {
                    "camera": detail_camera,
                    "local_id": fragment.first.local_id,
                    "start_ms": fragment.first.time,
                    "aligned_observations": accepted,
                }
            )

    result = [[o for o in group if o not in moved] for group in groups]
    occupied = {(i, o.frame) for i in anchors for o in result[i] if o.camera == detail_camera}
    for (ident, frame), candidates in proposals.items():
        candidates.sort(key=lambda entry: (entry[0], entry[1].confidence or 0), reverse=True)
        if (ident, frame) not in occupied:
            result[ident].append(candidates.pop(0)[1])
        pending.extend(o for _, o in candidates)
    return [group for group in result if group], pending, report
