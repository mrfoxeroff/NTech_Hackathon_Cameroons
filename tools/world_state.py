"""Use the same clock buckets for the plan, source frames and state API."""

from collections import defaultdict
from math import dist
from statistics import median

STEP_MS = 120


def position(sources, previous):
    projected = [source for source in sources if source["point"] is not None]
    if not projected:
        return None
    observed = [source for source in projected if source["point"]["status"] == "observed"]
    choices = observed or projected

    def coordinates(source):
        point = source["point"]
        return point["x"], point["y"]

    if len(choices) == 1:
        source = choices[0]
        if previous is not None and previous["camera_id"] != source["camera_id"]:
            if dist(coordinates(previous), coordinates(source)) > 24:
                return None
        return source

    agreement = [
        [other for other in choices if dist(coordinates(source), coordinates(other)) <= 14]
        for source in choices
    ]
    largest = max(map(len, agreement))
    if largest == 1:
        if previous is None:
            return None
        old = coordinates(previous)
        ordered = sorted(choices, key=lambda source: dist(old, coordinates(source)))
        if dist(old, coordinates(ordered[0])) <= 13 and (
            dist(old, coordinates(ordered[1])) - dist(old, coordinates(ordered[0])) >= 5
        ):
            return ordered[0]
        return None
    options = [group for group in agreement if len(group) == largest]
    if len({frozenset(source["camera_id"] for source in group) for group in options}) > 1:
        return None
    cluster = options[0]
    representative = max(cluster, key=lambda source: source["confidence"])
    point = representative["point"] | {
        key: round(median(source["point"][key] for source in cluster), 2) for key in ("x", "y")
    }
    return representative | {"point": point}


def build_world_state(observations):
    """Return position samples per identity and sparse simultaneous state rows."""
    tracks = defaultdict(dict)
    buckets = defaultdict(list)
    previous = {}
    for (ident, bucket), cameras in sorted(observations.items()):
        sources = list(cameras.values())
        before = previous.get(ident)
        if before is not None and bucket - before[0] > 3:
            before = None
        result = position(sources, before[1] if before else None)
        point = result["point"] if result else None
        if result:
            previous[ident] = bucket, result
            tracks[ident][bucket] = point | {
                "t": result["timestamp_sec"],
                "cam": result["camera_id"],
                "segment": result["source_segment"],
            }
        buckets[bucket].append({"id": ident, "position": point, "sources": sources})
    return tracks, {"schema_version": "1.0", "step_ms": STEP_MS, "buckets": buckets}
