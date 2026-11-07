"""Remove short local fragments and reconcile duplicate IDs using overlapping boxes."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from associate import unit


def iou(a, b):
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area - intersection) if intersection else 0


def repair(rows, min_frames, min_iou, min_similarity):
    cameras = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        cameras[row["camera_id"]][row["local_id"]][row["frame_index"]] = row
    result = []
    for camera, grouped in cameras.items():
        tracks = [(local_id, frames) for local_id, frames in grouped.items() if len(frames) >= min_frames]
        for index, (local_id, frames) in enumerate(tracks):
            if not frames:
                continue
            for _, other in tracks[index + 1:]:
                common = frames.keys() & other.keys()
                if len(common) < 3 or any(iou(frames[t]["bbox_xyxy"], other[t]["bbox_xyxy"]) < min_iou
                                          for t in common):
                    continue
                if any(row.get("confirmed_global_exit", False) for row in [*frames.values(), *other.values()]):
                    continue
                # Use only the neighborhood of the overlap to avoid unrelated later poses.
                lo, hi = min(common) - 8, max(common) + 8
                vectors = []
                for fragment in (frames, other):
                    features = [unit(row["embedding"]) for t, row in fragment.items()
                                if lo <= t <= hi and row.get("embedding")]
                    features = [value for value in features if value is not None]
                    vectors.append(unit(tuple(map(sum, zip(*features, strict=True)))) if features else None)
                if None in vectors or sum(x * y for x, y in zip(*vectors, strict=True)) < min_similarity:
                    continue
                for frame, row in other.items():
                    if frame not in frames or row["confidence"] > frames[frame]["confidence"]:
                        frames[frame] = row
                other.clear()
            for row in frames.values():
                original_id = row["local_id"]
                result.append(row | {"local_id": local_id, "source_local_id": original_id})
    return sorted(result, key=lambda row: (row["timestamp_ms"], row["camera_id"], row["frame_index"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--min-frames", type=int, default=5)
    parser.add_argument("--min-iou", type=float, default=0.38)
    parser.add_argument("--min-similarity", type=float, default=0.44)
    args = parser.parse_args()
    with args.input.open(encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]
    corrected = repair(rows, args.min_frames, args.min_iou, args.min_similarity)
    with args.output.open("x", encoding="utf-8") as file:
        for row in corrected:
            file.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"Observations: {len(rows)} -> {len(corrected)}")


if __name__ == "__main__":
    main()
