"""Compare two association exports of the same detections and index every old ID."""

import argparse
import csv
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import median

from PIL import Image

from local_tracking.artifacts import write_json
from mtmc.associate import Observation, synchronized_pairs
from tools.export_web_session import load_calibration, observation_key, project, read_json, rows


def load_assignments(path):
    result = {}
    for item in rows(path / "assignments.jsonl"):
        key = observation_key(item)
        if key in result:
            raise ValueError(f"Repeated observation: {key}")
        result[key] = item["global_id"]
    return result


def track_statistics(root, assignments):
    tracks = read_json(root / "tracks.json")["tracks"]
    cameras = defaultdict(Counter)
    for track in tracks:
        names = track["cameras"]
        span = track["end_ms"] - track["start_ms"]
        for name in names:
            cameras[name]["all"] += 1
            cameras[name]["single_camera"] += len(names) == 1
            cameras[name]["at_most_1_sec"] += span <= 1000
    keys = [(camera, frame, ident) for (camera, _, frame), ident in assignments.items() if ident is not None]
    times = defaultdict(list)
    for item in rows(root / "assignments.jsonl"):
        if item["global_id"] is not None:
            times[item["global_id"]].append(item["timestamp_ms"])
    flashes = 0
    for stamps in times.values():
        stamps.sort()
        start = last = stamps[0]
        for timestamp in stamps[1:]:
            if timestamp - last > 350:
                flashes += last - start <= 500
                start = timestamp
            last = timestamp
        flashes += last - start <= 500
    return {
        "tracks": len(tracks),
        "single_camera_tracks": sum(len(track["cameras"]) == 1 for track in tracks),
        "at_most_1_sec": sum(track["end_ms"] - track["start_ms"] <= 1000 for track in tracks),
        "at_most_2_sec": sum(track["end_ms"] - track["start_ms"] <= 2000 for track in tracks),
        "visibility_runs_at_most_half_sec": flashes,
        "confirmed_observations": len(keys),
        "unconfirmed_observations": len(assignments) - len(keys),
        "repeated_ids_in_same_camera_frame": len(keys) - len(set(keys)),
        "by_camera": dict(sorted(cameras.items())),
    }


def geometry_statistics(observations, assignments, calibrations, size):
    groups = defaultdict(lambda: defaultdict(list))
    for row in observations:
        ident = assignments[observation_key(row)]
        if ident is None or row.get("footpoint_status") != "observed":
            continue
        camera = row["camera_id"]
        point = project(row, calibrations[camera], size)
        if point:
            groups[ident][camera].append(Observation.parse(row | {"map_xy": (point["x"], point["y"])}))
    distances = defaultdict(list)
    for group in groups.values():
        for a, b in combinations(sorted(group), 2):
            left, right = [sorted(group[c], key=lambda o: o.time) for c in (a, b)]
            distances[f"{a}/{b}"].extend(
                math.dist(p.xy, q.xy) for p, q in synchronized_pairs(left, right, 150)
            )
    return {
        pair: {
            "paired_points": len(values),
            "median_px": round(median(values), 2),
            "over_24_px": sum(d > 24 for d in values),
        }
        for pair, values in sorted(distances.items())
        if values
    }


def evidence_summary(observations):
    heights = [o["bbox_xyxy"][3] - o["bbox_xyxy"][1] for o in observations]
    clipped = sum(any((o.get("border_flags") or {}).values()) for o in observations)
    estimated = sum(o.get("footpoint_status") != "observed" for o in observations)
    height = round(median(heights), 1)
    confidence = round(median(o["confidence"] for o in observations), 3)
    return height, confidence, clipped, estimated


def audit(before, after, output, plan, calibration_dir):
    old, new = load_assignments(before), load_assignments(after)
    if old.keys() != new.keys():
        raise ValueError("The two runs must contain the same source observations")
    old_tracks = read_json(before / "tracks.json")["tracks"]
    mapping = defaultdict(Counter)
    contributors = defaultdict(set)
    for key, ident in old.items():
        next_id = new[key]
        mapping[ident][next_id] += 1
        if ident is not None and next_id is not None:
            contributors[next_id].add(ident)
    manifest = read_json(after / "manifest.json")
    with Image.open(plan) as image:
        size = image.size
    calibrations = {
        camera: load_calibration(
            calibration_dir, camera, read_json(Path(info["input"]) / "manifest.json")["source"], size
        )
        for camera, info in manifest["cameras"].items()
    }
    observations = list(rows(after / "observations.jsonl"))
    if len(observations) != len(new) or {observation_key(o) for o in observations} != new.keys():
        raise ValueError("Observation catalog disagrees with assignments")
    old_evidence = defaultdict(list)
    for row in observations:
        ident = old[observation_key(row)]
        if ident is not None:
            old_evidence[ident].append(row)
    output.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with (output / "track_dispositions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "old_id",
                "start_ms",
                "end_ms",
                "cameras",
                "new_id:observations",
                "pending_observations",
                "status",
                "median_height_px",
                "median_confidence",
                "clipped_observations",
                "no_observed_floor",
            ]
        )
        for track in old_tracks:
            ident = track["global_id"]
            buckets = mapping[ident]
            targets = {key: buckets[key] for key in sorted(key for key in buckets if key is not None)}
            if not targets:
                status = "withheld"
            elif len(targets) > 1:
                status = "split"
            elif len(contributors[next(iter(targets))]) > 1:
                status = "joined"
            else:
                status = "retained"
            counts[status] += 1
            writer.writerow(
                [
                    ident,
                    track["start_ms"],
                    track["end_ms"],
                    ",".join(track["cameras"]),
                    ",".join(f"{key}:{value}" for key, value in targets.items()),
                    buckets[None],
                    status,
                    *evidence_summary(old_evidence[ident]),
                ]
            )
    report = {
        "before": track_statistics(before, old),
        "after": track_statistics(after, new),
        "old_track_dispositions": counts,
        "before_projection_agreement": geometry_statistics(observations, old, calibrations, size),
        "after_projection_agreement": geometry_statistics(observations, new, calibrations, size),
        "lifecycle": manifest["identity_consistency"].get("lifecycle", {}),
    }
    write_json(output / "audit.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--plan", type=Path, default=Path("office.jpg"))
    parser.add_argument("--calibrations", type=Path, default=Path("."))
    args = parser.parse_args()
    audit(args.before, args.after, args.output, args.plan, args.calibrations)
