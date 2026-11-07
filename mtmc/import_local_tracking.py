"""Bridge misha's per-frame artifacts to our ReID and global association module."""

import argparse
import json
import math
from pathlib import Path

from associate import DEFAULTS, Observation, associate, export_tracks, unit
from local_tracking.artifacts import write_json
from local_tracking.pipeline import file_hash
from local_tracking.validate import validate_output


UPSTREAM_COMMIT = "00853f1242037ed4833eab3dd98cab8604e2d84a"


def artifact_path(root, reference):
    path = (root / reference).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Artifact path escapes source output directory")
    return path


def crop_weight(observation):
    """Bounded quality heuristic, not a probability of identity correctness."""
    quality = observation.get("crop_quality")
    if not quality:
        return 0.0
    width, height = quality["width"], quality["height"]
    sharpness = quality["laplacian_variance"]
    if not all(math.isfinite(x) for x in (width, height, sharpness)) or min(width, height) <= 0:
        raise ValueError("Invalid crop quality measurements")
    if height < 40 or width < 16:
        return 0.0
    return (observation["detection_confidence"] * min(1.0, math.sqrt(width * height) / 128)
            * min(1.0, max(0.0, sharpness) / 100) * (0.5 if quality["truncated"] else 1.0))


def polygon_area(points):
    if len(points) < 3:
        return 0.0
    return abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:] + points[:1]))) / 2


def convert_observation(frame, observation, manifest, root, segment):
    if observation["local_track_id"] is None:
        raise ValueError("Unconfirmed detections must not become a shared null track")
    common_time = frame["timestamp_sec"]
    if not math.isfinite(common_time) or common_time < 0:
        raise ValueError("Common timestamps must be nonnegative; use one shared epoch for all cameras")
    gap = frame.get("source_gap_before_sec", 0.0)
    if not math.isfinite(gap) or gap < 0:
        raise ValueError("source_gap_before_sec must be finite and nonnegative")
    polygons = observation["silhouette_polygons"]
    row = {
        "recording_id": frame["recording_id"], "camera_id": frame["camera_id"],
        "frame_index": frame["frame_index"], "timestamp_ms": round(common_time * 1000),
        "local_id": observation["local_track_id"], "observation_id": observation["observation_id"],
        "bbox_xyxy": observation["bbox_xyxy"], "confidence": observation["detection_confidence"],
        "contour": max(polygons, key=polygon_area) if polygons else [],
        "silhouette_polygons": polygons,
        "video_time_sec": frame["video_time_sec"], "media_time_sec": frame["media_time_sec"],
        "timestamp_source": frame["timestamp_source"], "timestamp_sec": common_time,
        "source_segment": segment, "source_gap_before_ms": round(gap * 1000),
        "artifact_root": str(root.resolve()),
        **{key: observation.get(key) for key in (
            "mask_ref", "mask_origin_px", "mask_size", "crop_ref", "crop_quality", "border_flags",
            "footpoint_px", "footpoint_method", "touches_frame_bottom")},
    }
    # Recover source PTS only when the exported media clock and stream time base exist.
    time_base = manifest["source"].get("time_base")
    if frame["timestamp_source"] == "pts" and frame["media_time_sec"] is not None and time_base:
        row["source_pts"] = round(frame["media_time_sec"] / time_base)
        row["source_time_base_sec"] = time_base
    for key in ("map_xy", "zone_id", "confirmed_global_exit"):
        if key in observation:
            row[key] = observation[key]
    Observation.parse(row)
    return row


class IntelEncoder:
    def __init__(self, weights):
        import openvino as ov

        for path in (weights, weights.with_suffix(".bin")):
            if not path.is_file():
                raise FileNotFoundError(path)
        self.compiled = ov.Core().compile_model(str(weights), "CPU", {"INFERENCE_NUM_THREADS": 4})
        self.provenance = {"name": "person-reidentification-retail-0288", "device": "CPU",
                           "xml_sha256": file_hash(weights), "bin_sha256": file_hash(weights.with_suffix(".bin"))}

    def __call__(self, crop):
        import cv2
        import numpy as np

        tensor = cv2.resize(crop, (128, 256)).transpose(2, 0, 1)[None].astype(np.float32)
        return self.compiled([tensor])[self.compiled.output(0)].reshape(-1).tolist()


def add_appearance(row, source, root, encoder, min_quality):
    import cv2
    import numpy as np

    if not source.get("crop_ref"):
        return "no_crop"
    weight = crop_weight(source)
    if weight <= 0 or weight < min_quality:
        return "rejected_crop"
    crop = cv2.imdecode(np.fromfile(artifact_path(root, source["crop_ref"]), dtype=np.uint8), cv2.IMREAD_COLOR)
    if crop is None:
        raise ValueError("Unreadable ReID crop")
    embedding = encoder(crop)
    if not embedding or not all(math.isfinite(x) for x in embedding):
        raise ValueError("ReID returned an empty or nonfinite vector")
    normalized = unit(embedding)
    if normalized is None:
        raise ValueError("ReID returned a zero vector")
    row["embedding"] = list(normalized)
    row["appearance_weight"] = weight
    # Match the existing masked torso-brightness feature; never use the background.
    if source.get("mask_ref"):
        mask = cv2.imdecode(np.fromfile(artifact_path(root, source["mask_ref"]), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != crop.shape[:2]:
            raise ValueError("Crop and mask ROI sizes differ")
        top, bottom = int(crop.shape[0] * 0.22), int(crop.shape[0] * 0.52)
        pixels = crop[top:bottom][mask[top:bottom] != 0]
        if len(pixels):
            row["torso_brightness"] = float(np.median(pixels.max(axis=1))) / 255
    return "embeddings"


def import_outputs(inputs, output, encoder, settings=None, min_quality=0.05, run_association=False):
    settings = settings or {}
    if not 0 <= min_quality <= 1:
        raise ValueError("min_quality must be between zero and one")
    if not inputs:
        raise ValueError("At least one camera output is required")
    sources, seen_cameras, recordings, origins = [], set(), set(), set()
    for input_path in inputs:
        root = Path(input_path).resolve()
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != "1.0":
            raise ValueError("Unsupported local-tracking schema")
        anchors = manifest.get("clock_anchors", [])
        if anchors:
            if len(anchors) < 2:
                raise ValueError("Clock mapping needs at least two anchors")
            previous = (-math.inf, -math.inf)
            for anchor in anchors:
                pair = (anchor["video_time_sec"], anchor["timestamp_sec"])
                if not all(math.isfinite(x) and x > old for x, old in zip(pair, previous)):
                    raise ValueError("Clock anchors must increase on both axes")
                previous = pair
        validation = validate_output(root)
        camera = manifest["camera_id"]
        if camera in seen_cameras:
            raise ValueError("Multiple runs for one camera cannot be merged using their local IDs")
        seen_cameras.add(camera)
        recordings.add(manifest["recording_id"])
        origins.add(manifest.get("timeline_origin"))
        sources.append((root, manifest, validation))
    if len(recordings) != 1 or len(origins) != 1:
        raise ValueError("Inputs must share recording_id and timeline_origin")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {"stage": "local_import", "status": "running", "upstream_commit": UPSTREAM_COMMIT,
              "recording_id": next(iter(recordings)), "timeline_origin": next(iter(origins)),
              "min_crop_quality": min_quality, "settings": settings,
              "reid": getattr(encoder, "provenance", {"name": "injected_encoder"}),
              "cameras": {}, "timestamp_validation": "Structural validation only; physical alignment not certified"}
    write_json(output / "manifest.json", report)
    try:
        with ((output / "observations.jsonl").open("x", encoding="utf-8") as observations,
              (output / "frames.jsonl").open("x", encoding="utf-8") as frames_out,
              (output / "unassigned.jsonl").open("x", encoding="utf-8") as unassigned):
            for root, manifest, validation in sources:
                camera = manifest["camera_id"]
                stats = {"frames": 0, "observations": 0, "unassigned": 0, "excluded": 0,
                         "source_gaps": 0, "embeddings": 0, "rejected_crop": 0, "no_crop": 0}
                report["cameras"][camera] = {"input": str(root), "validation": validation, "stats": stats,
                    "frames_sha256": file_hash(root / "frames.jsonl"),
                    "source_sha256": manifest.get("source_sha256"),
                    "clock_anchors": manifest.get("clock_anchors", []),
                    "time_offset_sec": manifest["time_offset_sec"]}
                segment = 0
                with (root / "frames.jsonl").open(encoding="utf-8") as stream:
                    for line in stream:
                        frame = json.loads(line)
                        stats["frames"] += 1
                        gap = frame.get("source_gap_before_sec", 0)
                        if not math.isfinite(gap) or gap < 0:
                            raise ValueError("Invalid source gap metadata")
                        if gap > 0:
                            segment += 1
                            stats["source_gaps"] += 1
                        frame_info = {k: v for k, v in frame.items() if k != "observations"}
                        frame_info.update(source_segment=segment, observation_ids=[], unassigned_ids=[], excluded_ids=[])
                        for source in frame["observations"]:
                            x1, y1, x2, y2 = source["bbox_xyxy"]
                            foot_x = (x1 + x2) / 2
                            if any(left <= foot_x <= right and top <= y2 <= bottom
                                   for left, top, right, bottom in settings.get("excluded_image_rects", {}).get(camera, [])):
                                stats["excluded"] += 1
                                frame_info["excluded_ids"].append(source["observation_id"])
                                continue
                            if source["local_track_id"] is None:
                                stats["unassigned"] += 1
                                frame_info["unassigned_ids"].append(source["observation_id"])
                                unassigned.write(json.dumps({"camera_id": camera, "recording_id": frame["recording_id"],
                                    "frame_index": frame["frame_index"], "timestamp_sec": frame["timestamp_sec"],
                                    "artifact_root": str(root), "observation": source}, allow_nan=False) + "\n")
                                continue
                            row = convert_observation(frame, source, manifest, root, segment)
                            stats[add_appearance(row, source, root, encoder, min_quality)] += 1
                            observations.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                            stats["observations"] += 1
                            frame_info["observation_ids"].append(source["observation_id"])
                        frames_out.write(json.dumps(frame_info, separators=(",", ":"), allow_nan=False) + "\n")
        if run_association:
            with (output / "observations.jsonl").open(encoding="utf-8") as stream:
                groups = associate([Observation.parse(json.loads(line)) for line in stream], settings)
            export_tracks(groups, output, (DEFAULTS | settings)["visibility_gap_ms"])
            report["total_tracks"] = len(groups)
        report["status"] = "complete"
        write_json(output / "manifest.json", report)
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        write_json(output / "manifest.json", report)
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True, help="Completed camera output directory")
    parser.add_argument("--output", type=Path, required=True, help="New directory, never overwrite")
    parser.add_argument("--reid", type=Path, default=Path(__file__).with_name("weights") / "person-reidentification-retail-0288.xml")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("camera_graph.improved.json"))
    parser.add_argument("--min-crop-quality", type=float, default=0.05)
    parser.add_argument("--associate", action="store_true")
    args = parser.parse_args()
    try:
        settings = json.loads(args.config.read_text(encoding="utf-8"))
        result = import_outputs(args.input, args.output, IntelEncoder(args.reid), settings,
                                args.min_crop_quality, args.associate)
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
