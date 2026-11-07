"""Audit serialized outputs and referenced image files before handing them off."""

import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .config import CameraConfig


def validate_output(root: Path) -> dict:
    root = root.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError(f"Incomplete run: {manifest['status']}")
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    tracks = json.loads((root / "local_tracks.json").read_text(encoding="utf-8"))["tracks"]
    camera = CameraConfig(
        manifest["camera_id"],
        Path(manifest["source"]["path"]),
        manifest["time_offset_sec"],
        manifest.get("clock_anchors", []),
    )
    counts = Counter()
    previous_index, previous_time = -1, -math.inf
    frame_count = observation_count = empty_count = 0

    def require(condition, message):
        if not condition:
            raise ValueError(f"{root.name}: {message}")

    def image_ref(ref):
        path = (root / ref).resolve()
        require(path.is_relative_to(root), "Artifact path escapes output directory")
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        require(image is not None, f"Unreadable image: {ref}")
        return image

    with (root / "frames.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            frame = json.loads(line)
            require(frame["camera_id"] == manifest["camera_id"], "Camera identity mismatch")
            require(frame["recording_id"] == manifest["recording_id"], "Recording identity mismatch")
            require(frame["frame_index"] > previous_index, "Frame indices must increase")
            require(
                math.isfinite(frame["timestamp_sec"]) and frame["timestamp_sec"] > previous_time,
                "Timestamps must be finite and increase",
            )
            require(
                abs(frame["timestamp_sec"] - camera.common_time(frame["video_time_sec"])) < 1e-7,
                "Clock alignment mismatch",
            )
            previous_index, previous_time = frame["frame_index"], frame["timestamp_sec"]
            width, height = frame["frame_width"], frame["frame_height"]
            require(
                (width, height) == (manifest["source"]["width"], manifest["source"]["height"]),
                "Source resolution mismatch",
            )
            frame_count += 1
            empty_count += not frame["observations"]
            ids = set()
            for obs in frame["observations"]:
                observation_count += 1
                x1, y1, x2, y2 = obs["bbox_xyxy"]
                require(0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height, "Invalid source box")
                require(0 <= obs["detection_confidence"] <= 1, "Invalid confidence")
                local_id = obs["local_track_id"]
                if local_id is not None:
                    require(
                        type(local_id) is int and local_id > 0 and local_id not in ids,
                        "Invalid or duplicate local ID in a frame",
                    )
                    ids.add(local_id)
                    counts[local_id] += 1
                if obs["mask_ref"]:
                    mask = image_ref(obs["mask_ref"])
                    require(
                        mask.ndim == 2 and list(mask.shape[::-1]) == obs["mask_size"], "Mask size mismatch"
                    )
                    require(np.isin(mask, [0, 255]).all(), "Mask must be binary")
                    ox, oy = obs["mask_origin_px"]
                    mw, mh = obs["mask_size"]
                    require(
                        ox >= 0 and oy >= 0 and ox + mw <= width and oy + mh <= height,
                        "Mask outside source image",
                    )
                if obs["crop_ref"]:
                    crop = image_ref(obs["crop_ref"])
                    require(
                        crop.shape[1] == obs["crop_quality"]["width"]
                        and crop.shape[0] == obs["crop_quality"]["height"],
                        "Crop size mismatch",
                    )
    require(frame_count == metrics["processed_frames"] > 0, "Processed frame count mismatch")
    require(observation_count == metrics["observations"], "Observation count mismatch")
    require(empty_count == metrics["empty_frames"], "Empty frame count mismatch")
    require(len(tracks) == len(counts) == metrics["local_track_count"], "Local track count mismatch")
    require(
        {t["local_track_id"]: t["observation_count"] for t in tracks} == dict(counts),
        "Local summary does not match observations",
    )
    for track in tracks:
        for selected in track["selected_crops"]:
            image_ref(selected["crop_ref"])
    return {
        "camera_id": manifest["camera_id"],
        "frames": frame_count,
        "observations": observation_count,
        "local_tracks": len(tracks),
        "valid": True,
    }
