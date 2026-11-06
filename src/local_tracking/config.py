"""Validated, relocatable configuration shared by CLI and Python callers."""

import math
import re
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path

import yaml


def safe_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError(f"Invalid identifier {value!r}; use letters, digits, underscore or hyphen")
    return value


@dataclass
class CameraConfig:
    camera_id: str
    video: Path
    time_offset_sec: float = 0.0
    clock_anchors: list[dict] = field(default_factory=list)
    scene_regions: list[dict] = field(default_factory=list)
    occluder_polygons: list = field(default_factory=list)

    def common_time(self, video_time_sec: float) -> float:
        """Interpolate measured OSD clock anchors, extrapolate at boundaries."""
        if not self.clock_anchors:
            return video_time_sec + self.time_offset_sec
        index = bisect_right([a["video_time_sec"] for a in self.clock_anchors], video_time_sec) - 1
        index = max(0, min(index, len(self.clock_anchors) - 2))
        left, right = self.clock_anchors[index : index + 2]
        slope = (right["timestamp_sec"] - left["timestamp_sec"]) / (
            right["video_time_sec"] - left["video_time_sec"]
        )
        return (
            left["timestamp_sec"] + (video_time_sec - left["video_time_sec"]) * slope + self.time_offset_sec
        )


@dataclass
class Settings:
    recording_id: str
    output_root: Path
    model: Path
    cameras: list[CameraConfig]
    timeline_origin: str | None = None
    device: str = "auto"
    imgsz: int = 640
    fp16: bool = True
    confidence: float = 0.1
    iou: float = 0.7
    stride: int = 1
    lost_track_sec: float = 1.0
    crop_interval_sec: float = 0.5
    selected_crops_per_track: int = 5
    save_masks: bool = True
    save_preview: bool = True
    preview_max_width: int = 1280
    tracker: dict = field(default_factory=dict)

    def validate(self):
        safe_id(self.recording_id)
        if not isinstance(self.device, str) or not re.fullmatch(r"auto|cpu|(?:cuda:)?\d+", self.device):
            raise ValueError("device must be auto, cpu, 0, 1, ... or cuda:0, ...")
        for key in ("fp16", "save_masks", "save_preview"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be a YAML boolean")
        if not self.cameras:
            raise ValueError("At least one camera is required")
        ids = [safe_id(c.camera_id) for c in self.cameras]
        if len(set(ids)) != len(ids):
            raise ValueError("Camera identifiers must be unique")
        for camera in self.cameras:
            for points in camera.occluder_polygons:
                if len(points) < 3 or not all(
                    len(p) == 2 and all(math.isfinite(v) and 0 <= v <= 1 for v in p) for p in points
                ):
                    raise ValueError("Occluder polygons require normalized coordinates")
            for region in camera.scene_regions:
                if region.get("mode") not in ("screen", "reflection", "static_object") or not region.get(
                    "name"
                ):
                    raise ValueError("Scene region requires name and valid mode")
                points = region.get("polygon", [])
                if len(points) < 3 or not all(
                    len(p) == 2 and all(math.isfinite(v) and 0 <= v <= 1 for v in p) for p in points
                ):
                    raise ValueError("Scene polygons require normalized coordinates in [0, 1]")
                if region["mode"] == "static_object" and not Path(region.get("reference", "")).is_file():
                    raise ValueError("Static object filter requires an existing reference image")
                for key in ("min_overlap", "min_changed_fraction"):
                    if not 0 < region.get(key, 0.85 if key == "min_overlap" else 0.15) <= 1:
                        raise ValueError(f"Invalid scene threshold: {key}")
            if not camera.video.is_file():
                raise ValueError(f"Video does not exist: {camera.video}")
            if not math.isfinite(camera.time_offset_sec):
                raise ValueError("Camera time offset must be finite")
            if camera.clock_anchors:
                if len(camera.clock_anchors) < 2:
                    raise ValueError("clock_anchors requires at least two anchors")
                previous = (-math.inf, -math.inf)
                for anchor in camera.clock_anchors:
                    if set(anchor) != {"video_time_sec", "timestamp_sec"}:
                        raise ValueError("Clock anchor requires video_time_sec and timestamp_sec")
                    pair = (anchor["video_time_sec"], anchor["timestamp_sec"])
                    if not all(math.isfinite(x) for x in pair) or not all(
                        a > b for a, b in zip(pair, previous)
                    ):
                        raise ValueError("Clock anchors must be finite and strictly increasing on both axes")
                    previous = pair
        for key in ("imgsz", "stride", "selected_crops_per_track", "preview_max_width"):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("confidence", "iou"):
            if not 0 < getattr(self, key) <= 1:
                raise ValueError(f"{key} must be in (0, 1]")
        for key in ("lost_track_sec", "crop_interval_sec"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive and finite")
        allowed = {"track_high_thresh", "track_low_thresh", "new_track_thresh", "match_thresh", "fuse_score"}
        if set(self.tracker) - allowed:
            raise ValueError(f"Unknown tracker settings: {set(self.tracker) - allowed}")
        params = tracker_parameters(self)
        if type(params["fuse_score"]) is not bool:
            raise ValueError("tracker.fuse_score must be a boolean")
        for key in allowed - {"fuse_score"}:
            if not 0 <= params[key] <= 1:
                raise ValueError(f"tracker.{key} must be in [0, 1]")
        if not self.confidence <= params["track_low_thresh"] <= params["track_high_thresh"]:
            raise ValueError("Require confidence <= track_low_thresh <= track_high_thresh")
        if params["new_track_thresh"] < params["track_high_thresh"]:
            raise ValueError("new_track_thresh must be >= track_high_thresh")


def tracker_parameters(settings: Settings) -> dict:
    return {
        "track_high_thresh": 0.25,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.3,
        "match_thresh": 0.8,
        "fuse_score": True,
        **settings.tracker,
    }


def load_settings(path: Path) -> Settings:
    path = path.resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a YAML mapping")
    raw = dict(raw)

    def resolve(value):
        return (path.parent / value).resolve()

    camera_values = raw.pop("cameras")
    if not isinstance(camera_values, list):
        raise ValueError("cameras must be a list")
    for camera in camera_values:
        if not isinstance(camera, dict) or set(camera) - {
            "camera_id",
            "video",
            "time_offset_sec",
            "clock_anchors",
            "scene_regions",
            "occluder_polygons",
        }:
            raise ValueError("Invalid camera mapping or unknown camera field")
    cameras = [
        CameraConfig(
            camera_id=c["camera_id"],
            video=resolve(c["video"]),
            time_offset_sec=float(c.get("time_offset_sec", 0)),
            clock_anchors=c.get("clock_anchors", []),
            occluder_polygons=c.get("occluder_polygons", []),
            scene_regions=[
                {**r, **({"reference": str(resolve(r["reference"]))} if "reference" in r else {})}
                for r in c.get("scene_regions", [])
            ],
        )
        for c in camera_values
    ]
    settings = Settings(
        cameras=cameras,
        model=resolve(raw.pop("model")),
        output_root=resolve(raw.pop("output_root")),
        **raw,
    )
    settings.validate()
    return settings
