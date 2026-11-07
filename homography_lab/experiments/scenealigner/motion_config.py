"""Portable configuration and early validation for the motion calibration lab."""

import json
import re
from copy import deepcopy
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB = HERE.parents[1]
REPOSITORY = HERE.parents[2]
OFFICE = HERE / "assets" / "office"


def load_config(path, video_root=None, check_files=False):
    path = Path(path).resolve()
    config = deepcopy(json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(config.get("sample_fps"), int) or not 1 <= config["sample_fps"] <= 25:
        raise ValueError("sample_fps must be an integer between 1 and 25")
    if not config.get("cameras") or not config.get("windows"):
        raise ValueError("Provide cameras and nonempty time windows")
    identifiers = set()
    for window in config["windows"]:
        name = window["id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in identifiers:
            raise ValueError("Window IDs must be unique safe directory names")
        identifiers.add(name)
        if window["split"] not in ("train", "holdout"):
            raise ValueError("Window split must be train or holdout")
        if not 0 <= window["start"] < window["end"]:
            raise ValueError("Windows require 0 <= start < end")
    windows = sorted(config["windows"], key=lambda item: item["start"])
    for first, second in zip(windows, windows[1:]):
        if second["start"] < first["end"]:
            raise ValueError("Time windows overlap: source frames must not be reused across splits")
    config["windows"] = windows

    def resolve(value):
        candidate = Path(value)
        return str((candidate if candidate.is_absolute() else path.parent / candidate).resolve())

    config["clock_templates"] = resolve(config["clock_templates"])
    for camera, source in config["cameras"].items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", camera):
            raise ValueError("Camera IDs must be safe directory names")
        source["video"] = (
            str((Path(video_root).resolve() / Path(source["video"]).name).resolve())
            if video_root
            else resolve(source["video"])
        )
        source["floor_mask"] = resolve(source["floor_mask"])
    if check_files:
        missing = [
            value
            for source in config["cameras"].values()
            for value in (source["video"], source["floor_mask"])
            if not Path(value).is_file()
        ]
        if not Path(config["clock_templates"]).is_dir():
            missing.append(config["clock_templates"])
        if missing:
            raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))
    return config


def dump_json(path, data):
    """Reject NaN/Infinity instead of silently writing nonstandard JSON."""
    Path(path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
