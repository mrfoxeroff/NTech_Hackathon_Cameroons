"""Reject static furniture hallucinated as people, using a training-only temporal background."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def motion_pixels(image, background):
    difference = image.astype(np.float32) - background.astype(np.float32)
    difference -= np.median(difference.reshape(-1, 3), axis=0)
    return np.max(abs(cv2.GaussianBlur(difference, (3, 3), 0)), axis=2) > 25


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--observations", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frames = json.loads(args.frames.read_text())
    backgrounds = {}
    report = {"background_uses_holdout": False, "cameras": {}, "rejected_static_feet": 0}
    for camera in frames["config"]["cameras"]:
        training = [f for f in frames["frames"] if f["camera"] == camera and f["split"] == "train"]
        selected = [training[i] for i in np.linspace(0, len(training) - 1, min(9, len(training))).astype(int)]
        images = []
        for frame in selected:
            image = cv2.imread(frame["image"])
            images.append(cv2.resize(image, (648, 486)))
        background = np.median(images, axis=0).astype(np.float32)
        backgrounds[camera] = background
        cv2.imwrite(str(args.output / (camera + "-background.jpg")), background.astype(np.uint8))
        report["cameras"][camera] = {"training_frame_ids": [f["id"] for f in selected]}
    rows = [json.loads(line) for line in args.observations.read_text().splitlines() if line]
    previous = None
    motion = None
    with (args.output / "observations.jsonl").open("x") as out:
        for row in rows:
            if row["image"] != previous:
                image = cv2.resize(cv2.imread(row["image"]), (648, 486)).astype(np.float32)
                motion = motion_pixels(image, backgrounds[row["camera"]])
                previous = row["image"]
            scale = np.array([648 / row["size"][0], 486 / row["size"][1]])
            box = np.rint(np.array(row["box"]) * np.tile(scale, 2)).astype(int)
            x1, y1, x2, y2 = box
            roi = motion[max(0, y1) : y2, max(0, x1) : x2]
            fraction = float(roi.mean()) if roi.size else 0.0
            row["foreground_fraction"] = fraction
            if row.get("foot") is not None and fraction < 0.03:
                row.update(foot=None, foot_confidence=0.0, foot_rejection="no_foreground_motion")
                report["rejected_static_feet"] += 1
            out.write(json.dumps(row, separators=(",", ":")) + "\n")
    report["usable_feet"] = sum(r.get("foot") is not None for r in rows)
    report["limitation"] = "Foreground evidence is a filter, not a person or identity ground-truth label."
    (args.output / "foreground.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "Static false feet rejected",
        report["rejected_static_feet"],
        "retained",
        report["usable_feet"],
        flush=True,
    )


if __name__ == "__main__":
    main()
