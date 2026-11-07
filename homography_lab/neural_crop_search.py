"""Explore generic overlapping plan crops without camera-specific coordinates."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image

from neural_auto import ROOT, estimate, infer_map, load_model, sample_correspondences, visualize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera", default="cam_1")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    import torch

    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    args.output.mkdir(parents=True)
    plan = Image.open(args.session / "plan.png").convert("RGB")
    photo = Image.open(args.session / f"{args.camera}.jpg").convert("RGB")
    floor = cv2.imread(str(args.baseline / f"{args.camera}-floor.png"), cv2.IMREAD_GRAYSCALE) / 255
    model = load_model(ROOT / "neural_weights/C3Po.pth", "cpu")
    width, height = round(plan.width * 0.65), round(plan.height * 0.65)
    trials = []
    for x in (0, plan.width - width):
        for y in (0, plan.height - height):
            region = plan.crop((x, y, x + width, y + height))
            if (np.asarray(region).mean(2) < 140).mean() < 0.05:
                continue
            name = f"crop-{x}-{y}"
            print(name, "inference", flush=True)
            started = perf_counter()
            mapping, pt, ct = infer_map(model, region, photo, "cpu")
            pairs = sample_correspondences(mapping, floor, pt, ct)
            for p in pairs:
                p["plan"][0] += x
                p["plan"][1] += y
            calibration, report = estimate(pairs, photo.size, plan.size)
            report.update(
                crop=[x, y, x + width, y + height],
                pairs=len(pairs),
                elapsed_sec=perf_counter() - started,
                manual_landmarks_used=False,
            )
            trials.append(report)
            if calibration:
                (args.output / f"{name}-calibration.json").write_text(json.dumps(calibration, indent=2))
            visualize(photo, plan, pairs, calibration, args.output / f"{name}-matches.jpg")
            (args.output / "trials.json").write_text(json.dumps(trials, indent=2) + "\n")
            print(name, report["status"], "pairs", len(pairs), flush=True)
    print("Search complete. No hypothesis is independently verified.", flush=True)


if __name__ == "__main__":
    main()
