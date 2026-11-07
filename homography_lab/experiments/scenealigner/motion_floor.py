"""Training-only semantic negative evidence; uncertain floor is not declared non-walkable."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from floor_ray_geometry import floor_homography, project_floor
from motion_observations import visible_foot
from PIL import Image


def admissible_floor(floor_probability, obstacle_probability):
    return (floor_probability >= 0.2) | (obstacle_probability <= 0.7)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--observations", type=Path, required=True)
    p.add_argument("--lenses", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--hf-cache", type=Path)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import run_experiment as runtime
    import torch

    frames = json.loads(args.frames.read_text())
    lenses = json.loads(args.lenses.read_text())["cameras"]
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

    name = "nvidia/segformer-b0-finetuned-ade-512-512"
    revision = "489d5cd81a0b59fab9b7ea758d3548ebe99677da"
    cache = args.hf_cache or runtime.LAB / "neural_weights/huggingface"
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    processor = AutoImageProcessor.from_pretrained(
        name, revision=revision, cache_dir=cache, local_files_only=True, use_fast=False
    )
    model = (
        AutoModelForSemanticSegmentation.from_pretrained(
            name, revision=revision, cache_dir=cache, local_files_only=True, use_safetensors=True
        )
        .eval()
        .to(device)
    )
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    labels = {v.lower().strip(): int(k) for k, v in model.config.id2label.items()}
    negative = [
        labels[k]
        for k in [
            "wall",
            "ceiling",
            "table",
            "television receiver",
            "cabinet",
            "windowpane",
            "counter",
            "shelf",
            "bed",
            "sofa",
        ]
        if k in labels
    ]
    floor_ids = [labels[k] for k in ["floor", "rug"] if k in labels]
    masks = {}
    report = {
        "manual_plan_landmarks_used": False,
        "holdout_images_used_to_prepare_semantics": False,
        "cameras": {},
    }
    for camera in frames["config"]["cameras"]:
        training = [f for f in frames["frames"] if f["camera"] == camera and f["split"] == "train"]
        times = frames["config"].get("semantic_sample_times_sec")
        selected = (
            [min(training, key=lambda f: abs(f["timestamp_sec"] - t)) for t in times]
            if times
            else [training[i] for i in np.linspace(0, len(training) - 1, min(5, len(training))).astype(int)]
        )
        probabilities = []
        for frame in selected:
            image = Image.open(frame["image"]).convert("RGB")
            with torch.inference_mode():
                prob = model(**processor(images=image, return_tensors="pt").to(device)).logits.softmax(1)
                channels = torch.stack(
                    (prob[:, floor_ids].sum(1), prob[:, negative].sum(1), prob[:, labels["person"]]), dim=1
                )
                resized = (
                    torch.nn.functional.interpolate(
                        channels, size=image.size[::-1], mode="bilinear", align_corners=False
                    )[0]
                    .cpu()
                    .numpy()
                )
            resized[:2, resized[2] > 0.5] = np.nan
            probabilities.append(resized[:2])
        with __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore", RuntimeWarning)
            estimate = np.nanmedian(np.stack(probabilities), axis=0)
        estimate = np.nan_to_num(estimate, nan=0.0)
        masks[camera] = admissible_floor(estimate[0], estimate[1]).astype(np.uint8) * 255
        cv2.imwrite(str(args.output / (camera + "-admissible.png")), masks[camera])
        np.savez_compressed(
            args.output / (camera + "-semantics.npz"),
            floor=estimate[0].astype(np.float16),
            obstacle=estimate[1].astype(np.float16),
        )
        report["cameras"][camera] = {
            "training_frame_ids": [f["id"] for f in selected],
            "admissible_fraction": float(np.mean(masks[camera] > 0)),
            "policy": "reject_only_confident_obstacles_unless_floor_supported",
        }
        print(camera, report["cameras"][camera], flush=True)
    rows = [json.loads(line) for line in args.observations.read_text().splitlines() if line]
    counts = {"bodies": len(rows), "usable_feet": 0, "rejections": {}}
    by_frame = {}
    for row in rows:
        by_frame.setdefault(row["frame_id"], []).append(row)
    with (args.output / "observations.jsonl").open("x") as out:
        for row in rows:
            camera = row["camera"]
            others = [r["box"] for r in by_frame[row["frame_id"]] if r is not row]
            foot, reason = visible_foot(row["box"], row["keypoints"], masks[camera], others)
            if foot is not None:
                _, good = project_floor(
                    foot[None],
                    lenses[camera],
                    floor_homography(lenses[camera]["new_K"], lenses[camera]["gravity"]),
                )
                if not good[0]:
                    foot, reason = None, "invalid_floor_ray"
            row.update(foot=None if foot is None else foot.tolist(), foot_rejection=reason)
            kp = np.asarray(row["keypoints"])[15:17]
            row["foot_confidence"] = float(np.mean(kp[kp[:, 2] >= 0.55, 2])) if foot is not None else 0.0
            if foot is not None:
                counts["usable_feet"] += 1
            else:
                counts["rejections"][reason] = counts["rejections"].get(reason, 0) + 1
            out.write(json.dumps(row, separators=(",", ":")) + "\n")
    report["counts"] = counts
    (args.output / "semantics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(counts, flush=True)


if __name__ == "__main__":
    main()
