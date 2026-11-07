"""Persistent floor/boundary semantics and robust camera parameters, without plan landmarks."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import run_experiment as runtime
import torch
from floor_ray_geometry import boundary_geometry
from PIL import Image


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inference", type=Path, required=True)
    p.add_argument("--registration", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = read(args.inference / "protocol.json")
    registrations = read(args.registration)["cases"]
    temporal = sorted(
        [case for case in protocol["cases"] if case["kind"] in ["uniform", "foreground_stress"]],
        key=lambda case: case["actual_seconds"],
    )
    training = temporal[::2]
    if any(not registrations[case["id"]]["accepted"] for case in training):
        raise ValueError("Camera motion prevents unqualified temporal aggregation")
    lens = deepcopy(training[0]["lens"])
    intrinsic = np.median(np.array([case["lens"]["K"] for case in training]), axis=0)
    gravity = np.median(np.array([case["lens"]["gravity"] for case in training]), axis=0)
    gravity /= np.linalg.norm(gravity)
    lens.update(
        K=intrinsic.tolist(),
        new_K=intrinsic.tolist(),
        k1=float(np.median([case["lens"]["k1"] for case in training])),
        gravity=gravity.tolist(),
        estimation_source="robust_training_frame_camera_estimates_no_plan_landmarks",
    )
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

    name = "nvidia/segformer-b0-finetuned-ade-512-512"
    cache = runtime.LAB / "neural_weights/huggingface"
    processor = AutoImageProcessor.from_pretrained(
        name, cache_dir=cache, local_files_only=True, use_fast=False
    )
    model = (
        AutoModelForSemanticSegmentation.from_pretrained(
            name, cache_dir=cache, local_files_only=True, use_safetensors=True
        )
        .eval()
        .cuda()
    )
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    labels = {label.lower().strip(): int(i) for i, label in model.config.id2label.items()}
    categories = {
        "floor": [labels["floor"]],
        "boundary": [labels[k] for k in ["wall", "column", "windowpane", "door"] if k in labels],
        "window": [labels["windowpane"]],
        "person": [labels["person"]],
    }
    all_cases = [
        {
            "id": case["id"],
            "kind": "training",
            "image": case["original_image"],
            "actual_seconds": case["actual_seconds"],
        }
        for case in training
    ] + read(args.validation)["cases"]
    aligned = {"floor": [], "boundary": []}
    metadata = {
        "manual_landmarks_used": False,
        "training_ids": [case["id"] for case in training],
        "validation_ids": [case["id"] for case in all_cases if case["kind"] != "training"],
        "lens": lens,
        "cases": all_cases,
    }
    for case in all_cases:
        image = Image.open(case["image"]).convert("RGB")
        masks = {}
        with torch.inference_mode():
            logits = model(**processor(images=image, return_tensors="pt").to("cuda")).logits.softmax(1)
            probabilities = {}
            for key, indexes in categories.items():
                probabilities[key] = (
                    torch.nn.functional.interpolate(
                        logits[:, indexes].sum(1, keepdim=True),
                        size=(image.height, image.width),
                        mode="bilinear",
                        align_corners=False,
                    )[0, 0]
                    .cpu()
                    .numpy()
                )
        masks["floor"] = (probabilities["floor"] >= 0.6).astype(np.uint8)
        masks["boundary"] = ((probabilities["boundary"] >= 0.4) & (probabilities["person"] < 0.3)).astype(
            np.uint8
        )
        masks["window"] = (probabilities["window"] >= 0.35).astype(np.uint8)
        folder = args.output / case["id"]
        folder.mkdir()
        for key, mask in masks.items():
            cv2.imwrite(str(folder / (key + ".png")), mask * 255)
        case["fractions"] = {key: float(mask.mean()) for key, mask in masks.items()}
        if case["kind"] == "training":
            inverse = cv2.invertAffineTransform(np.asarray(registrations[case["id"]]["matrix"]))
            for key in aligned:
                aligned[key].append(
                    cv2.warpAffine(masks[key], inverse, (image.width, image.height), flags=cv2.INTER_NEAREST)
                )
        print(case["id"], case["fractions"], flush=True)
    aggregated = {key: (np.mean(images, axis=0) >= 0.6).astype(np.uint8) for key, images in aligned.items()}
    for key, mask in aggregated.items():
        cv2.imwrite(str(args.output / (key + "-aggregate.png")), mask * 255)
    geometry = boundary_geometry(aggregated["floor"], aggregated["boundary"], lens, gravity)
    cv2.imwrite(str(args.output / "analytic-boundary-density.png"), geometry["image"])
    np.savez_compressed(args.output / "geometry.npz", **geometry)
    metadata["features"] = {
        "wall_points": len(geometry["wall_points"]),
        "floor_points": len(geometry["floor_points"]),
        "visibility_rays": len(geometry["ray_endpoints_density"]),
    }
    runtime.save_json(args.output / "features.json", metadata)
    print(metadata["features"], flush=True)


if __name__ == "__main__":
    main()
