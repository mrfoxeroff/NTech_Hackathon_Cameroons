"""Neural wall/door masks plus automatic drawing structure, without landmarks."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import run_experiment as runtime
import torch
from lens_geometry import remap_grid
from PIL import Image
from structural_features import plan_structure


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--rectified", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for mode in ["original"] + (["rectified"] if args.rectified else []):
        for label in ["wall", "door", "preview"]:
            (args.output / mode / label).mkdir(parents=True)
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
        "wall": [labels[label] for label in ["wall", "column"] if label in labels],
        "door": [labels["door"]],
    }
    report = {
        "model": name,
        "labels": categories,
        "thresholds": {"wall": 0.4, "door": 0.35},
        "manual_landmarks_used": False,
        "cameras": {},
    }
    lenses = (
        __import__("json").loads((args.rectified / "lens.json").read_text(encoding="utf-8"))["cameras"]
        if args.rectified
        else {}
    )
    for path in sorted((args.inputs / "views").glob("*.jpg")):
        image = Image.open(path).convert("RGB")
        masks = {}
        with torch.inference_mode():
            probs = model(**processor(images=image, return_tensors="pt").to("cuda")).logits.softmax(1)
            for category, indexes in categories.items():
                score = (
                    torch.nn.functional.interpolate(
                        probs[:, indexes].sum(1, keepdim=True),
                        size=(image.height, image.width),
                        mode="bilinear",
                        align_corners=False,
                    )[0, 0]
                    .cpu()
                    .numpy()
                )
                masks[category] = (score >= report["thresholds"][category]).astype(np.uint8) * 255
        report["cameras"][path.stem] = {
            key + "_fraction": float((mask > 0).mean()) for key, mask in masks.items()
        }
        for mode in ["original"] + (["rectified"] if args.rectified else []):
            shown = np.asarray(image).copy()
            converted = masks
            if mode == "rectified":
                mx, my, _ = remap_grid(lenses[path.stem])
                shown = np.asarray(Image.open(args.rectified / "views" / path.name).convert("RGB")).copy()
                converted = {
                    key: cv2.remap(mask, mx, my, cv2.INTER_NEAREST, borderValue=0)
                    for key, mask in masks.items()
                }
            for category, color in [("wall", [240, 85, 70]), ("door", [40, 190, 240])]:
                mask = converted[category]
                cv2.imwrite(str(args.output / mode / category / f"{path.stem}.png"), mask)
                shown[mask > 0] = (0.45 * shown[mask > 0] + 0.55 * np.array(color)).astype(np.uint8)
            Image.fromarray(shown).save(args.output / mode / "preview" / f"{path.stem}.jpg")
        print(path.stem, report["cameras"][path.stem], flush=True)
    plan = Image.open(args.inputs / "plan.jpg").convert("RGB")
    structure, wall, doors = plan_structure(plan)
    structure.save(args.output / "plan-structure.png")
    cv2.imwrite(str(args.output / "plan-walls.png"), wall)
    report["plan_doors"] = doors
    runtime.save_json(args.output / "semantics.json", report)
    print("Drawing door proposals:", len(doors))


if __name__ == "__main__":
    main()
