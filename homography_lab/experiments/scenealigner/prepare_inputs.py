"""Prepare median camera backgrounds and neural floor masks, without landmarks.

Run backgrounds with the existing PyAV environment; masks with .venv-scene.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB = HERE.parents[1]
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["backgrounds", "masks"])
    parser.add_argument("--workspace", type=Path, default=LAB.parent)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "backgrounds":
        if args.output.exists():
            parser.error("Use a new output directory")
        sys.path.insert(0, str(LAB))
        from features import background, write_image

        views = args.output / "views"
        views.mkdir(parents=True)
        shutil.copy2(HERE / "inputs/plan.jpg", args.output / "plan.jpg")
        manifest = {"manual_landmarks_used": False, "cameras": {}}
        session_path = LAB / "runs/office-01/session.json"
        previous_cameras = (
            json.loads(session_path.read_text(encoding="utf-8"))["cameras"] if session_path.is_file() else {}
        )
        for number in range(1, 6):
            video = args.workspace / f"camera_new_{number}.mkv"
            target = views / f"cam_{number}.jpg"
            source_digest = digest(video)
            previous = HERE / f"inputs/views/cam_{number}.jpg"
            existing = previous_cameras.get(f"cam_{number}", {})
            print(f"Preparing camera {number}", flush=True)
            if (
                previous.is_file()
                and source_digest == existing.get("source_sha256")
                and digest(previous) == existing.get("reference_image_sha256")
            ):
                shutil.copy2(previous, target)
                metadata = existing["background"] | {
                    "background_method": "reuse_verified_temporal_median",
                    "reused_from": str(previous.resolve()),
                }
            else:
                image, _, metadata = background(video)
                write_image(target, image)
            manifest["cameras"][f"cam_{number}"] = metadata | {
                "source_video": str(video.resolve()),
                "video_sha256": source_digest,
                "image_sha256": digest(target),
            }
            (args.output / "preparation.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
        return
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

    torch.set_num_threads(4)
    model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
    cache = LAB / "neural_weights/huggingface"
    processor = AutoImageProcessor.from_pretrained(
        model_name, cache_dir=cache, local_files_only=True, use_fast=False
    )
    model = (
        AutoModelForSemanticSegmentation.from_pretrained(
            model_name, cache_dir=cache, local_files_only=True, use_safetensors=True
        )
        .eval()
        .to("cuda")
    )
    ids = [int(i) for i, label in model.config.id2label.items() if label.lower().strip() == "floor"]
    if len(ids) != 1:
        raise ValueError("Expected a single floor class")
    masks = args.output / "floor"
    masks.mkdir(exist_ok=False)
    report = {
        "model": model_name,
        "floor_class": ids[0],
        "threshold": 0.6,
        "manual_masks_used": False,
        "cameras": {},
    }
    for path in sorted((args.output / "views").glob("*.jpg")):
        with Image.open(path) as raw:
            image = raw.convert("RGB")
        with torch.inference_mode():
            logits = model(**processor(images=image, return_tensors="pt").to("cuda")).logits
            probability = (
                torch.nn.functional.interpolate(
                    logits.softmax(1)[:, ids[0] : ids[0] + 1],
                    size=(image.height, image.width),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
                .cpu()
                .numpy()
            )
        mask = (probability >= 0.6).astype(np.uint8) * 255
        cv2.imwrite(str(masks / f"{path.stem}.png"), mask)
        report["cameras"][path.stem] = {"floor_fraction": float((mask > 0).mean())}
        print(path.stem, report["cameras"][path.stem], flush=True)
    (args.output / "segmentation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
