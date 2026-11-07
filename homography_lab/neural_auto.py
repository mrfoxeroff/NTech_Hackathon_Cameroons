"""Automatic photo-to-floorplan correspondences: C3Po + neural floor segmentation.

No hand-authored landmark coordinates or existing office homographies are read.
Neural self-consistency is NOT independent ground-truth calibration accuracy.
"""

import argparse
import gc
import json
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "neural_deps"))

import cv2
import numpy as np
from PIL import Image

from features import write_image
from geometry import fit_calibration


def letterbox(image, size=512):
    ratio = min(size / image.width, size / image.height)
    width, height = int(image.width * ratio), int(image.height * ratio)
    offset = ((size - width) // 2, (size - height) // 2)
    result = Image.new("RGB", (size, size))
    result.paste(image.resize((width, height), Image.Resampling.BICUBIC), offset)
    return result, {"ratio": ratio, "offset": offset, "size": image.size}


def original_coordinates(value, transform):
    return (np.asarray(value, dtype=float) - transform["offset"]) / transform["ratio"]


def floor_mask(image, cache, device="cpu", allow_downloads=False):
    import torch
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

    name = "nvidia/segformer-b0-finetuned-ade-512-512"
    processor = AutoImageProcessor.from_pretrained(
        name, cache_dir=cache, local_files_only=not allow_downloads
    )
    model = (
        AutoModelForSemanticSegmentation.from_pretrained(
            name, cache_dir=cache, use_safetensors=True, local_files_only=not allow_downloads
        )
        .to(device)
        .eval()
    )
    labels = model.config.id2label
    floor_ids = [int(i) for i, label in labels.items() if label.lower().strip() == "floor"]
    if len(floor_ids) != 1:
        raise ValueError("Segmentation model must identify a unique floor class")
    with torch.inference_mode():
        inputs = processor(images=image, return_tensors="pt").to(device)
        logits = model(**inputs).logits
        probability = logits.softmax(1)[:, floor_ids[0] : floor_ids[0] + 1]
        probability = (
            torch.nn.functional.interpolate(
                probability, size=(image.height, image.width), mode="bilinear", align_corners=False
            )[0, 0]
            .cpu()
            .numpy()
        )
    del model
    gc.collect()
    return probability, {"model": name, "floor_class": floor_ids[0]}


def load_model(checkpoint, device):
    from argparse import Namespace

    import torch

    sys.path.insert(0, str(ROOT / "vendor/C3Po"))
    from dust3r.model import AsymmetricCroCo3DStereo

    torch.serialization.add_safe_globals([Namespace])
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    # Explicit architecture from the official demo; do not eval checkpoint text.
    with torch.device("meta"):
        model = AsymmetricCroCo3DStereo(
            pos_embed="RoPE100",
            patch_embed_cls="ManyAR_PatchEmbed",
            img_size=(512, 512),
            head_type="dpt",
            output_mode="pts3d",
            depth_mode=("exp", -float("inf"), float("inf")),
            conf_mode=("exp", 1, float("inf")),
            enc_embed_dim=1024,
            enc_depth=24,
            enc_num_heads=16,
            dec_embed_dim=768,
            dec_depth=12,
            dec_num_heads=12,
        )
    loaded = model.load_state_dict(state["model"], strict=False, assign=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise ValueError(f"Model checkpoint mismatch: {loaded}")
    model = model.to(device).eval()
    del state
    gc.collect()
    return model


def infer_map(model, plan, photo, device):
    import torch
    from torchvision.transforms.functional import to_tensor

    p, p_transform = letterbox(plan)
    c, c_transform = letterbox(photo)

    def view(image):
        return {
            "img": ((to_tensor(image) - 0.5) / 0.5)[None].to(device),
            "true_shape": torch.tensor([[512, 512]], dtype=torch.int32, device=device),
        }

    with torch.inference_mode():
        _, prediction = model(view(p), view(c))
    # Official C3Po ReverseCoordNorm: (value + 1) * (size - 1) / 2.
    mapping = (prediction["pts3d_in_other_view"][0, :, :, [0, 2]].cpu().numpy() + 1) * 511 / 2
    return mapping, p_transform, c_transform


def sample_correspondences(mapping, floor, plan_transform, photo_transform, *, stride=16, limit=120):
    """Uniform spatial proposals, not manually chosen landmark locations."""
    result = []
    for y in range(stride // 2, mapping.shape[0], stride):
        for x in range(stride // 2, mapping.shape[1], stride):
            camera = original_coordinates([x, y], photo_transform)
            plan = original_coordinates(mapping[y, x], plan_transform)
            if (
                not np.isfinite(plan).all()
                or np.any(camera < 0)
                or np.any(camera >= photo_transform["size"])
                or np.any(plan < 0)
                or np.any(plan >= plan_transform["size"])
            ):
                continue
            u, v = np.rint(camera).astype(int)
            u, v = min(u, floor.shape[1] - 1), min(v, floor.shape[0] - 1)
            if floor[v, u] < 0.60:
                continue
            result.append(
                {
                    "camera": camera.tolist(),
                    "plan": plan.tolist(),
                    "floor_probability": float(floor[v, u]),
                    "grid": [x // stride, y // stride],
                    "kind": "neural_floor_correspondence",
                }
            )
    # Spread candidates across the image before limiting, avoiding a single row.
    result.sort(key=lambda p: ((p["grid"][0] * 17 + p["grid"][1] * 31) % 127, -p["floor_probability"]))
    selected = []
    for point in result:
        if all(np.linalg.norm(np.array(point["plan"]) - p["plan"]) >= 3 for p in selected):
            selected.append(point)
        if len(selected) >= limit:
            break
    # This is held-out NETWORK agreement, never a human/physical ground truth.
    for i, point in enumerate(selected):
        point["role"] = "check" if i % 4 == 0 else "fit"
        point["id"] = str(i + 1)
    return selected


def estimate(pairs, camera_size, plan_size):
    report = {
        "status": "rejected",
        "validation_source": "held_out_predictions_of_same_network",
        "physical_accuracy_verified": False,
        "manual_landmarks_used": False,
    }
    if len(pairs) < 16:
        return None, report | {"reason": "insufficient_neural_floor_correspondences", "pairs": len(pairs)}
    try:
        result = fit_calibration(
            pairs, list(camera_size), list(plan_size), threshold_px=5, max_check_error_px=8
        )
    except ValueError as error:
        return None, report | {"reason": str(error)}
    quality = result["quality"]
    if result["status"] != "validated_on_check_points" or quality["camera_hull_fraction"] < 0.03:
        return result | {"status": "rejected_neural_hypothesis"}, report | {
            "reason": "neural_correspondences_not_geometrically_consistent",
            "quality": quality,
        }
    result["status"] = "automatic_candidate"
    result["validation_source"] = report["validation_source"]
    result["physical_accuracy_verified"] = False
    return result, report | {
        "status": "automatic_candidate",
        "quality": quality,
        "reason": "Needs independent temporal/geometry evaluation; no manual input required",
    }


def visualize(photo, plan, pairs, calibration, path):
    height = 600
    left = cv2.resize(np.asarray(photo)[:, :, ::-1], (int(photo.width * height / photo.height), height))
    right = cv2.resize(np.asarray(plan)[:, :, ::-1], (int(plan.width * height / plan.height), height))
    output = np.concatenate((left, right), axis=1)
    for p in pairs[:60]:
        a = tuple(np.rint(np.array(p["camera"]) * height / photo.height).astype(int))
        b = np.rint(np.array(p["plan"]) * height / plan.height).astype(int)
        b[0] += left.shape[1]
        color = (100, 210, 255) if p["role"] == "check" else (150, 230, 120)
        cv2.circle(output, a, 3, color, -1)
        cv2.circle(output, tuple(b), 3, color, -1)
        cv2.line(output, a, tuple(b), color, 1, cv2.LINE_AA)
    label = (
        "NEURAL CANDIDATE / NOT PHYSICALLY VERIFIED"
        if calibration and calibration["status"] == "automatic_candidate"
        else "REJECTED / NO VERIFIED CALIBRATION"
    )
    cv2.rectangle(output, (0, 0), (output.shape[1], 32), (20, 25, 35), -1)
    cv2.putText(output, label, (12, 23), 0, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    write_image(path, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "neural_weights/C3Po.pth")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera", action="append")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--allow-downloads", action="store_true", help="Allow downloading public floor-segmentation weights"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    if not args.checkpoint.is_file():
        parser.error("Official C3Po checkpoint is missing")
    import torch

    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    session = json.loads((args.session / "session.json").read_text(encoding="utf-8"))
    if session["status"] != "complete":
        parser.error("Session preparation did not complete")
    args.output.mkdir(parents=True)
    manifest = {
        "status": "running",
        "method": "C3Po+SegFormer_floor+RANSAC",
        "cameras": {},
        "manual_landmarks_used": False,
        "device": args.device,
    }

    def snapshot():
        path = args.output / "manifest.pending.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        path.replace(args.output / "manifest.json")

    snapshot()
    try:
        plan = Image.open(args.session / "plan.png").convert("RGB")
        cameras = args.camera or list(session["cameras"])
        manifest["cameras"] = {camera: {"status": "pending"} for camera in cameras}
        floors = {}
        photos = {}
        for camera in cameras:
            manifest["step"] = f"SegFormer: {camera}"
            snapshot()
            photos[camera] = Image.open(args.session / f"{camera}.jpg").convert("RGB")
            print(camera, "segmenting floor", flush=True)
            floors[camera], segmentation = floor_mask(
                photos[camera], ROOT / "neural_weights/huggingface", args.device, args.allow_downloads
            )
            write_image(args.output / f"{camera}-floor.png", np.uint8(floors[camera] > 0.6) * 255)
        print("Loading C3Po checkpoint", flush=True)
        manifest["step"] = "Loading C3Po"
        snapshot()
        model = load_model(args.checkpoint, args.device)
        for camera in cameras:
            manifest["step"] = f"C3Po: {camera}"
            snapshot()
            print(camera, "predicting photo-plan correspondences", flush=True)
            start = perf_counter()
            mapping, pt, ct = infer_map(model, plan, photos[camera], args.device)
            np.savez_compressed(args.output / f"{camera}-dense.npz", map=mapping)
            pairs = sample_correspondences(mapping, floors[camera], pt, ct)
            calibration, report = estimate(pairs, photos[camera].size, plan.size)
            report.update(
                inference_sec=perf_counter() - start,
                pairs=len(pairs),
                segmentation=segmentation,
                has_calibration=calibration is not None,
            )
            if calibration:
                calibration.update(
                    camera_id=camera,
                    provenance="C3Po automatic prediction",
                    plan_sha256=session["plan"]["sha256"],
                    source_sha256=session["cameras"][camera]["source_sha256"],
                )
                (args.output / f"{camera}-calibration.json").write_text(json.dumps(calibration, indent=2))
            (args.output / f"{camera}-pairs.json").write_text(
                json.dumps(
                    {
                        "camera_id": camera,
                        "camera_size": list(photos[camera].size),
                        "plan_size": list(plan.size),
                        "pairs": pairs,
                        "provenance": "neural_unverified",
                    },
                    indent=2,
                )
            )
            visualize(photos[camera], plan, pairs, calibration, args.output / f"{camera}-matches.jpg")
            manifest["cameras"][camera] = report
            snapshot()
            print(camera, json.dumps(report), flush=True)
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        snapshot()


if __name__ == "__main__":
    main()
