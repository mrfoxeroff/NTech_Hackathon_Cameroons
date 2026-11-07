"""Estimate lens geometry from images and rectify images/masks, without plan anchors."""

import argparse
import hashlib
import shutil
from pathlib import Path

import cv2
import numpy as np
import run_experiment as runtime
import torch
from lens_geometry import remap_grid, undistort_points
from PIL import Image


def lens_from_prediction(result, width, height, camera_model):
    """Shared geometry for the CLI and per-frame reliability experiment."""
    camera = result["camera"]
    intrinsic = camera.K[0].cpu().numpy()
    lens = {
        "model": camera_model,
        "K": intrinsic.tolist(),
        "new_K": intrinsic.tolist(),
        "k1": float(camera.k1[0]),
        "size": [width, height],
        "rectified_size": [width, height],
        "estimation_source": "neural_image_fields_no_manual_coordinates",
        "gravity": (-result["gravity"].vec3d[0]).cpu().numpy().tolist(),
    }
    border = np.concatenate(
        [np.column_stack((np.linspace(0, width - 1, 128), np.full(128, y))) for y in [0, height - 1]]
        + [np.column_stack((np.full(128, x), np.linspace(0, height - 1, 128))) for x in [0, width - 1]]
    )
    expanded, invertible = undistort_points(border, lens)
    if not invertible.any():
        raise ValueError("No invertible sampled camera border")
    ratios = abs(expanded[invertible] - intrinsic[:2, 2]) / np.array([width / 2, height / 2])
    factor = min(float(max(1.0, np.quantile(ratios, 0.95))), 1.8)
    new_intrinsic = intrinsic.copy()
    new_intrinsic[0, 0] /= factor
    new_intrinsic[1, 1] /= factor
    lens.update(
        new_K=new_intrinsic.tolist(),
        border_invertible_fraction=float(invertible.mean()),
        field_expansion=factor,
    )
    for key in ["focal_uncertainty", "k1_uncertainty", "roll_uncertainty", "pitch_uncertainty"]:
        if key in result:
            lens[key] = result[key].float().cpu().numpy().ravel().tolist()
    return lens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--camera-model", choices=["simple_radial", "simple_divisional"], default="simple_divisional"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "views").mkdir()
    (args.output / "floor").mkdir()
    (args.output / "valid").mkdir()
    shutil.copy2(args.inputs / "plan.jpg", args.output / "plan.jpg")
    from geocalib import GeoCalib

    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    cv2.setNumThreads(2)
    model = GeoCalib(weights="distorted").eval().cuda()
    report = {
        "method": "GeoCalib_distorted_" + args.camera_model,
        "manual_landmarks_used": False,
        "source": str(args.inputs.resolve()),
        "cameras": {},
    }
    for path in sorted((args.inputs / "views").glob("*.jpg")):
        image = np.asarray(Image.open(path).convert("RGB"))
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float().cuda() / 255
        with torch.inference_mode():
            result = model.calibrate(tensor, camera_model=args.camera_model)
        lens = lens_from_prediction(result, image.shape[1], image.shape[0], args.camera_model)
        mapx, mapy, valid = remap_grid(lens)
        corrected = cv2.remap(
            image, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
        )
        Image.fromarray(corrected).save(args.output / "views" / path.name, quality=95)
        floor = cv2.imread(str(args.inputs / "floor" / f"{path.stem}.png"), 0)
        corrected_floor = cv2.remap(
            floor, mapx, mapy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        cv2.imwrite(str(args.output / "floor" / f"{path.stem}.png"), corrected_floor)
        cv2.imwrite(str(args.output / "valid" / f"{path.stem}.png"), valid.astype(np.uint8) * 255)
        lens["valid_rectified_fraction"] = float(valid.mean())
        lens["input_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        report["cameras"][path.stem] = lens
        runtime.save_json(args.output / "lens.json", report)
        print(
            path.stem,
            "k1",
            lens["k1"],
            "expansion",
            lens["field_expansion"],
            "valid fraction",
            lens["valid_rectified_fraction"],
            flush=True,
        )
    runtime.save_json(args.output / "lens.json", report)


if __name__ == "__main__":
    main()
