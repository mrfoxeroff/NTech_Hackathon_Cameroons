"""Reference-assisted DIAGNOSTIC; not an automatic inference/calibration path."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scenealigner.inference import get_density_coordinates, reconstruct_projection_geometry


def fit_similarity(x, y):
    matrix = np.zeros((2 * len(x), 4))
    matrix[::2] = np.column_stack((x[:, 0], -x[:, 1], np.ones(len(x)), np.zeros(len(x))))
    matrix[1::2] = np.column_stack((x[:, 1], x[:, 0], np.zeros(len(x)), np.ones(len(x))))
    a, b, tx, ty = np.linalg.lstsq(matrix, y.ravel(), rcond=None)[0]
    return np.array([[a, -b, tx], [b, a, ty]])


def errors(matrix, x, y):
    distances = np.linalg.norm(np.column_stack((x, np.ones(len(x)))) @ matrix.T - y, axis=1)
    return {
        "median_px": float(np.median(distances)),
        "p95_px": float(np.quantile(distances, 0.95)),
        "points": len(x),
        "errors_px": distances.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output file")
    recon = dict(np.load(args.run / "reconstruction.npz"))
    manifest = json.loads((args.run / "manifest.json").read_text(encoding="utf-8"))
    pp, _, _ = reconstruct_projection_geometry(recon["points_cam0"], recon["conf"], recon["est_gravity_cam"])
    cameras = {}
    for i, image in enumerate(manifest["images"]):
        name = Path(image).stem
        reference = json.loads((args.reference / f"{name}_calibration.json").read_text(encoding="utf-8"))
        width, height = recon["image_sizes"][i]
        h, w = recon["conf"][i].shape
        source = np.asarray(reference["camera_points"], dtype=float)
        source *= [width / reference["camera_size"][0], height / reference["camera_size"][1]]
        source += [(w - width) // 2, (h - height) // 2]
        xyz = cv2.remap(
            recon["points_cam0"][i].astype(np.float32),
            source[:, 0:1].astype(np.float32),
            source[:, 1:2].astype(np.float32),
            cv2.INTER_LINEAR,
        )[:, 0]
        density = get_density_coordinates(xyz @ pp["R_rect"].T, pp)
        use = np.asarray(reference.get("inliers", [True] * len(source)), dtype=bool)
        cameras[name] = density[use], np.asarray(reference["plan_points"])[use]
    result = {
        "diagnostic_only": True,
        "manual_JSON_references_used": True,
        "automatic_outputs_unchanged": True,
        "per_camera_in_sample": {},
        "cross_camera_holdout": {},
    }
    for name, (x, y) in cameras.items():
        matrix = fit_similarity(x, y)
        result["per_camera_in_sample"][name] = errors(matrix, x, y)
        for other, (xx, yy) in cameras.items():
            if other != name:
                result["cross_camera_holdout"][f"fit_{name}_test_{other}"] = errors(matrix, xx, yy)
    x = np.concatenate([a for a, _ in cameras.values()])
    y = np.concatenate([b for _, b in cameras.values()])
    result["joint_in_sample"] = errors(fit_similarity(x, y), x, y)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
