"""Evaluate predictions against GitHub landmarks; never fit on those landmarks."""

import argparse
import json
from pathlib import Path

import numpy as np
from lens_geometry import project_calibration


def evaluate_camera(prediction, reference):
    if (
        prediction["camera_size"] != reference["camera_size"]
        or prediction["plan_size"] != reference["plan_size"]
    ):
        raise ValueError("Prediction/reference resolution mismatch")
    h = np.asarray(prediction["homography"], dtype=float)
    if h.shape != (3, 3) or not np.isfinite(h).all() or np.linalg.matrix_rank(h) != 3:
        raise ValueError("Invalid predicted homography")
    src, dst = (
        np.asarray(reference["camera_points"], dtype=float),
        np.asarray(reference["plan_points"], dtype=float),
    )
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2 or not np.isfinite([src, dst]).all():
        raise ValueError("Invalid reference point arrays")
    xy, valid = project_calibration(src, prediction)
    errors = np.linalg.norm(xy - dst, axis=1)
    use = np.asarray(reference.get("inliers", [True] * len(src)), dtype=bool)
    if len(use) != len(src) or not use.any():
        raise ValueError("No usable reference landmarks")
    scores = errors[use]
    finite = bool(np.isfinite(scores).all())
    return {
        "reference_points": len(src),
        "evaluated_reference_inliers": int(use.sum()),
        "excluded_reference_outliers": int((~use).sum()),
        "errors_px": [float(e) if np.isfinite(e) else None for e in errors],
        "median_error_px": float(np.median(scores)) if finite else None,
        "p95_error_px": float(np.quantile(scores, 0.95)) if finite else None,
        "fraction_within_10px": float(np.mean(scores <= 10)),
        "unprojectable_reference_inliers": int((~valid & use).sum()),
        "reference_kind": "existing_manual_JSON_correspondences_not_independent_survey",
        "landmarks_used_for_inference": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = {"cameras": {}, "evaluation_only": True, "thresholds_tuned_on_reference": False}
    for path in sorted(args.predictions.glob("cam_*-calibration.json")):
        camera = path.name.removesuffix("-calibration.json")
        ref = args.reference / f"{camera}_calibration.json"
        if not ref.is_file():
            continue
        prediction = json.loads(path.read_text(encoding="utf-8"))
        reference = json.loads(ref.read_text(encoding="utf-8"))
        report["cameras"][camera] = evaluate_camera(prediction, reference)
    expected = {
        path.name.removesuffix("_calibration.json") for path in args.reference.glob("cam_*_calibration.json")
    }
    if not expected:
        raise ValueError("No reference cameras found")
    report["expected_cameras"] = sorted(expected)
    report["missing_predictions"] = sorted(expected - set(report["cameras"]))
    report["coverage_ratio"] = len(report["cameras"]) / len(expected)
    report["status"] = "evaluated" if report["cameras"] else "no_predictions"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
