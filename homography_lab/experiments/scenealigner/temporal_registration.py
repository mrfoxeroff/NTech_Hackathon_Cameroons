"""Camera stability from image features; no floorplan or manual landmarks."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def fit_registration(source, target, shape, scale):
    count = len(source)
    result = {"matches": count, "accepted": False}
    if count < 80:
        return result | {"reason": "insufficient_matches"}
    order = np.random.default_rng(42).permutation(count)
    check = order[::4]
    fit = np.setdiff1d(np.arange(count), check)
    matrix, inliers = cv2.estimateAffinePartial2D(
        source[fit],
        target[fit],
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0,
        maxIters=5000,
        confidence=0.999,
    )
    if matrix is None:
        return result | {"reason": "fit_failed"}
    predictions = np.column_stack((source[check], np.ones(len(check)))) @ matrix.T
    residuals = np.linalg.norm(predictions - target[check], axis=1)
    good = inliers.ravel().astype(bool)
    hull = (
        cv2.contourArea(cv2.convexHull(source[fit][good].astype(np.float32))) / (shape[0] * shape[1])
        if good.sum() >= 3
        else 0.0
    )
    corners = np.array(
        [[0.0, 0.0], [shape[1] - 1.0, 0], [shape[1] - 1.0, shape[0] - 1.0], [0, shape[0] - 1.0]]
    )
    moved = np.column_stack((corners, np.ones(4))) @ matrix.T
    displacement = float(np.median(np.linalg.norm(moved - corners, axis=1)) / scale)
    relative = displacement / (np.hypot(*shape) / scale)
    original = matrix.copy()
    original[:, 2] /= scale
    accepted = good.sum() >= 50 and hull >= 0.2 and (residuals <= 2).mean() >= 0.5 and relative < 0.003
    return result | {
        "accepted": bool(accepted),
        "matrix": original.tolist(),
        "fit_inliers": int(good.sum()),
        "source_hull_fraction": float(hull),
        "holdout_matches": len(check),
        "holdout_fraction_within_2px_reduced": float(np.mean(residuals <= 2)),
        "holdout_median_residual_original_px": float(np.median(residuals) / scale),
        "median_corner_displacement_original_px": displacement,
        "relative_motion": relative,
        "reason": "small_motion_supported" if accepted else "camera_motion_or_insufficient_support",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output file")
    cv2.setNumThreads(2)
    cv2.setRNGSeed(42)
    samples = json.loads(args.samples.read_text(encoding="utf-8"))
    original = cv2.imread(str(args.background), 0)
    scale = min(1.0, 1280 / original.shape[1])
    background = cv2.resize(original, None, fx=scale, fy=scale)
    mask = np.full(background.shape, 255, np.uint8)
    mask[: round(0.08 * len(mask))] = 0
    detector = cv2.SIFT_create(nfeatures=3500)
    ka, da = detector.detectAndCompute(background, mask)

    def register(path):
        image = cv2.imread(str(path), 0)
        if image.shape != original.shape:
            return {"accepted": False, "reason": "resolution_changed"}
        image = cv2.resize(image, background.shape[::-1])
        kb, db = detector.detectAndCompute(image, mask)
        if da is None or db is None:
            return {"accepted": False, "reason": "no_descriptors"}
        matcher = cv2.BFMatcher()
        fw = [a for a, b in matcher.knnMatch(da, db, k=2) if a.distance < 0.7 * b.distance]
        bw = {
            a.queryIdx: a.trainIdx for a, b in matcher.knnMatch(db, da, k=2) if a.distance < 0.7 * b.distance
        }
        matched = [a for a in fw if bw.get(a.trainIdx) == a.queryIdx]
        source = np.array([ka[m.queryIdx].pt for m in matched], np.float32).reshape(-1, 2)
        target = np.array([kb[m.trainIdx].pt for m in matched], np.float32).reshape(-1, 2)
        return fit_registration(source, target, background.shape, scale)

    report = {
        "manual_landmarks_used": False,
        "method": "mutual_SIFT_070_affine_RANSAC_75_25_holdout",
        "reference_image_sha256": hashlib.sha256(args.background.read_bytes()).hexdigest(),
        "cases": {},
        "probes": [],
    }
    for case in samples["cases"]:
        result = register(case["image"])
        report["cases"][case["id"]] = result
        print(
            case["id"], result["accepted"], result.get("median_corner_displacement_original_px"), flush=True
        )
    for index, probe in enumerate(samples["probes"]):
        result = register(probe["path"])
        report["probes"].append({"actual_seconds": probe["actual_seconds"], "registration": result})
        if index % 15 == 0:
            print("Camera stability probe", index, result["accepted"], flush=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
