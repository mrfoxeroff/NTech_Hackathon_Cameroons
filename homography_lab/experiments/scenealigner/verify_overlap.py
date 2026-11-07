"""Cross-method overlap verification for small shared image regions; no calibration labels."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from neural_overlap import connected_groups, sampson_distance
from scipy.stats import binom


def independent_support(matrix, x, y):
    n = len(x)
    if n < 8:
        return {"inliers": 0, "matches": n, "null_probability": 1.0, "binomial_tail_approx": 1.0}
    inliers = int((sampson_distance(matrix, x, y) <= 1.5).sum())
    a, b = np.column_stack((x, np.ones(n))), np.column_stack((y, np.ones(n)))
    fa, ftb = a @ matrix.T, b @ matrix
    numerator = (b @ fa.T).T ** 2
    denominator = np.sum(fa[:, :2] ** 2, axis=1)[:, None] + np.sum(ftb[:, :2] ** 2, axis=1)[None, :] + 1e-12
    cross = numerator / denominator <= 1.5**2
    off_diagonal = ~np.eye(n, dtype=bool)
    probability = float((cross[off_diagonal].sum() + 1) / (n * (n - 1) + 2))
    return {
        "inliers": inliers,
        "matches": n,
        "null_probability": probability,
        "binomial_tail_approx": float(binom.sf(inliers - 1, n, probability)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--network-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output file")
    report = json.loads(args.network_report.read_text(encoding="utf-8"))
    if "groups" not in report:
        raise ValueError("Network matching must finish before verification")
    cv2.setNumThreads(4)
    features = {}
    for path in sorted(args.images.glob("*.jpg")):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        mask = np.full(image.shape, 255, np.uint8)
        mask[: round(0.08 * len(mask))] = 0
        points, desc = cv2.SIFT_create(nfeatures=18000, contrastThreshold=0.02).detectAndCompute(image, mask)
        scale = np.array(
            [640 / image.shape[1], round(image.shape[0] / image.shape[1] * 640 / 8) * 8 / image.shape[0]]
        )
        features[path.stem] = points, desc, scale
    for name, item in report["pairs"].items():
        a, b = item["cameras"]
        ka, da, sa = features[a]
        kb, db, sb = features[b]
        matcher = cv2.BFMatcher()
        fw = [m for m, n in matcher.knnMatch(da, db, k=2) if m.distance < 0.75 * n.distance]
        bw = {
            m.queryIdx: m.trainIdx for m, n in matcher.knnMatch(db, da, k=2) if m.distance < 0.75 * n.distance
        }
        pairs = [m for m in fw if bw.get(m.trainIdx) == m.queryIdx]
        x = np.array([ka[m.queryIdx].pt for m in pairs], np.float32).reshape(-1, 2) * sa
        y = np.array([kb[m.trainIdx].pt for m in pairs], np.float32).reshape(-1, 2) * sb
        evidence = (
            independent_support(np.asarray(item["fundamental_matrix"]), x, y)
            if "fundamental_matrix" in item
            else {}
        )
        item["independent_verification"] = evidence
        item["strict_ratio_policy_accepted"] = item["accepted_overlap_edge"]
        spatial = (
            min(item.get("inlier_hull_fractions", [0])) >= 0.02
            and min(item.get("occupied_grid_cells", [0])) >= 4
        )
        holdout = round(item.get("holdout_inlier_ratio", 0) * item.get("holdout_matches", 0)) >= 4
        supported = (
            spatial
            and holdout
            and item["verified_inliers"] >= 30
            and evidence.get("inliers", 0) >= 8
            and evidence.get("binomial_tail_approx", 1) <= 1e-4
        )
        item["accepted_overlap_edge"] = bool(supported)
        item["reason"] = "cross_method_geometric_support" if supported else "unconfirmed_overlap"
        print(name, evidence.get("inliers", 0), evidence.get("binomial_tail_approx"), supported, flush=True)
    report["groups"] = connected_groups(features, report["pairs"])
    report["method"] = "tiled_LoFTR_independent_SIFT_geometry_verification"
    report["verification_policy"] = {
        "minimum_network_inliers": 30,
        "minimum_network_holdout_inliers": 4,
        "minimum_independent_SIFT_inliers": 8,
        "max_approx_binomial_tail": 1e-4,
        "note": "Approximate shuffled-endpoint null, not a formal false-positive guarantee. No reference JSON tuning.",
    }
    report["source_report"] = str(args.network_report.resolve())
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Verified groups:", report["groups"])


if __name__ == "__main__":
    main()
