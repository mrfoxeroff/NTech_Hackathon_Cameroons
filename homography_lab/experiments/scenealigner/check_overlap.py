"""Evidence of image overlap, without consulting floorplan calibration JSONs."""

import argparse
import json
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    cv2.setNumThreads(4)
    cv2.setRNGSeed(42)
    sift = cv2.SIFT_create(nfeatures=5000)
    views = {}
    for path in sorted(args.images.glob("*.jpg")):
        image = cv2.imread(str(path))
        scale = min(1.0, 1280 / max(image.shape[:2]))
        image = cv2.resize(image, None, fx=scale, fy=scale)
        mask = np.full(image.shape[:2], 255, np.uint8)
        mask[: int(0.08 * mask.shape[0])] = 0  # suppress timestamp overlays
        keypoints, descriptors = sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), mask)
        views[path.stem] = image, keypoints, descriptors
    report = {
        "method": "mutual_SIFT_ratio_075_fundamental_USAC_MAGSAC",
        "manual_landmarks_used": False,
        "interpretation": "Overlap evidence only; absence of matches does not prove disconnection.",
        "thresholds": {"min_inliers": 20, "min_inlier_ratio": 0.35, "min_hull_fraction_both": 0.02},
        "pairs": {},
    }
    links = {name: set() for name in views}
    for first, second in combinations(views, 2):
        a, ka, da = views[first]
        b, kb, db = views[second]
        matches = []
        if da is not None and db is not None and min(len(da), len(db)) >= 2:
            matcher = cv2.BFMatcher()
            fwd = [
                m
                for pair in matcher.knnMatch(da, db, k=2)
                if len(pair) == 2
                for m, n in [pair]
                if m.distance < 0.75 * n.distance
            ]
            back = {
                m.queryIdx: m.trainIdx
                for pair in matcher.knnMatch(db, da, k=2)
                if len(pair) == 2
                for m, n in [pair]
                if m.distance < 0.75 * n.distance
            }
            matches = [m for m in fwd if back.get(m.trainIdx) == m.queryIdx]
        item = {"mutual_matches": len(matches), "verified_inliers": 0, "accepted_overlap_edge": False}
        if len(matches) >= 12:
            x = np.float32([ka[m.queryIdx].pt for m in matches])
            y = np.float32([kb[m.trainIdx].pt for m in matches])
            matrix, inliers = cv2.findFundamentalMat(x, y, cv2.USAC_MAGSAC, 1.5, 0.999, 10000)
            if matrix is not None and inliers is not None:
                inliers = inliers.ravel().astype(bool)
                count = int(inliers.sum())
                coverage = [
                    float(
                        cv2.contourArea(cv2.convexHull(points[inliers])) / (image.shape[0] * image.shape[1])
                    )
                    if count >= 3
                    else 0.0
                    for points, image in [(x, a), (y, b)]
                ]
                accepted = count >= 20 and inliers.mean() >= 0.35 and min(coverage) >= 0.02
                item |= {
                    "verified_inliers": count,
                    "inlier_ratio": float(inliers.mean()),
                    "inlier_hull_fractions": coverage,
                    "accepted_overlap_edge": bool(accepted),
                }
                if accepted:
                    links[first].add(second)
                    links[second].add(first)
                    preview = cv2.drawMatches(
                        a,
                        ka,
                        b,
                        kb,
                        [m for m, ok in zip(matches, inliers) if ok][:80],
                        None,
                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                    )
                    cv2.imwrite(str(args.output / f"{first}-{second}.jpg"), preview)
        report["pairs"][f"{first}-{second}"] = item
    remaining = set(views)
    groups = []
    while remaining:
        group, pending = set(), [min(remaining)]
        while pending:
            name = pending.pop()
            if name not in group:
                group.add(name)
                pending.extend(links[name] - group)
        remaining -= group
        groups.append(sorted(group))
    report["components_with_detected_overlap"] = groups
    (args.output / "overlap.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
