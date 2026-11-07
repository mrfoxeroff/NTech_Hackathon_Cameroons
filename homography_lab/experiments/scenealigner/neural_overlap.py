"""LoFTR indoor overlap graph with held-out epipolar checks; no plan or landmarks."""

import argparse
import hashlib
import json
from copy import deepcopy
from itertools import combinations
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def sampson_distance(matrix, x, y):
    a, b = np.column_stack((x, np.ones(len(x)))), np.column_stack((y, np.ones(len(y))))
    fa, ftb = a @ matrix.T, b @ matrix
    return abs(np.sum(b * fa, axis=1)) / np.sqrt(np.sum(fa[:, :2] ** 2 + ftb[:, :2] ** 2, axis=1) + 1e-12)


def assess_pair(x, y, shape_a, shape_b):
    """Thresholds fixed before the office run; RANSAC never sees check matches."""
    count = len(x)
    result = {"confident_matches": count, "accepted_overlap_edge": False, "verified_inliers": 0}
    good = np.zeros(count, bool)
    if count < 40:
        return result | {"reason": "too_few_confident_matches"}, good
    order = np.random.default_rng(42).permutation(count)
    check = order[::4]
    fit = np.setdiff1d(np.arange(count), check)
    matrix, _ = cv2.findFundamentalMat(x[fit], y[fit], cv2.USAC_MAGSAC, 1.5, 0.999, 10000)
    if matrix is None or matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        return result | {"reason": "epipolar_fit_failed"}, good
    distances = sampson_distance(matrix, x, y)
    good = distances <= 1.5
    coverage = [
        float(cv2.contourArea(cv2.convexHull(points[good].astype(np.float32))) / (shape[0] * shape[1]))
        if good.sum() >= 3
        else 0.0
        for points, shape in [(x, shape_a), (y, shape_b)]
    ]
    cells = [
        len(set(map(tuple, np.minimum((points[good] / [shape[1], shape[0]] * 4).astype(int), 3))))
        for points, shape in [(x, shape_a), (y, shape_b)]
    ]
    holdout_ratio = float(good[check].mean())
    accepted = (
        good.sum() >= 30
        and good.mean() >= 0.4
        and holdout_ratio >= 0.4
        and min(coverage) >= 0.02
        and min(cells) >= 4
    )
    result |= {
        "verified_inliers": int(good.sum()),
        "inlier_ratio": float(good.mean()),
        "fundamental_matrix": matrix.tolist(),
        "holdout_matches": len(check),
        "holdout_inlier_ratio": holdout_ratio,
        "inlier_hull_fractions": coverage,
        "occupied_grid_cells": cells,
        "median_sampson_px": float(np.median(distances[good])) if good.any() else None,
        "accepted_overlap_edge": bool(accepted),
        "reason": "verified" if accepted else "insufficient_geometric_support",
    }
    return result, good


def connected_groups(names, pairs):
    links = {name: set() for name in names}
    for pair in pairs.values():
        if pair["accepted_overlap_edge"]:
            a, b = pair["cameras"]
            links[a].add(b)
            links[b].add(a)
    remaining, groups = set(names), []
    while remaining:
        group, pending = set(), [min(remaining)]
        while pending:
            node = pending.pop()
            if node not in group:
                group.add(node)
                pending.extend(links[node] - group)
        remaining -= group
        groups.append(sorted(group))
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=HERE / "cache/loftr_indoor_ds_new.ckpt")
    parser.add_argument("--variant", choices=["indoor_new", "outdoor"], default="indoor_new")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--tiles", action="store_true")
    parser.add_argument("--augment-sift", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from kornia.feature import LoFTR
    from kornia.feature.loftr.loftr import default_cfg

    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    cv2.setRNGSeed(42)
    config = deepcopy(default_cfg)
    config["coarse"]["temp_bug_fix"] = args.variant == "indoor_new"
    model = LoFTR(pretrained=None, config=config)
    model.load_state_dict(
        torch.load(args.weights, map_location="cpu", weights_only=True)["state_dict"], strict=True
    )
    model = model.eval().cuda()
    views, tiles, sift_features = {}, {}, {}
    for path in sorted(args.images.glob("*.jpg")):
        original = cv2.imread(str(path))
        image = original
        width = 640
        height = round(image.shape[0] / image.shape[1] * width / 8) * 8
        image = cv2.resize(image, (width, height))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        views[path.stem] = image, torch.tensor(gray / 255.0, dtype=torch.float32, device="cuda")[None, None]
        boxes = [(0, 0, width, height)]
        if args.tiles:
            tw, th = round(width * 0.65), round(height * 0.65)
            boxes += [(x, y, tw, th) for y in [0, height - th] for x in [0, width - tw]]
        tiles[path.stem] = []
        for x, y, w, h in boxes:
            resized = cv2.resize(gray[y : y + h, x : x + w], (640, 480))
            tiles[path.stem].append(
                (
                    torch.tensor(resized / 255.0, dtype=torch.float32, device="cuda")[None, None],
                    np.array([w / 640, h / 480]),
                    np.array([x, y]),
                )
            )
        if args.augment_sift:
            mask = np.full(original.shape[:2], 255, np.uint8)
            mask[: round(0.08 * len(mask))] = 0
            kp, desc = cv2.SIFT_create(nfeatures=18000, contrastThreshold=0.02).detectAndCompute(
                cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), mask
            )
            sift_features[path.stem] = (
                kp,
                desc,
                np.array([width / original.shape[1], height / original.shape[0]]),
            )
    report = {
        "status": "running",
        "method": "LoFTR_" + args.variant + "_heldout_epipolar",
        "manual_landmarks_used": False,
        "tiled_matching": args.tiles,
        "augmented_full_resolution_sift": args.augment_sift,
        "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "thresholds": {
            "confidence": args.confidence,
            "minimum_matches": 40,
            "minimum_inliers": 30,
            "sampson_px": 1.5,
            "min_inlier_ratio": 0.4,
            "min_holdout_ratio": 0.4,
            "min_hull_fraction": 0.02,
            "min_4x4_grid_cells": 4,
        },
        "interpretation": "Detected overlap evidence; missing edges mean insufficient evidence, not proof of no overlap.",
        "pairs": {},
    }
    start = perf_counter()
    for first, second in combinations(views, 2):
        a, ta = views[first]
        b, tb = views[second]
        found, total_matches, tile_evidence = [], 0, []
        sx, sy = np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
        if args.augment_sift:
            ka, da, sa = sift_features[first]
            kb, db, sb = sift_features[second]
            matcher = cv2.BFMatcher()
            forward = [m for m, n in matcher.knnMatch(da, db, k=2) if m.distance < 0.75 * n.distance]
            backward = {
                m.queryIdx: m.trainIdx
                for m, n in matcher.knnMatch(db, da, k=2)
                if m.distance < 0.75 * n.distance
            }
            matched = [m for m in forward if backward.get(m.trainIdx) == m.queryIdx]
            sx = np.asarray([np.array(ka[m.queryIdx].pt) * sa for m in matched], np.float32).reshape(-1, 2)
            sy = np.asarray([np.array(kb[m.trainIdx].pt) * sb for m in matched], np.float32).reshape(-1, 2)
        for at, ascale, aoffset in tiles[first]:
            for bt, bscale, boffset in tiles[second]:
                with torch.inference_mode():
                    prediction = model({"image0": at, "image1": bt})
                x = prediction["keypoints0"].cpu().numpy() * ascale + aoffset
                y = prediction["keypoints1"].cpu().numpy() * bscale + boffset
                conf = prediction["confidence"].cpu().numpy()
                keep = (
                    (conf >= args.confidence) & (x[:, 1] > 0.08 * a.shape[0]) & (y[:, 1] > 0.08 * b.shape[0])
                )
                total_matches += len(keep)
                found.extend((float(c), p, q) for p, q, c in zip(x[keep], y[keep], conf[keep]))
                tx, ty, tc = x[keep].astype(np.float32), y[keep].astype(np.float32), conf[keep]
                evidence, tile_inliers = assess_pair(tx, ty, a.shape, b.shape)
                if args.augment_sift and "fundamental_matrix" in evidence:
                    independent = sampson_distance(np.asarray(evidence["fundamental_matrix"]), sx, sy) <= 1.5
                    evidence["independent_sift_matches"] = len(sx)
                    evidence["independent_sift_inliers"] = int(independent.sum())
                    if independent.sum() < 8:
                        evidence["accepted_overlap_edge"] = False
                        evidence["reason"] = "failed_independent_sift_check"
                tile_evidence.append((evidence, tx, ty, tc, tile_inliers))
        if args.augment_sift:
            found += [(0.5, p, q) for p, q in zip(sx, sy)]
        # Repeated observations of the same endpoints must not inflate support.
        unique, seen_a, seen_b = [], set(), set()
        for item in sorted(found, key=lambda value: -value[0]):
            ca, cb = tuple(np.rint(item[1] / 3).astype(int)), tuple(np.rint(item[2] / 3).astype(int))
            if ca not in seen_a and cb not in seen_b:
                unique.append(item)
                seen_a.add(ca)
                seen_b.add(cb)
        x = np.asarray([item[1] for item in unique]).reshape(-1, 2).astype(np.float32)
        y = np.asarray([item[2] for item in unique]).reshape(-1, 2).astype(np.float32)
        conf = np.asarray([item[0] for item in unique])
        # Select a coherent crop pair before pooling: unrelated crop combinations
        # otherwise dilute the true overlap with many confident floor-pattern matches.
        ranked = sorted(
            tile_evidence,
            key=lambda item: (
                item[0]["accepted_overlap_edge"],
                item[0].get("independent_sift_inliers", 0),
                item[0]["verified_inliers"],
            ),
            reverse=True,
        )
        if ranked:
            score, x, y, conf, inliers = ranked[0]
            score["tile_pairs_tested"] = len(ranked)
            score["tile_pairs_verified"] = sum(item[0]["accepted_overlap_edge"] for item in ranked)
        else:
            score, inliers = assess_pair(x, y, a.shape, b.shape)
        score["cameras"] = [first, second]
        score["total_network_matches"] = total_matches
        report["pairs"][f"{first}-{second}"] = score
        np.savez_compressed(
            args.output / f"{first}-{second}.npz", points0=x, points1=y, confidence=conf, inliers=inliers
        )
        if inliers.any():
            preview = np.concatenate((a, b), axis=1)
            indexes = np.flatnonzero(inliers)[:: max(1, int(inliers.sum()) // 60)][:60]
            for index in indexes:
                p, q = np.rint(x[index]).astype(int), np.rint(y[index]).astype(int) + [a.shape[1], 0]
                cv2.line(preview, tuple(p), tuple(q), (80, 220, 80), 1, cv2.LINE_AA)
            cv2.imwrite(str(args.output / f"{first}-{second}.jpg"), preview)
        print(first, second, score["verified_inliers"], score["accepted_overlap_edge"], flush=True)
        (args.output / "overlap.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["groups"] = connected_groups(views, report["pairs"])
    report["status"] = "complete"
    report["elapsed_sec"] = perf_counter() - start
    (args.output / "overlap.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Groups:", report["groups"], flush=True)


if __name__ == "__main__":
    main()
