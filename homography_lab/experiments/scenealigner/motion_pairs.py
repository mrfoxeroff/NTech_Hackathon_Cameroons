"""Appearance/time pairing and independent floor-plane relative geometry. No map labels."""

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
if __package__:
    from .floor_ray_geometry import floor_homography, project_floor, transform_points
    from .lens_geometry import distort_points, undistort_points
else:
    from floor_ray_geometry import floor_homography, project_floor, transform_points
    from lens_geometry import distort_points, undistort_points


def mutual_appearance(a, b, minimum=0.6, margin=0.06):
    if not a or not b:
        return []
    x, y = np.asarray([v["embedding"] for v in a]), np.asarray([v["embedding"] for v in b])
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    y = y / np.linalg.norm(y, axis=1, keepdims=True)
    similarities = x @ y.T
    result = []
    for i, j in enumerate(np.argmax(similarities, axis=1)):
        if np.argmax(similarities[:, j]) != i:
            continue
        score = float(similarities[i, j])
        alternatives_a = np.delete(similarities[i], j)
        alternatives_b = np.delete(similarities[:, j], i)
        gap = min(
            score - (alternatives_a.max() if len(alternatives_a) else -1.0),
            score - (alternatives_b.max() if len(alternatives_b) else -1.0),
        )
        if score >= minimum and gap >= margin:
            result.append((i, int(j), score, float(gap)))
    return result


def matching_floor_points(left, right, lenses):
    """Compare the same anatomical contact; average after plane projection, not in the image."""
    if left.get("foot") is None or right.get("foot") is None:
        return None
    if not lenses or left.get("keypoints") is None or right.get("keypoints") is None:
        return [left["foot"], right["foot"]]
    a, b = np.asarray(left["keypoints"]), np.asarray(right["keypoints"])
    common = [i for i in [15, 16] if min(a[i, 2], b[i, 2]) >= 0.55]
    if not common:
        return None
    ground = []
    homographies = []
    validities = []
    for row, keypoints in [(left, a), (right, b)]:
        lens = lenses[row["camera"]]
        h = floor_homography(lens["new_K"], lens["gravity"])
        points, valid = project_floor(keypoints[common, :2], lens, h)
        ground.append(points)
        homographies.append(h)
        validities.append(valid)
    valid = validities[0] & validities[1]
    if not valid.any():
        return None
    result = []
    for row, points, h in zip([left, right], ground, homographies):
        ideal = transform_points(np.mean(points[valid], axis=0)[None], np.linalg.inv(h))
        original, good = distort_points(ideal, lenses[row["camera"]])
        if not good[0]:
            return None
        result.append(original[0].tolist())
    return result


def pair_observations(
    observations, max_dt=0.18, minimum_support=4, appearance="osnet", require_feet=True, lenses=None
):
    tracks = defaultdict(list)
    for row in observations:
        tracks[row["track_id"]].append(row)
    grouped = defaultdict(lambda: defaultdict(list))
    for row in observations:
        if sum(v.get("foot") is not None for v in tracks[row["track_id"]]) < 3:
            continue
        nearby = [
            v["embedding"]
            for v in tracks[row["track_id"]]
            if abs(v["timestamp_sec"] - row["timestamp_sec"]) <= 1.0
        ]
        vector = np.mean(nearby, axis=0)
        vector /= max(np.linalg.norm(vector), 1e-9)
        if appearance == "dual":
            second = np.mean(
                [
                    v["embedding_intel"]
                    for v in tracks[row["track_id"]]
                    if abs(v["timestamp_sec"] - row["timestamp_sec"]) <= 1.0
                ],
                axis=0,
            )
            second /= max(np.linalg.norm(second), 1e-9)
            vector = np.r_[vector, second] / np.sqrt(2.0)
        row = row | {"embedding": (vector / max(np.linalg.norm(vector), 1e-9)).tolist()}
        # Global IDs are deliberately ignored even when a caller supplies them.
        grouped[(row["window"], row["split"], row["camera"])][row["timestamp_sec"]].append(row)
    candidates = []
    for window, split in sorted({(k[0], k[1]) for k in grouped}):
        cameras = sorted(k[2] for k in grouped if k[:2] == (window, split))
        for ca, cb in combinations(cameras, 2):
            left, right = grouped[(window, split, ca)], grouped[(window, split, cb)]
            times = np.array(sorted(right))
            options = []
            for ta in sorted(left):
                tb = float(times[np.argmin(abs(times - ta))])
                if abs(ta - tb) <= max_dt:
                    options.append((abs(ta - tb), ta, tb))
            used_left, used_right = set(), set()
            for dt, ta, tb in sorted(options):
                if ta in used_left or tb in used_right:
                    continue
                used_left.add(ta)
                used_right.add(tb)
                a, b = left[ta], right[tb]
                for i, j, similarity, gap in mutual_appearance(
                    a, b, minimum=0.45 if appearance == "dual" else 0.6
                ):
                    x, y = a[i], b[j]
                    foot_points = matching_floor_points(x, y, lenses)
                    if require_feet and foot_points is None:
                        continue
                    candidates.append(
                        {
                            "cameras": [ca, cb],
                            "window": window,
                            "split": split,
                            "timestamp_sec": (ta + tb) / 2,
                            "time_difference_sec": dt,
                            "appearance_cosine": similarity,
                            "appearance_margin": gap,
                            "track_ids": [x["track_id"], y["track_id"]],
                            "points": foot_points if foot_points is not None else [None, None],
                            "frame_ids": [x["frame_id"], y["frame_id"]],
                            "images": [x["image"], y["image"]],
                            "boxes": [x["box"], y["box"]],
                            "sizes": [x["size"], y["size"]],
                            "keypoints": [x.get("keypoints"), y.get("keypoints")],
                            "foot_confidences": [x["foot_confidence"], y["foot_confidence"]],
                        }
                    )
    votes = Counter((tuple(v["cameras"]), tuple(v["track_ids"])) for v in candidates)
    accepted = [
        v for v in candidates if votes[(tuple(v["cameras"]), tuple(v["track_ids"]))] >= minimum_support
    ]
    for row in accepted:
        row["track_pair_support"] = votes[(tuple(row["cameras"]), tuple(row["track_ids"]))]
    return accepted, {
        "frame_pairs_before_temporal_support": len(candidates),
        "pairs_after_temporal_support": len(accepted),
    }


def balanced_matches(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(tuple(row["track_ids"]), int(row["timestamp_sec"]))].append(row)
    return [
        max(v, key=lambda row: min(row["foot_confidences"]) * row["appearance_cosine"])
        for v in groups.values()
    ]


def relative_errors(matrix, a, b, ha, hb, lens_a, lens_b, original_a, original_b):
    predicted_b = transform_points(transform_points(a, matrix), np.linalg.inv(hb))
    predicted_a = transform_points(transform_points(b, np.linalg.inv(matrix)), np.linalg.inv(ha))
    ideal_a, _ = undistort_points(original_a, lens_a)
    ideal_b, _ = undistort_points(original_b, lens_b)
    errors_a = np.linalg.norm(predicted_a - ideal_a, axis=1)
    errors_b = np.linalg.norm(predicted_b - ideal_b, axis=1)
    normalized = np.maximum(errors_a / np.hypot(*lens_a["size"]), errors_b / np.hypot(*lens_b["size"]))
    return normalized, np.maximum(errors_a, errors_b)


def fit_relative_edge(rows, lens_a, lens_b):
    train, holdout = (
        balanced_matches([v for v in rows if v["split"] == "train"]),
        balanced_matches([v for v in rows if v["split"] == "holdout"]),
    )
    result = {
        "accepted": False,
        "train_supported": False,
        "method": "gravity_constrained_floor_similarity_RANSAC",
        "train_pairs": len(train),
        "holdout_pairs": len(holdout),
        "manual_plan_landmarks_used": False,
        "global_track_IDs_used": False,
    }
    if len(train) < 12:
        return result | {"reason": "insufficient_training_pairs"}
    ha = floor_homography(lens_a["new_K"], lens_a["gravity"])
    hb = floor_homography(lens_b["new_K"], lens_b["gravity"])

    def coordinates(items):
        pa, pb = np.array([v["points"][0] for v in items]), np.array([v["points"][1] for v in items])
        a, va = project_floor(pa, lens_a, ha)
        b, vb = project_floor(pb, lens_b, hb)
        return a, b, va & vb, pa, pb

    a, b, good, pa, pb = coordinates(train)
    if good.sum() < 12:
        return result | {"reason": "insufficient_valid_floor_rays"}
    a, b, pa, pb = a[good], b[good], pa[good], pb[good]
    valid_rows = [v for v, g in zip(train, good) if g]
    if max(np.ptp(a, axis=0)) < 0.15 or max(np.ptp(b, axis=0)) < 0.15:
        return result | {"reason": "insufficient_motion_extent"}
    matrix, mask = cv2.estimateAffinePartial2D(
        a, b, method=cv2.RANSAC, ransacReprojThreshold=0.1, maxIters=10000, confidence=0.999
    )
    if matrix is None:
        return result | {"reason": "relative_fit_failed"}
    matrix = np.vstack((matrix, [0.0, 0, 1.0]))
    inliers = mask.ravel().astype(bool)
    normalized, pixels = relative_errors(matrix, a, b, ha, hb, lens_a, lens_b, pa, pb)
    trajectories = {tuple(v["track_ids"]) for v, use in zip(valid_rows, inliers) if use}
    result.update(
        floor_similarity=matrix.tolist(),
        ideal_a_to_b=(np.linalg.inv(hb) @ matrix @ ha).tolist(),
        train_inliers=int(inliers.sum()),
        train_inlier_fraction=float(inliers.mean()),
        supporting_trajectory_segments=len(trajectories),
        train_median_symmetric_px=float(np.median(pixels[inliers])) if inliers.any() else None,
        train_p95_symmetric_px=float(np.quantile(pixels[inliers], 0.95)) if inliers.any() else None,
        training_inlier_keys=[[v["frame_ids"], v["track_ids"]] for v, use in zip(valid_rows, inliers) if use],
    )
    reasons = []
    if inliers.sum() < 12 or inliers.mean() < 0.6:
        reasons.append("insufficient_relative_inliers")
    if len(trajectories) < 2:
        reasons.append("only_one_trajectory_segment")
    result["train_supported"] = not reasons
    if len(holdout) < 8:
        reasons.append("insufficient_independent_time_holdout")
    else:
        a, b, good, pa, pb = coordinates(holdout)
        normalized, pixels = relative_errors(
            matrix, a[good], b[good], ha, hb, lens_a, lens_b, pa[good], pb[good]
        )
        if len(pixels) < 8:
            reasons.append("insufficient_valid_holdout_rays")
        else:
            result["holdout"] = {
                "pairs": len(pixels),
                "median_symmetric_px": float(np.median(pixels)),
                "p95_symmetric_px": float(np.quantile(pixels, 0.95)),
                "median_diagonal_fraction": float(np.median(normalized)),
                "p95_diagonal_fraction": float(np.quantile(normalized, 0.95)),
            }
            if np.median(normalized) > 0.015 or np.quantile(normalized, 0.95) > 0.04:
                reasons.append("relative_geometry_does_not_generalize")
    result["accepted"] = not reasons
    result["reason"] = (
        "relative_floor_alignment_supported_not_absolute_location" if not reasons else ";".join(reasons)
    )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--observations", type=Path, required=True)
    p.add_argument("--lenses", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--appearance", choices=["osnet", "dual"], default="osnet")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setRNGSeed(42)
    observations = [json.loads(line) for line in args.observations.read_text().splitlines() if line]
    lenses = json.loads(args.lenses.read_text())["cameras"]
    pairs, diagnostic = pair_observations(observations, appearance=args.appearance, lenses=lenses)
    report = {
        "manual_plan_landmarks_used": False,
        "global_track_IDs_used": False,
        "appearance_and_time_only_pairing": True,
        "appearance": args.appearance,
        "diagnostic": diagnostic,
        "pairs": pairs,
        "edges": {},
        "thresholds": {
            "appearance_cosine": 0.45 if args.appearance == "dual" else 0.6,
            "appearance_margin": 0.06,
            "time_tolerance_sec": 0.18,
            "temporal_pair_support": 4,
            "relative_RANSAC_camera_height_units": 0.1,
            "holdout_median_diagonal_fraction": 0.015,
            "holdout_p95_diagonal_fraction": 0.04,
        },
    }
    cameras = sorted({v["camera"] for v in observations})
    for a, b in combinations(cameras, 2):
        current = [v for v in pairs if v["cameras"] == [a, b]]
        report["edges"][a + "-" + b] = fit_relative_edge(current, lenses[a], lenses[b])
        report["edges"][a + "-" + b]["cameras"] = [a, b]
        print(
            a,
            b,
            {
                k: v
                for k, v in report["edges"][a + "-" + b].items()
                if k not in ["floor_similarity", "ideal_a_to_b", "training_inlier_keys"]
            },
            flush=True,
        )
    (args.output / "pairs.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
