"""Relative pose from moving body joints, floor scale from the few jointly visible feet."""

import argparse
import json
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from floor_ray_geometry import transform_points
from lens_geometry import undistort_points
from motion_pairs import balanced_matches, pair_observations


def joint_correspondences(rows, lens_a, lens_b):
    a, b, owners = [], [], []
    for row_index, row in enumerate(rows):
        if row.get("keypoints") is None or any(v is None for v in row["keypoints"]):
            continue
        ka, kb = np.asarray(row["keypoints"][0]), np.asarray(row["keypoints"][1])
        for joint in [5, 6, 11, 12, 13, 14, 15, 16]:
            if min(ka[joint, 2], kb[joint, 2]) >= 0.6:
                a.append(ka[joint, :2])
                b.append(kb[joint, :2])
                owners.append(row_index)
    if not a:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty(0, int)
    a, va = undistort_points(np.asarray(a), lens_a)
    b, vb = undistort_points(np.asarray(b), lens_b)
    a = (a - np.asarray(lens_a["new_K"])[:2, 2]) / np.diag(lens_a["new_K"])[:2]
    b = (b - np.asarray(lens_b["new_K"])[:2, 2]) / np.diag(lens_b["new_K"])[:2]
    good = va & vb
    return a[good], b[good], np.asarray(owners)[good]


def epipolar_errors(e, a, b):
    ah = np.column_stack((a, np.ones(len(a))))
    bh = np.column_stack((b, np.ones(len(b))))
    ea = ah @ e.T
    etb = bh @ e
    return abs(np.sum(bh * ea, axis=1)) / np.sqrt(np.sum(ea[:, :2] ** 2 + etb[:, :2] ** 2, axis=1) + 1e-12)


def floor_error(h, rows, lens_a, lens_b):
    if not rows:
        return np.empty(0), np.empty(0)
    a, va = undistort_points(np.asarray([r["points"][0] for r in rows]), lens_a)
    b, vb = undistort_points(np.asarray([r["points"][1] for r in rows]), lens_b)
    pa = transform_points(b, np.linalg.inv(h))
    pb = transform_points(a, h)
    ea = np.linalg.norm(pa - a, axis=1)
    eb = np.linalg.norm(pb - b, axis=1)
    normalized = np.maximum(ea / np.hypot(*lens_a["size"]), eb / np.hypot(*lens_b["size"]))
    pixels = np.maximum(ea, eb)
    normalized[~(va & vb)] = np.inf
    pixels[~(va & vb)] = np.inf
    return normalized, pixels


def fit_body_edge(rows, lens_a, lens_b):
    training = balanced_matches([r for r in rows if r["split"] == "train"])
    holdout = balanced_matches([r for r in rows if r["split"] == "holdout"])
    result = {
        "method": "moving_body_essential_matrix_plus_visible_foot_floor_scale",
        "train_supported": False,
        "accepted": False,
        "train_body_frame_pairs": len(training),
        "holdout_body_frame_pairs": len(holdout),
        "manual_plan_landmarks_used": False,
        "global_track_IDs_used": False,
        "lenses": [lens_a, lens_b],
    }
    a, b, owners = joint_correspondences(training, lens_a, lens_b)
    if len(a) < 40 or len(training) < 8:
        return result | {"reason": "insufficient_training_body_correspondences"}
    e, mask = cv2.findEssentialMat(
        a, b, np.eye(3), method=cv2.RANSAC, prob=0.999, threshold=0.006, maxIters=10000
    )
    if e is None or e.shape != (3, 3):
        return result | {"reason": "essential_fit_failed_or_ambiguous"}
    count, rotation, translation, cheirality = cv2.recoverPose(e, a, b, np.eye(3), mask=mask.copy())
    use = cheirality.ravel() > 0
    ids = {tuple(training[i]["track_ids"]) for i in owners[use]}
    result.update(
        essential_matrix=e.tolist(),
        relative_rotation=rotation.tolist(),
        relative_translation_direction=translation.ravel().tolist(),
        body_inliers=int(use.sum()),
        body_inlier_fraction=float(use.mean()),
        body_trajectory_segments=len(ids),
    )
    ga = np.asarray(lens_a["gravity"], float)
    ga /= np.linalg.norm(ga)
    gb = np.asarray(lens_b["gravity"], float)
    gb /= np.linalg.norm(gb)
    gravity_angle = float(np.degrees(np.arccos(np.clip((rotation @ ga) @ gb, -1, 1))))
    result["gravity_disagreement_degrees"] = gravity_angle
    valid_frames = {training[i]["frame_ids"][0] for i in owners[use]}
    feet = balanced_matches(
        [
            r
            for r in rows
            if r["split"] == "train"
            and all(v is not None for v in r["points"])
            and r["frame_ids"][0] in valid_frames
        ]
    )
    validation_feet = balanced_matches(
        [r for r in rows if r["split"] == "holdout" and all(v is not None for v in r["points"])]
    )
    result.update(train_pairs=len(feet), holdout_pairs=len(validation_feet))
    reasons = []
    if use.sum() < 30 or use.mean() < 0.4:
        reasons.append("weak_body_epipolar_support")
    if len(ids) < 2:
        reasons.append("only_one_body_trajectory_segment")
    if gravity_angle > 20:
        reasons.append("gravity_disagrees_with_body_pose")
    if len(feet) < 3:
        return result | {"reason": ";".join(reasons + ["insufficient_visible_floor_scale_pairs"])}
    pa, va = undistort_points(np.asarray([r["points"][0] for r in feet]), lens_a)
    pb, vb = undistort_points(np.asarray([r["points"][1] for r in feet]), lens_b)
    ra = np.column_stack((pa, np.ones(len(pa)))) @ np.linalg.inv(np.asarray(lens_a["new_K"])).T
    rb = np.column_stack((pb, np.ones(len(pb)))) @ np.linalg.inv(np.asarray(lens_b["new_K"])).T
    depth = ra @ ga
    valid = va & vb & (depth > 0.03)
    x = ra / np.maximum(depth[:, None], 0.03)
    cross_t = np.cross(rb, translation.ravel())
    cross_r = np.cross(rb, x @ rotation.T)
    denominator = np.sum(cross_t**2, axis=1)
    scales = -np.sum(cross_t * cross_r, axis=1) / np.maximum(denominator, 1e-12)
    scales = scales[valid & (denominator > 1e-8) & (scales > 0) & (scales < 20)]
    if len(scales) < 3:
        return result | {"reason": ";".join(reasons + ["unobservable_or_invalid_floor_scale"])}
    scale = float(np.median(scales))
    floor_height_b = 1 + float((rotation @ ga) @ translation.ravel()) * scale
    if floor_height_b <= 0.1:
        reasons.append("invalid_second_camera_floor_height")
    h = (
        np.asarray(lens_b["new_K"])
        @ (rotation + np.outer(translation.ravel() * scale, ga))
        @ np.linalg.inv(np.asarray(lens_a["new_K"]))
    )
    normalized, pixels = floor_error(h, feet, lens_a, lens_b)
    inliers = normalized <= 0.02
    if inliers.sum() < 3 or np.mean(inliers) < 0.6:
        reasons.append("floor_scale_does_not_fit_training")
    result.update(
        ideal_a_to_b=h.tolist(),
        baseline_in_first_camera_height_units=scale,
        second_camera_relative_height=floor_height_b,
        train_inliers=int(inliers.sum()),
        train_inlier_fraction=float(inliers.mean()),
        supporting_trajectory_segments=len(ids),
        train_median_symmetric_px=float(np.median(pixels)),
        train_p95_symmetric_px=float(np.quantile(pixels, 0.95)),
        training_inlier_keys=[[r["frame_ids"], r["track_ids"]] for r, g in zip(feet, inliers) if g],
    )
    result["train_supported"] = not reasons
    ah, bh, _ = joint_correspondences(holdout, lens_a, lens_b)
    if len(ah) < 16:
        reasons.append("insufficient_body_holdout")
    else:
        error = epipolar_errors(e, ah, bh)
        result["body_holdout"] = {
            "points": len(ah),
            "median_normalized_error": float(np.median(error)),
            "p95_normalized_error": float(np.quantile(error, 0.95)),
        }
        if np.median(error) > 0.006 or np.quantile(error, 0.95) > 0.02:
            reasons.append("body_pose_does_not_generalize")
    normalized, pixels = floor_error(h, validation_feet, lens_a, lens_b)
    if len(pixels) < 3:
        reasons.append("insufficient_floor_holdout")
    else:
        result["holdout"] = {
            "pairs": len(pixels),
            "median_symmetric_px": float(np.median(pixels)),
            "p95_symmetric_px": float(np.quantile(pixels, 0.95)),
        }
        if np.median(normalized) > 0.015 or np.quantile(normalized, 0.95) > 0.04:
            reasons.append("floor_plane_does_not_generalize")
    result["accepted"] = not reasons
    result["reason"] = (
        "relative_motion_geometry_supported_absolute_place_unverified" if not reasons else ";".join(reasons)
    )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--observations", type=Path, required=True)
    p.add_argument("--lenses", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setRNGSeed(42)
    observations = [json.loads(line) for line in args.observations.read_text().splitlines() if line]
    lenses = json.loads(args.lenses.read_text())["cameras"]
    pairs, diagnostic = pair_observations(observations, appearance="dual", require_feet=False, lenses=lenses)
    report = {
        "manual_plan_landmarks_used": False,
        "global_track_IDs_used": False,
        "appearance_and_time_only_pairing": True,
        "method": "body_epipolar_motion_with_sparse_floor_contacts",
        "diagnostic": diagnostic,
        "pairs": pairs,
        "edges": {},
    }
    for a, b in combinations(sorted({r["camera"] for r in observations}), 2):
        result = fit_body_edge([r for r in pairs if r["cameras"] == [a, b]], lenses[a], lenses[b]) | {
            "cameras": [a, b]
        }
        report["edges"][a + "-" + b] = result
        print(
            a,
            b,
            {
                k: v
                for k, v in result.items()
                if k
                not in [
                    "lenses",
                    "essential_matrix",
                    "relative_rotation",
                    "relative_translation_direction",
                    "ideal_a_to_b",
                    "training_inlier_keys",
                ]
            },
            flush=True,
        )
    (args.output / "pairs.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
