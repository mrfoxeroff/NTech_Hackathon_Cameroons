"""Resolve ambiguous training identities using a common relative floor transform, never map positions."""

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from floor_ray_geometry import floor_homography, project_floor, transform_points
from motion_pairs import balanced_matches, fit_relative_edge, matching_floor_points, pair_observations
from scipy.optimize import linear_sum_assignment


def candidate_pairs(observations, lenses):
    tracks = defaultdict(list)
    for row in observations:
        tracks[row["track_id"]].append(row)
    grouped = defaultdict(list)
    for row in observations:
        if row["split"] != "train" or row.get("foot") is None:
            continue
        recent = [v for v in tracks[row["track_id"]] if abs(v["timestamp_sec"] - row["timestamp_sec"]) <= 1.0]
        x = np.mean([v["embedding"] for v in recent], axis=0)
        x /= max(np.linalg.norm(x), 1e-9)
        y = np.mean([v["embedding_intel"] for v in recent], axis=0)
        y /= max(np.linalg.norm(y), 1e-9)
        vector = np.r_[x, y] / np.sqrt(2.0)
        lens = lenses[row["camera"]]
        xy, valid = project_floor(
            np.asarray(row["foot"])[None], lens, floor_homography(lens["new_K"], lens["gravity"])
        )
        if valid[0]:
            grouped[(row["window"], row["camera"], row["timestamp_sec"])].append(
                row | {"descriptor": vector, "ground": xy[0]}
            )
    result = []
    for window in sorted({k[0] for k in grouped}):
        cameras = sorted({k[1] for k in grouped if k[0] == window})
        for ca, cb in combinations(cameras, 2):
            right = sorted(k for k in grouped if k[:2] == (window, cb))
            options = []
            for key in sorted(k for k in grouped if k[:2] == (window, ca)):
                if not right:
                    continue
                other = min(right, key=lambda k: abs(k[2] - key[2]))
                if abs(key[2] - other[2]) <= 0.18:
                    options.append((abs(key[2] - other[2]), key, other))
            used = set()
            for dt, key, other in sorted(options):
                if other in used:
                    continue
                used.add(other)
                a, b = grouped[key], grouped[other]
                similarities = (
                    np.asarray([v["descriptor"] for v in a]) @ np.asarray([v["descriptor"] for v in b]).T
                )
                for i in range(len(a)):
                    for j in np.argsort(-similarities[i])[:3]:
                        if similarities[i, j] < 0.35:
                            continue
                        x, y = a[i], b[j]
                        foot_points = matching_floor_points(x, y, lenses)
                        if foot_points is None:
                            continue
                        ground = []
                        for camera, point in zip([ca, cb], foot_points):
                            lens = lenses[camera]
                            xy, _ = project_floor(
                                np.asarray(point)[None],
                                lens,
                                floor_homography(lens["new_K"], lens["gravity"]),
                            )
                            ground.append(xy[0].tolist())
                        result.append(
                            {
                                "cameras": [ca, cb],
                                "window": window,
                                "split": "train",
                                "timestamp_sec": (key[2] + other[2]) / 2,
                                "track_ids": [x["track_id"], y["track_id"]],
                                "frame_ids": [x["frame_id"], y["frame_id"]],
                                "points": foot_points,
                                "ground": ground,
                                "appearance_cosine": float(similarities[i, j]),
                                "appearance_margin": None,
                                "foot_confidences": [x["foot_confidence"], y["foot_confidence"]],
                                "images": [x["image"], y["image"]],
                                "boxes": [x["box"], y["box"]],
                                "time_difference_sec": dt,
                            }
                        )
    return result


def assign_under_geometry(candidates, matrix):
    grouped = defaultdict(list)
    for row in candidates:
        grouped[tuple(row["frame_ids"])].append(row)
    chosen = []
    for rows in grouped.values():
        aa = sorted({r["track_ids"][0] for r in rows})
        bb = sorted({r["track_ids"][1] for r in rows})
        costs = np.full((len(aa), len(bb)), 100.0)
        lookup = {}
        for row in rows:
            i, j = aa.index(row["track_ids"][0]), bb.index(row["track_ids"][1])
            error = np.linalg.norm(
                transform_points(np.asarray(row["ground"][0])[None], matrix)[0] - row["ground"][1]
            )
            if error <= 0.12:
                costs[i, j] = error / 0.12 + 0.2 * (1 - row["appearance_cosine"])
                lookup[(i, j)] = row
        for i, j in zip(*linear_sum_assignment(costs)):
            if (i, j) in lookup:
                chosen.append(lookup[(i, j)])
    return balanced_matches(chosen)


def resolve_training(candidates):
    if len(candidates) < 12:
        return [], []
    a = np.array([r["ground"][0] for r in candidates])
    b = np.array([r["ground"][1] for r in candidates])
    hypotheses = []
    for seed in range(42, 48):
        cv2.setRNGSeed(seed)
        # Bootstrap source proposals, keeping the original data for one-to-one scoring.
        order = np.random.default_rng(seed).permutation(len(a))
        matrix, _ = cv2.estimateAffinePartial2D(
            a[order], b[order], method=cv2.RANSAC, ransacReprojThreshold=0.1, maxIters=15000, confidence=0.999
        )
        if matrix is None:
            continue
        matrix = np.vstack((matrix, [0.0, 0, 1.0]))
        selected = assign_under_geometry(candidates, matrix)
        if len(selected) < 6:
            continue
        for _ in range(2):
            x = np.array([r["ground"][0] for r in selected])
            y = np.array([r["ground"][1] for r in selected])
            refined, _ = cv2.estimateAffinePartial2D(x, y, method=cv2.LMEDS)
            if refined is None:
                break
            matrix = np.vstack((refined, [0.0, 0, 1.0]))
            selected = assign_under_geometry(candidates, matrix)
        if not selected:
            continue
        errors = np.linalg.norm(
            transform_points(np.array([r["ground"][0] for r in selected]), matrix)
            - np.array([r["ground"][1] for r in selected]),
            axis=1,
        )
        signature = transform_points(a[np.linspace(0, len(a) - 1, min(24, len(a))).astype(int)], matrix)
        if any(
            np.sqrt(np.mean(np.sum((signature - np.array(h["signature"])) ** 2, axis=1))) < 0.1
            for h in hypotheses
        ):
            continue
        hypotheses.append(
            {
                "support": len(selected),
                "median_floor_error": float(np.median(errors)),
                "matrix": matrix.tolist(),
                "signature": signature.tolist(),
                "selected": selected,
            }
        )
    hypotheses.sort(key=lambda h: (-h["support"], h["median_floor_error"]))
    return hypotheses[0]["selected"] if hypotheses else [], hypotheses


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--observations", type=Path, required=True)
    p.add_argument("--lenses", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    observations = [json.loads(line) for line in args.observations.read_text().splitlines() if line]
    lenses = json.loads(args.lenses.read_text())["cameras"]
    proposals = candidate_pairs(observations, lenses)
    independent, _ = pair_observations(observations, appearance="dual", lenses=lenses)
    # Holdout pairs remain appearance/time-only. Geometry never chooses the validation identities.
    holdout = [r for r in independent if r["split"] == "holdout"]
    report = {
        "manual_plan_landmarks_used": False,
        "global_track_IDs_used": False,
        "holdout_pairing_uses_geometry": False,
        "method": "latent_training_associations_with_independent_holdout_pairing",
        "edges": {},
        "pairs": [],
        "training_proposals": len(proposals),
    }
    for ca, cb in combinations(sorted({r["camera"] for r in observations}), 2):
        current = [r for r in proposals if r["cameras"] == [ca, cb]]
        selected, hypotheses = resolve_training(current)
        validation = [r for r in holdout if r["cameras"] == [ca, cb]]
        rows = selected + validation
        edge = fit_relative_edge(rows, lenses[ca], lenses[cb]) | {
            "cameras": [ca, cb],
            "training_identity_resolution": "relative_geometry_hypotheses_one_to_one_per_frame",
            "relative_hypotheses": [
                {
                    "support": h["support"],
                    "median_floor_error": h["median_floor_error"],
                    "matrix": h["matrix"],
                }
                for h in hypotheses
            ],
        }
        if len(hypotheses) > 1 and hypotheses[1]["support"] >= 0.9 * hypotheses[0]["support"]:
            edge.update(train_supported=False, accepted=False, reason="ambiguous_relative_identity_geometry")
        report["edges"][ca + "-" + cb] = edge
        report["pairs"].extend(rows)
        print(
            ca,
            cb,
            {
                k: v
                for k, v in edge.items()
                if k
                not in ["training_inlier_keys", "floor_similarity", "ideal_a_to_b", "relative_hypotheses"]
            },
            flush=True,
        )
    (args.output / "pairs.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
