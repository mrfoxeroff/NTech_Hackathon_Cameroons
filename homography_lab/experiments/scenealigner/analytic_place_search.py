"""Floor-plane place hypotheses with visible-space constraints; no manual landmarks."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
from floor_ray_geometry import project_floor, ray_blocking_fraction, transform_points
from lens_geometry import project_calibration
from structural_matching import rescore_and_propose
from temporal_consensus import floor_probes, rms_distance


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def proposal_seeds(features, geometry, inference, registration, probes):
    density, good = project_floor(probes, features["lens"], geometry["floor_homography"])
    density = transform_points(density, geometry["normalization"])
    cases = {case["id"]: case for case in read(inference / "protocol.json")["cases"]}
    proposals = []
    for case_id in features["training_ids"]:
        case = cases[case_id]
        if not case.get("prediction") or not registration[case_id]["accepted"]:
            continue
        prediction = read(case["prediction"])
        query = (
            np.column_stack((probes, np.ones(len(probes)))) @ np.asarray(registration[case_id]["matrix"]).T
        )
        candidates = read(inference / case_id / "search/search.json")["candidates"]
        best = np.asarray(candidates[0]["density_to_original"])
        for rank, candidate in enumerate(candidates, 1):
            current = deepcopy(prediction)
            current["homography"] = (
                np.asarray(candidate["density_to_original"])
                @ np.linalg.inv(best)
                @ np.asarray(prediction["homography"])
            ).tolist()
            mapped, valid = project_calibration(query, current)
            valid &= good
            if valid.sum() < 8:
                continue
            matrix, mask = cv2.estimateAffinePartial2D(
                density[valid],
                mapped[valid],
                method=cv2.RANSAC,
                ransacReprojThreshold=8.0,
                maxIters=3000,
                confidence=0.999,
            )
            if matrix is None or mask.sum() < 8:
                continue
            matrix = np.vstack((matrix, [0, 0, 1.0]))
            residual = np.median(
                np.linalg.norm(transform_points(density[valid], matrix) - mapped[valid], axis=1)
            )
            proposals.append(
                {
                    "density_to_original": matrix.tolist(),
                    "score": candidate.get("neural_score", 0.0),
                    "neural_score": candidate.get("neural_score", 0.0),
                    "method": "neural_proposal_projected_into_analytic_floor",
                    "source_case": case_id,
                    "source_rank": rank,
                    "proposal_similarity_residual_px": float(residual),
                }
            )
    return proposals


def select_place(candidates, signatures, ray_count):
    """Conservative output gate: a high score is not a physical accuracy certificate."""
    if not candidates:
        return {"status": "no_location", "reasons": ["no_feasible_hypothesis"], "selected_rank": None}
    first = candidates[0]
    metrics = first["structural_metrics"]
    # Nearby numerical variants are not distinct alternatives; compare projected floor probes.
    second = next(
        (i for i in range(1, len(candidates)) if rms_distance(signatures[0], signatures[i]) > 18.0), None
    )
    gap = (
        (first["score"] - candidates[second]["score"]) / max(first["score"], 1e-12)
        if second is not None
        else None
    )
    reasons = []
    if ray_count < 4:
        reasons.append("insufficient_visibility_directions")
    if metrics.get("blocked_visibility_fraction") is None or metrics["blocked_visibility_fraction"] > 0.25:
        reasons.append("visibility_conflict_or_not_tested")
    if metrics["floor_wall_fraction"] > 0.1:
        reasons.append("floor_intersects_walls")
    if metrics["wall_trimmed_distance_px"] > 5.0:
        reasons.append("weak_boundary_alignment")
    if metrics.get("door_width_absolute_log_ratio") is not None and metrics[
        "door_width_absolute_log_ratio"
    ] > np.log(2.0):
        reasons.append("door_scale_conflict")
    if gap is None or gap < 0.2:
        reasons.append("insufficient_margin_between_distinct_places")
    return {
        "status": "candidate_requires_validation" if not reasons else "abstained",
        "reasons": reasons,
        "selected_rank": 1 if not reasons else None,
        "best_hypothesis_rank": 1,
        "distinct_runner_up_rank": None if second is None else second + 1,
        "relative_score_gap": gap,
        "accuracy_certified": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--inference", type=Path, required=True)
    p.add_argument("--registration", type=Path, required=True)
    p.add_argument("--plan-walls", type=Path, required=True)
    p.add_argument(
        "--doors", type=Path, help="Automatically detected door anchors, never reference calibration JSON"
    )
    p.add_argument("--plan-semantics", type=Path)
    p.add_argument(
        "--door-size-prior",
        action="store_true",
        help="Penalize collapse using automatically detected door width",
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(2)
    cv2.setRNGSeed(42)
    features = read(args.features / "features.json")
    geometry = dict(np.load(args.features / "geometry.npz"))
    walls = cv2.imread(str(args.plan_walls), 0)
    probes = floor_probes(cv2.imread(str(args.features / "floor-aggregate.png"), 0), count=48)
    seeds = proposal_seeds(features, geometry, args.inference, read(args.registration)["cases"], probes)
    floors = geometry["floor_density"]
    floors = floors[np.linspace(0, len(floors) - 1, min(180, len(floors))).astype(int)]
    visibility = {"origin": geometry["camera_density"], "ends": geometry["ray_endpoints_density"]}
    density, valid = project_floor(probes, features["lens"], geometry["floor_homography"])
    density = transform_points(density[valid], geometry["normalization"])
    report = {
        "method": "analytic_floor_boundary_and_visible_space",
        "manual_landmarks_used": False,
        "reference_JSONs_opened": False,
        "training_ids": features["training_ids"],
        "validation_used_for_selection": False,
        "proposal_count": len(seeds),
        "visibility_rays": len(visibility["ends"]),
        "modes": {},
        "thresholds": {
            "visibility_weight": 12,
            "maximum_blocked_fraction": 0.25,
            "minimum_relative_gap": 0.2,
            "distinct_place_projection_rms_px": 18.0,
            "thresholds_chosen_before_reference_evaluation": True,
        },
    }
    modes = (
        ["boundary_only", "boundary_and_visibility"] if not args.doors else ["doors_boundary_and_visibility"]
    )
    if args.doors and not args.plan_semantics:
        raise ValueError("Plan door proposals are required with image door anchors")
    door_points = (
        np.asarray(read(args.doors)["density_anchors"]).reshape(-1, 2) if args.doors else np.empty((0, 2))
    )
    plan_doors = read(args.plan_semantics)["plan_doors"] if args.doors else []
    report["detected_floor_door_anchors"] = door_points.tolist()
    report["door_weight"] = 1.0 if args.doors else 0.3
    widths = read(args.doors).get("stable_density_widths") if args.doors and args.door_size_prior else None
    if args.door_size_prior and (widths is None or len(widths) != len(door_points)):
        raise ValueError("Door size prior requires a width for each persistent anchor")
    report["door_size_prior"] = {
        "used": args.door_size_prior,
        "absolute_log_ratio_weight": 6.0,
        "density_widths": widths,
    }
    for mode in modes:
        ranked = rescore_and_propose(
            geometry["image"],
            door_points,
            floors,
            walls,
            plan_doors,
            seeds,
            visibility=visibility if mode.endswith("visibility") else None,
            door_weight=1.0 if args.doors else 0.3,
            explore_yaw=bool(args.doors),
            door_widths=widths,
        )
        # Keep genuinely different placements, not numerous local variants of one room.
        distinct, signatures = [], []
        for candidate in ranked:
            signature = transform_points(density, np.asarray(candidate["density_to_original"]))
            if any(rms_distance(signature, other) < 8.0 for other in signatures):
                continue
            candidate["visibility_diagnostic"] = ray_blocking_fraction(
                np.asarray(candidate["density_to_original"]), visibility["origin"], visibility["ends"], walls
            )
            calibration = {
                "camera": "cam_2",
                "camera_size": features["lens"]["size"],
                "plan_size": [walls.shape[1], walls.shape[0]],
                "homography": (
                    np.asarray(candidate["density_to_original"]) @ geometry["image_to_density"]
                ).tolist(),
                "lens_correction_applied": True,
                "homography_input_space": "ideal_pixels",
                "lens": features["lens"],
                "floor_ray_denominator": geometry["floor_homography"][2].tolist(),
                "minimum_floor_ray_denominator": 0.03,
                "manual_landmarks_used": False,
                "status": "unverified_hypothesis",
            }
            candidate["calibration"] = calibration
            distinct.append(candidate)
            signatures.append(signature)
            if len(distinct) >= 20:
                break
        decision = select_place(distinct, signatures, len(visibility["ends"]))
        report["modes"][mode] = {"decision": decision, "candidates": distinct}
        print(mode, "candidates", len(distinct), "decision", decision, flush=True)
        folder = args.output / mode
        folder.mkdir()
        if decision["selected_rank"]:
            (folder / "cam_2-calibration.json").write_text(
                json.dumps(distinct[0]["calibration"], indent=2) + "\n", encoding="utf-8"
            )
        if distinct:
            (folder / "best-hypothesis.json").write_text(
                json.dumps(distinct[0]["calibration"], indent=2) + "\n", encoding="utf-8"
            )
    (args.output / "search.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
