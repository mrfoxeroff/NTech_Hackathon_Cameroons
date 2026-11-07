"""Post-selection accuracy and unseen-frame semantics; reference data is evaluation-only."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
from evaluate import evaluate_camera
from floor_ray_geometry import boundary_geometry, project_floor, ray_blocking_fraction, transform_points
from temporal_evaluate import accuracy_pass, transferred_reference


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--search", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--registration", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--plan-walls", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    search = read(args.search)
    features = read(args.features / "features.json")
    registrations = read(args.registration)["cases"]
    walls = cv2.imread(str(args.plan_walls), 0)
    reference = read(args.reference)
    report = {
        "evaluation_only": True,
        "reference_used_for_selection": False,
        "limitation": "Existing approximate manual landmarks; holdout masks and registration are not independent surveyed positions.",
        "modes": {},
        "holdout": {},
    }
    for mode, data in search["modes"].items():
        scores = []
        for rank, candidate in enumerate(data["candidates"], 1):
            score = evaluate_camera(candidate["calibration"], reference)
            scores.append(
                {
                    "rank": rank,
                    **score,
                    "passes_criteria": accuracy_pass(score),
                    "blocked_visibility_fraction": candidate["visibility_diagnostic"],
                }
            )
        report["modes"][mode] = {
            "decision": data["decision"],
            "best_hypothesis": scores[0] if scores else None,
            "all_hypotheses": scores,
            "oracle_any_pass_diagnostic_only": any(s["passes_criteria"] for s in scores),
        }
    # An oracle similarity fit ONLY diagnoses whether camera geometry can explain the labels.
    # It is deliberately kept in this evaluator and is never exported as a calibration.
    geometry = dict(np.load(args.features / "geometry.npz"))
    src, valid = project_floor(
        np.asarray(reference["camera_points"]), features["lens"], geometry["floor_homography"]
    )
    use = np.asarray(reference["inliers"], bool) & valid
    matrix, _ = cv2.estimateAffinePartial2D(
        src[use], np.asarray(reference["plan_points"])[use].astype(float), method=cv2.LMEDS
    )
    if matrix is not None:
        errors = np.linalg.norm(
            transform_points(src[use], np.vstack((matrix, [0, 0, 1.0])))
            - np.asarray(reference["plan_points"])[use],
            axis=1,
        )
        report["oracle_similarity_family_diagnostic_not_a_solution"] = {
            "median_px": float(np.median(errors)),
            "p95_px": float(np.quantile(errors, 0.95)),
            "points": int(use.sum()),
            "uses_reference": True,
            "not_used_for_ranking_or_export": True,
        }
    for case in features["cases"]:
        if case["kind"] == "training":
            continue
        registration = registrations[case["id"]]
        item = {
            "registration_accepted": registration["accepted"],
            "estimated_corner_displacement_px": registration.get("median_corner_displacement_original_px"),
        }
        if not registration["accepted"]:
            report["holdout"][case["id"]] = item
            continue
        mask_root = args.features / case["id"]
        inverse = cv2.invertAffineTransform(np.asarray(registration["matrix"]))
        size = tuple(features["lens"]["size"])
        masks = {
            key: cv2.warpAffine(
                (cv2.imread(str(mask_root / (key + ".png")), 0) > 128).astype(np.uint8),
                inverse,
                size,
                flags=cv2.INTER_NEAREST,
            )
            for key in ["floor", "boundary"]
        }
        current = boundary_geometry(
            masks["floor"], masks["boundary"], features["lens"], features["lens"]["gravity"]
        )
        item["boundary_observations"] = len(current["wall_points"])
        item["visibility_rays"] = len(current["ray_endpoints_density"])
        item["modes"] = {}
        distance = cv2.distanceTransform((walls == 0).astype(np.uint8), cv2.DIST_L2, 3)
        for mode, data in search["modes"].items():
            if not data["candidates"]:
                continue
            calibration = data["candidates"][0]["calibration"]
            # Analytical local coordinates have unit camera height in every frame.
            selected = np.asarray(data["candidates"][0]["density_to_original"]) @ geometry["normalization"]
            mapped = transform_points(current["wall_points"], selected)
            values = cv2.remap(
                distance,
                mapped[:, 0:1].astype(np.float32),
                mapped[:, 1:2].astype(np.float32),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=30,
            )[:, 0]
            trimmed = np.sort(values)[: max(1, int(0.8 * len(values)))]
            blocking = ray_blocking_fraction(
                selected @ np.linalg.inv(current["normalization"]),
                current["camera_density"],
                current["ray_endpoints_density"],
                walls,
            )
            frame_reference = transferred_reference(reference, registration["matrix"])
            # Registration is applied BEFORE the nonlinear lens transform.
            frame_points = np.asarray(frame_reference["camera_points"])
            common = np.column_stack((frame_points, np.ones(len(frame_points)))) @ inverse.T
            check_reference = deepcopy(reference)
            check_reference["camera_points"] = common.tolist()
            item["modes"][mode] = {
                "wall_trimmed_distance_px": float(trimmed.mean()),
                "blocked_visibility_fraction": blocking,
                "propagated_reference_score": evaluate_camera(calibration, check_reference),
            }
        report["holdout"][case["id"]] = item
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({mode: item["best_hypothesis"] for mode, item in report["modes"].items()}, indent=2),
        flush=True,
    )
    print(
        "Geometry family oracle (diagnostic only):",
        report.get("oracle_similarity_family_diagnostic_not_a_solution"),
    )


if __name__ == "__main__":
    main()
