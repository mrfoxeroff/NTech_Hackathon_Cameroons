"""Post-inference evaluation of frozen calibration, temporal reruns and controls."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
from evaluate import evaluate_camera


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def transferred_reference(reference, matrix):
    result = deepcopy(reference)
    points = np.asarray(reference["camera_points"], float)
    result["camera_points"] = (
        np.column_stack((points, np.ones(len(points)))) @ np.asarray(matrix).T
    ).tolist()
    return result


def accuracy_pass(score):
    return bool(
        score
        and score["median_error_px"] is not None
        and score["p95_error_px"] is not None
        and score["median_error_px"] <= 10
        and score["p95_error_px"] <= 20
        and score.get("unprojectable_reference_inliers", 0) == 0
    )


def patch_agreement(background, frame, source, matrix):
    results = []
    mapped = np.column_stack((source, np.ones(len(source)))) @ np.asarray(matrix).T
    for point, destination in zip(source, mapped):
        if (
            (point < 21).any()
            or (point >= np.array(background.shape[::-1]) - 21).any()
            or (destination < 21).any()
            or (destination >= np.array(frame.shape[::-1]) - 21).any()
        ):
            results.append(None)
            continue
        original = cv2.getRectSubPix(background, (41, 41), tuple(point.astype(np.float32)))
        observed = cv2.getRectSubPix(frame, (41, 41), tuple(destination.astype(np.float32)))
        if original.std() < 5 or observed.std() < 5:
            results.append(None)
            continue
        results.append(float(cv2.matchTemplate(observed, original, cv2.TM_CCOEFF_NORMED)[0, 0]))
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inference", type=Path, required=True)
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--registration", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--frozen-calibration", type=Path, required=True)
    p.add_argument("--background", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("Use a new output file")
    protocol = read(args.inference / "protocol.json")
    if protocol["status"] != "complete":
        raise ValueError("Inference must finish before reading reference labels")
    samples = read(args.samples)
    registration = read(args.registration)
    reference = read(args.reference)
    frozen = read(args.frozen_calibration)
    background = cv2.imread(str(args.background), 0)
    report = {
        "evaluation_only": True,
        "reference_labels_used_for_inference": False,
        "recalibration_mode": protocol.get("mode", "independent_full_recalibration"),
        "criteria_declared_before_evaluation": {
            "max_median_px": 10,
            "max_p95_px": 20,
            "all_reference_inliers_projectable": True,
        },
        "ground_truth_limitation": "Original manual JSON landmarks propagated by independently fitted image registration; not fresh manual annotations or measured person positions.",
        "cases": {},
        "controls": {},
        "frozen_reference_score": evaluate_camera(frozen, reference),
    }
    source = np.asarray(reference["camera_points"], float)
    for item in protocol["cases"]:
        is_new = item["kind"] in ["uniform", "foreground_stress"]
        reg = registration["cases"].get(item["id"]) if is_new else None
        supported = bool(reg["accepted"]) if is_new else True
        frame_reference = (
            transferred_reference(reference, reg["matrix"])
            if reg and reg.get("matrix")
            else deepcopy(reference)
        )
        result = {
            "kind": item["kind"],
            "actual_seconds": item.get("actual_seconds"),
            "status": item.get("status"),
            "registration_supported": supported,
            "search_ambiguous": item.get("search_ambiguous"),
            "search_score_gap": item.get("search_score_gap"),
            "error": item.get("error"),
        }
        if item.get("prediction"):
            prediction = read(item["prediction"])
            result["fresh_score"] = evaluate_camera(prediction, frame_reference)
            result["fresh_pass"] = supported and accuracy_pass(result["fresh_score"])
            search = read(args.inference / item["id"] / "search/search.json")
            best = np.asarray(search["candidates"][0]["density_to_original"])
            oracle = []
            for rank, candidate in enumerate(search["candidates"], 1):
                hypothesis = dict(prediction)
                hypothesis["homography"] = (
                    np.asarray(candidate["density_to_original"])
                    @ np.linalg.inv(best)
                    @ np.asarray(prediction["homography"])
                ).tolist()
                score = evaluate_camera(hypothesis, frame_reference)
                oracle.append(
                    {
                        "rank": rank,
                        "median_error_px": score["median_error_px"],
                        "p95_error_px": score["p95_error_px"],
                        "passes_reference_criteria": accuracy_pass(score),
                    }
                )
            result["oracle_top10_diagnostic_not_automatic"] = oracle
            result["oracle_any_pass"] = supported and any(
                record["passes_reference_criteria"] for record in oracle
            )
        else:
            result.update(fresh_pass=False, oracle_any_pass=False)
        if is_new:
            if supported:
                result["frozen_score"] = evaluate_camera(frozen, frame_reference)
                result["frozen_pass"] = accuracy_pass(result["frozen_score"])
                original = cv2.imread(item.get("image", item.get("original_image")), 0)
                agreement = patch_agreement(background, original, source, reg["matrix"])
                result["control_patch_ncc"] = agreement
                result["textured_patches"] = sum(value is not None for value in agreement)
                result["patches_with_ncc_at_least_070"] = sum(
                    value is not None and value >= 0.7 for value in agreement
                )
            else:
                result.update(frozen_pass=False, frozen_score=None)
            result["registration"] = reg
            report["cases"][item["id"]] = result
        else:
            report["controls"][item["id"]] = result
    selected = list(report["cases"].values())
    assert len(selected) == len(samples["cases"])
    report["summary"] = {
        "new_frames": len(selected),
        "fresh_predictions": sum(case.get("fresh_score") is not None for case in selected),
        "fresh_pass_count": sum(case["fresh_pass"] for case in selected),
        "frozen_pass_count": sum(case["frozen_pass"] for case in selected),
        "oracle_any_pass_count_diagnostic": sum(case["oracle_any_pass"] for case in selected),
        "ambiguous_searches": sum(bool(case.get("search_ambiguous")) for case in selected),
        "accepted_camera_stability_probes": sum(
            probe["registration"]["accepted"] for probe in registration["probes"]
        ),
        "camera_stability_probes": len(registration["probes"]),
    }
    report["by_kind"] = {}
    for kind in ["uniform", "foreground_stress"]:
        members = [case for case in selected if case["kind"] == kind]
        report["by_kind"][kind] = {
            "frames": len(members),
            "fresh_pass_count": sum(case["fresh_pass"] for case in members),
            "frozen_pass_count": sum(case["frozen_pass"] for case in members),
        }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    for name, case in report["cases"].items():
        print(
            name,
            "fresh",
            case.get("fresh_score", {}).get("median_error_px"),
            "P95",
            case.get("fresh_score", {}).get("p95_error_px"),
            "pass",
            case["fresh_pass"],
            "frozen",
            (case.get("frozen_score") or {}).get("median_error_px"),
            flush=True,
        )
    print("Controls:")
    for name, case in report["controls"].items():
        print(name, case.get("fresh_score", {}).get("median_error_px"), "pass", case["fresh_pass"])


if __name__ == "__main__":
    main()
