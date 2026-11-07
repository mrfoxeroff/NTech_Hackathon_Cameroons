"""Assemble diverse automatic map proposals; manual/reference calibrations are not inputs."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
from floor_ray_geometry import floor_homography
from lens_geometry import project_calibration, undistort_points
from motion_config import OFFICE
from temporal_consensus import floor_probes

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_sources(frames, preparation):
    for camera, item in frames["cameras"].items():
        recorded = preparation["cameras"].get(camera, {})
        if item["source_sha256"] != recorded.get("video_sha256"):
            raise ValueError("Automatic map candidates belong to another source video: " + camera)


def validate_bank(report):
    if report.get("manual_plan_landmarks_used") is not False:
        raise ValueError("Candidate bank must explicitly declare automatic, non-manual provenance")
    for camera, data in report.get("cameras", {}).items():
        probes = np.asarray(data["probes"], float)
        if probes.ndim != 2 or probes.shape[1] != 2 or not np.isfinite(probes).all():
            raise ValueError("Invalid probes for " + camera)
        for candidate in data["candidates"]:
            calibration = candidate["calibration"]
            h = np.asarray(calibration["homography"], float)
            if h.shape != (3, 3) or not np.isfinite(h).all() or np.linalg.matrix_rank(h) != 3:
                raise ValueError("Invalid automatic homography for " + camera)
            if calibration.get("manual_landmarks_used") is not False:
                raise ValueError("Manual calibrations are not inference candidates")
            if not np.isfinite(candidate["unary_cost"]):
                raise ValueError("Invalid candidate cost")
    if not report.get("cameras"):
        raise ValueError("Empty candidate bank")
    return report


def propagate_candidates(report, pair_report, lenses):
    originals = deepcopy(report["cameras"])
    counts = {camera: 0 for camera in originals}
    for edge in pair_report["edges"].values():
        if not edge.get("train_supported") or not edge.get("ideal_a_to_b"):
            continue
        a, b = edge["cameras"]
        relative = np.asarray(edge["ideal_a_to_b"])
        for source, target, h_source_to_target in [(a, b, relative), (b, a, np.linalg.inv(relative))]:
            lens_a, lens_b = lenses[source], lenses[target]
            probes = np.asarray(originals[source]["probes"])
            ideal, valid = undistort_points(probes, lens_a)
            for candidate in originals[source]["candidates"]:
                calibration = candidate["calibration"]
                lens = calibration.get("lens")
                if lens and np.allclose(lens["K"], lens_a["K"]) and np.isclose(lens["k1"], lens_a["k1"]):
                    h = (
                        np.asarray(calibration["homography"])
                        @ np.asarray(lens["new_K"])
                        @ np.linalg.inv(np.asarray(lens_a["new_K"]))
                    )
                else:
                    mapped, good = project_calibration(probes, calibration)
                    good &= valid
                    if good.sum() < 8:
                        continue
                    h, _ = cv2.findHomography(ideal[good], mapped[good], 0)
                    if h is None:
                        continue
                transferred = deepcopy(calibration)
                transferred.update(
                    camera_id=target,
                    camera_size=lens_b["size"],
                    homography=(h @ np.linalg.inv(h_source_to_target)).tolist(),
                    lens=lens_b,
                    lens_correction_applied=True,
                    homography_input_space="ideal_pixels",
                    floor_ray_denominator=floor_homography(lens_b["new_K"], lens_b["gravity"])[2].tolist(),
                    status="motion_transferred_unverified_absolute_location",
                )
                transferred["method"] = "motion_transfer_from_automatic_map_hypothesis"
                transferred["quality"] = {
                    "training_only_motion_transfer": True,
                    "independent_physical_accuracy_verified": False,
                }
                signature, good = project_calibration(np.asarray(originals[target]["probes"]), transferred)
                if good.sum() < 0.5 * len(signature) or np.nanmax(abs(signature)) > 1800:
                    continue
                report["cameras"][target]["candidates"].append(
                    {
                        "id": target + ":motion-from:" + candidate["id"],
                        "calibration": transferred,
                        "unary_cost": candidate["unary_cost"] + 0.05,
                        "source": candidate["source"],
                        "rank": candidate["rank"],
                        "source_score": candidate["source_score"],
                        "signature": np.where(good[:, None], signature, -10000.0).tolist(),
                        "motion_source_camera": source,
                        "relative_edge_holdout_accepted": edge["accepted"],
                    }
                )
                counts[target] += 1
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--maximum", type=int, default=30)
    p.add_argument("--motion-pairs", type=Path)
    p.add_argument("--lenses", type=Path)
    p.add_argument(
        "--frames", type=Path, help="Source-bound frame manifest; auto-discovered next to the output"
    )
    p.add_argument("--source-manifest", type=Path, default=OFFICE / "source-manifest.json")
    p.add_argument("--bank", type=Path, default=OFFICE / "automatic-candidates.json")
    p.add_argument(
        "--build-from-archives", action="store_true", help="Optional legacy SceneAligner cache adapter"
    )
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.build_from_archives:
        report = deepcopy(validate_bank(read(args.bank)))
        frames = args.frames or args.output.parent / "frames/frames.json"
        if frames.exists():
            verify_sources(read(frames), read(args.source_manifest))
            report["motion_source_videos_match_prepared_scene"] = True
        if args.motion_pairs:
            if not args.lenses:
                raise ValueError("Lens transforms are required for motion propagation")
            report["motion_propagation"] = propagate_candidates(
                report, read(args.motion_pairs), read(args.lenses)["cameras"]
            )
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return
    roots = [
        HERE / "runs" / name
        for name in [
            "office-divisional-groups/search-lens-structure-v2",
            "office-grouped-fragments/search-structure-only-v2",
            "office-grouped-fragments/fragment-search-topk",
        ]
    ]
    report = {
        "manual_plan_landmarks_used": False,
        "reference_JSONs_opened": False,
        "cameras": {},
        "proposal_sources": [str(p) for p in roots],
    }
    frames = args.frames or args.output.parent / "frames/frames.json"
    if frames.exists():
        verify_sources(read(frames), read(args.source_manifest))
        report["motion_source_videos_match_prepared_scene"] = True
    for camera in [f"cam_{i}" for i in range(1, 6)]:
        probes = floor_probes(cv2.imread(str(HERE / f"inputs/office-five/floor/{camera}.png"), 0))
        choices = []
        for source_index, root in enumerate(roots):
            summary = read(root / "summary.json")
            for group, item in summary["groups"].items():
                if camera not in item.get("cameras", {}):
                    continue
                prediction_path = root / f"predictions/{camera}-calibration.json"
                if not prediction_path.exists():
                    continue
                prediction = read(prediction_path)
                if prediction.get("manual_landmarks_used", True):
                    raise ValueError("Only explicitly automatic predictions are eligible")
                candidates = read(root / group / "search.json")["candidates"]
                best = np.asarray(candidates[0]["density_to_original"])
                for rank, candidate in enumerate(candidates, 1):
                    calibration = deepcopy(prediction)
                    calibration["homography"] = (
                        np.asarray(candidate["density_to_original"])
                        @ np.linalg.inv(best)
                        @ np.asarray(prediction["homography"])
                    ).tolist()
                    projected, valid = project_calibration(probes, calibration)
                    if valid.sum() < 0.5 * len(probes) or np.nanmax(abs(projected)) > 1800:
                        continue
                    # Missing rays are represented explicitly, not treated as matching locations.
                    signature = np.where(valid[:, None], projected, -10000.0)
                    choices.append(
                        {
                            "id": f"{camera}:source{source_index}:rank{rank}",
                            "calibration": calibration,
                            "unary_cost": float(0.2 * np.log(rank)),
                            "source": str(root),
                            "rank": rank,
                            "source_score": candidate["score"],
                            "signature": signature.tolist(),
                        }
                    )
        choices.sort(key=lambda v: (v["unary_cost"], v["id"]))
        distinct = []
        for choice in choices:
            signature = np.array(choice["signature"])
            if any(
                np.sqrt(np.mean(np.sum((signature - np.array(v["signature"])) ** 2, axis=1))) < 8
                for v in distinct
            ):
                continue
            distinct.append(choice)
            if len(distinct) >= args.maximum:
                break
        report["cameras"][camera] = {
            "probes": probes.tolist(),
            "candidates": distinct,
            "status": "unverified_automatic_hypotheses",
        }
        print(camera, len(distinct), flush=True)
    if args.motion_pairs:
        if not args.lenses:
            raise ValueError("Lens transforms are required for motion propagation")
        report["motion_propagation"] = propagate_candidates(
            report, read(args.motion_pairs), read(args.lenses)["cameras"]
        )
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
