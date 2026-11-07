"""Select a calibration medoid by agreement between frames, without reference landmarks."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
from lens_geometry import project_calibration


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def floor_probes(mask, count=16):
    components, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 128).astype(np.uint8))
    if components <= 1:
        raise ValueError("No floor component")
    component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    distance = cv2.distanceTransform((labels == component).astype(np.uint8), cv2.DIST_L2, 5)
    yy, xx = np.mgrid[: mask.shape[0] : 32, : mask.shape[1] : 32]
    use = distance[yy, xx] > 12
    points = np.column_stack((xx[use], yy[use])).astype(float)
    if len(points) < count:
        raise ValueError("Too few spatially spread floor probes")
    chosen = [int(np.argmax(distance[points[:, 1].astype(int), points[:, 0].astype(int)]))]
    for _ in range(count - 1):
        distances = np.min(np.linalg.norm(points[:, None] - points[chosen][None], axis=-1), axis=1)
        chosen.append(int(np.argmax(distances)))
    return points[chosen]


def rms_distance(a, b):
    return np.sqrt(np.mean(np.sum((a - b) ** 2, axis=-1)))


def cluster_candidates(candidates, threshold, total_frames):
    if not candidates:
        return []
    signatures = np.asarray([c["signature"] for c in candidates], float)
    distances = np.sqrt(np.mean(np.sum((signatures[:, None] - signatures[None]) ** 2, axis=-1), axis=-1))
    frames = sorted({candidate["case_id"] for candidate in candidates})
    by_frame = {
        frame: [i for i, candidate in enumerate(candidates) if candidate["case_id"] == frame]
        for frame in frames
    }
    clusters = []
    for seed in range(len(candidates)):
        medoid = seed
        for _ in range(3):
            members = []
            for frame in frames:
                indexes = by_frame[frame]
                nearest = min(indexes, key=lambda index: distances[medoid, index])
                if distances[medoid, nearest] <= threshold:
                    members.append(nearest)
            if not members:
                break
            medoid = min(members, key=lambda index: np.median(distances[index, members]))
        members = []
        for frame in frames:
            nearest = min(by_frame[frame], key=lambda index: distances[medoid, index])
            if distances[medoid, nearest] <= threshold:
                members.append(nearest)
        if not members:
            continue
        cluster = {
            "support_frames": len(members),
            "total_frames": total_frames,
            "support_case_ids": [candidates[index]["case_id"] for index in members],
            "median_rms_px": float(np.median(distances[medoid, members])),
            "max_rms_px": float(np.max(distances[medoid, members])),
            "mean_candidate_score": float(np.mean([candidates[index]["score"] for index in members])),
            "medoid_index": medoid,
        }
        clusters.append(cluster)
    clusters.sort(
        key=lambda cluster: (
            -cluster["support_frames"],
            cluster["median_rms_px"],
            -cluster["mean_candidate_score"],
        )
    )
    distinct = []
    for cluster in clusters:
        if any(distances[cluster["medoid_index"], other["medoid_index"]] <= threshold for other in distinct):
            continue
        distinct.append(cluster)
        if len(distinct) >= 10:
            break
    return distinct


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inference", type=Path, required=True)
    p.add_argument("--registration", type=Path, required=True)
    p.add_argument("--floor-mask", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = read(args.inference / "protocol.json")
    if protocol["status"] != "complete":
        raise ValueError("Inference is incomplete")
    registration = read(args.registration)["cases"]
    probes = floor_probes(cv2.imread(str(args.floor_mask), 0))
    cases = sorted(
        [case for case in protocol["cases"] if case["kind"] in ["uniform", "foreground_stress"]],
        key=lambda case: case["actual_seconds"],
    )
    candidates = []
    plan_size = None
    for case in cases:
        reg = registration.get(case["id"], {})
        if not case.get("prediction") or not reg.get("accepted"):
            continue
        prediction = read(case["prediction"])
        plan_size = prediction["plan_size"]
        matrix = np.asarray(reg["matrix"])
        query = np.column_stack((probes, np.ones(len(probes)))) @ matrix.T
        search = read(args.inference / case["id"] / "search/search.json")
        best = np.asarray(search["candidates"][0]["density_to_original"])
        for rank, candidate in enumerate(search["candidates"], 1):
            calibration = deepcopy(prediction)
            calibration["homography"] = (
                np.asarray(candidate["density_to_original"])
                @ np.linalg.inv(best)
                @ np.asarray(prediction["homography"])
            ).tolist()
            projected, valid = project_calibration(query, calibration)
            if not valid.all() or np.max(abs(projected)) > 2 * np.hypot(*plan_size):
                continue
            candidates.append(
                {
                    "case_id": case["id"],
                    "rank": rank,
                    "score": candidate["score"],
                    "signature": projected.tolist(),
                    "calibration": calibration,
                    "input_affine": matrix.tolist(),
                }
            )
    if not candidates:
        raise ValueError("No usable candidates")
    case_ids = [case["id"] for case in cases]
    subsets = {"all_frames": case_ids, "alternating_A": case_ids[::2], "alternating_B": case_ids[1::2]}
    result = {
        "method": "one_vote_per_frame_floor_projection_medoid",
        "manual_landmarks_used": False,
        "reference_JSONs_opened": False,
        "probes": probes.tolist(),
        "candidates": len(candidates),
        "subsets": subsets,
        "default_threshold_fraction": 0.02,
        "trials": [],
    }
    for subset, selected in subsets.items():
        current = [candidate for candidate in candidates if candidate["case_id"] in selected]
        for fraction in [0.01, 0.02, 0.04]:
            threshold = fraction * np.hypot(*plan_size)
            clusters = cluster_candidates(current, threshold, len(selected))
            trial = {
                "subset": subset,
                "threshold_fraction": fraction,
                "threshold_px": float(threshold),
                "input_frames": selected,
                "clusters": [],
            }
            for cluster in clusters:
                medoid = current[cluster["medoid_index"]]
                item = {key: value for key, value in cluster.items() if key != "medoid_index"}
                item.update(
                    calibration=medoid["calibration"],
                    input_affine=medoid["input_affine"],
                    medoid_case=medoid["case_id"],
                    medoid_rank=medoid["rank"],
                    signature=medoid["signature"],
                )
                trial["clusters"].append(item)
            if clusters:
                first = clusters[0]["support_frames"]
                second = clusters[1]["support_frames"] if len(clusters) > 1 else 0
                coherent = first >= np.ceil(0.6 * len(selected)) and first - second >= max(
                    2, 0.2 * len(selected)
                )
                trial["status"] = (
                    "coherent_candidate_not_physically_verified" if coherent else "ambiguous_consensus"
                )
                trial["support_gap_frames"] = first - second
                print(subset, fraction, "support", first, "runner-up", second, trial["status"], flush=True)
            else:
                trial["status"] = "no_consensus"
            result["trials"].append(trial)
    # Calibration remains in the medoid frame's pixel coordinates. The input affine
    # must be applied before lens correction; do not average homography coefficients.
    result["projection_recipe"] = (
        "common source pixels -> input_affine -> lens undistortion -> selected homography"
    )
    result["automatic_deployment_ready"] = False
    (args.output / "consensus.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
