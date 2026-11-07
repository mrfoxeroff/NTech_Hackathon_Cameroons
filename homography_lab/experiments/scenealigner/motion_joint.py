"""Joint map-hypothesis selection and bounded refinement from training foot trajectories."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
from floor_ray_geometry import transform_points
from lens_geometry import project_calibration
from motion_pairs import balanced_matches
from scipy.optimize import least_squares


def pair_projection_errors(rows, calibrations):
    if not rows:
        return np.empty(0), 0
    a, b = rows[0]["cameras"]
    x, vx = project_calibration(np.array([v["points"][0] for v in rows]), calibrations[a])
    y, vy = project_calibration(np.array([v["points"][1] for v in rows]), calibrations[b])
    errors = np.linalg.norm(x - y, axis=1)
    errors[~(vx & vy)] = np.inf
    return errors, int((~(vx & vy)).sum())


def projected_point_covariance(points, calibration):
    points = np.asarray(points, float)
    center, good = project_calibration(points, calibration)
    dx, vx = project_calibration(points + [1.0, 0.0], calibration)
    dy, vy = project_calibration(points + [0.0, 1.0], calibration)
    jacobian = np.stack((dx - center, dy - center), axis=-1)
    covariance = jacobian @ np.swapaxes(jacobian, 1, 2)
    return center, covariance, good & vx & vy


def scale_invariant_errors(rows, calibrations):
    if not rows:
        return np.empty(0)
    a, b = rows[0]["cameras"]
    x, cx, vx = projected_point_covariance([r["points"][0] for r in rows], calibrations[a])
    y, cy, vy = projected_point_covariance([r["points"][1] for r in rows], calibrations[b])
    result = np.full(len(rows), np.inf)
    diagonal = np.mean([np.hypot(*calibrations[c]["camera_size"]) for c in [a, b]])
    for i in np.flatnonzero(vx & vy):
        covariance = cx[i] + cy[i]
        if not np.isfinite(covariance).all() or np.linalg.det(covariance) <= 1e-18:
            continue
        delta = x[i] - y[i]
        result[i] = np.sqrt(max(0.0, delta @ np.linalg.solve(covariance, delta))) / diagonal
    return result


def error_summary(rows, calibrations):
    errors, invalid = pair_projection_errors(rows, calibrations)
    finite = errors[np.isfinite(errors)]
    normalized = scale_invariant_errors(rows, calibrations)
    return {
        "pairs": len(errors),
        "unprojectable_pairs": invalid,
        "median_px": float(np.median(finite)) if len(finite) else None,
        "p95_px": float(np.quantile(finite, 0.95)) if len(finite) else None,
        "median_scale_invariant_diagonal_fraction": float(np.median(normalized))
        if len(normalized) and np.isfinite(normalized).all()
        else None,
        "p95_scale_invariant_diagonal_fraction": float(np.quantile(normalized, 0.95))
        if len(normalized) and np.isfinite(normalized).all()
        else None,
    }


def train_rows(pair_report, edge):
    keys = {json.dumps(key, sort_keys=True) for key in edge.get("training_inlier_keys", [])}
    return [
        v
        for v in balanced_matches(
            [v for v in pair_report["pairs"] if v["cameras"] == edge["cameras"] and v["split"] == "train"]
        )
        if json.dumps([v["frame_ids"], v["track_ids"]], sort_keys=True) in keys
    ]


def connected_components(cameras, edges):
    links = {c: set() for c in cameras}
    for edge in edges:
        a, b = edge["cameras"]
        links[a].add(b)
        links[b].add(a)
    result = []
    remaining = set(cameras)
    while remaining:
        pending = [min(remaining)]
        component = set()
        while pending:
            current = pending.pop()
            if current not in component:
                component.add(current)
                pending.extend(links[current] - component)
        result.append(sorted(component))
        remaining -= component
    return result


def motion_map_cost(rows, camera, calibration, wall_mask):
    selected = [r for r in rows if camera in r["cameras"]]
    if not selected or wall_mask is None:
        return 0.0
    points = np.array([r["points"][r["cameras"].index(camera)] for r in selected])
    mapped, valid = project_calibration(points, calibration)
    height, width = wall_mask.shape
    inside = valid & np.isfinite(mapped).all(1)
    inside &= (mapped >= 0).all(1) & (mapped < [width, height]).all(1)
    collision = np.ones(len(points), bool)
    for i in np.flatnonzero(inside):
        x, y = np.rint(mapped[i]).astype(int)
        x, y = np.clip([x, y], [0, 0], [width - 1, height - 1])
        collision[i] = wall_mask[y, x] > 0
    groups = {}
    for i, row in enumerate(selected):
        groups.setdefault(row["track_ids"][row["cameras"].index(camera)], []).append(i)
    crossed = []
    for indexes in groups.values():
        indexes.sort(key=lambda i: selected[i]["timestamp_sec"])
        for i, j in zip(indexes, indexes[1:]):
            if (
                not (inside[i] and inside[j])
                or selected[j]["timestamp_sec"] - selected[i]["timestamp_sec"] > 1.5
            ):
                continue
            length = np.linalg.norm(mapped[j] - mapped[i])
            if length < 3:
                continue
            samples = mapped[i] + np.linspace(0.0, 1.0, max(2, int(np.ceil(length / 0.75)) + 1))[:, None] * (
                mapped[j] - mapped[i]
            )
            xy = np.rint(samples).astype(int)
            xy = np.clip(xy, [0, 0], [width - 1, height - 1])
            crossed.append(bool(np.any(wall_mask[xy[:, 1], xy[:, 0]] > 0)))
    return float(3 * np.mean(collision) + (3 * np.mean(crossed) if crossed else 0.0))


def search_component(cameras, bank, edges_with_rows, beam_width=2048, wall_mask=None):
    degree = {c: sum(c in e["cameras"] for e, _ in edges_with_rows) for c in cameras}
    order = sorted(cameras, key=lambda c: (-degree[c], c))
    costs = {}
    all_rows = [r for _, rows in edges_with_rows for r in rows]
    solid = cv2.erode(wall_mask, np.ones((3, 3), np.uint8)) if wall_mask is not None else None
    route_cost = {
        c: [motion_map_cost(all_rows, c, item["calibration"], solid) for item in bank[c]["candidates"]]
        for c in cameras
    }
    for edge, rows in edges_with_rows:
        a, b = edge["cameras"]
        table = np.full((len(bank[a]["candidates"]), len(bank[b]["candidates"])), 50.0)
        for i, ca in enumerate(bank[a]["candidates"]):
            for j, cb in enumerate(bank[b]["candidates"]):
                errors = scale_invariant_errors(rows, {a: ca["calibration"], b: cb["calibration"]})
                invalid = int((~np.isfinite(errors)).sum())
                if len(errors) and invalid <= 0.1 * len(errors):
                    groups = {}
                    for row, error in zip(rows, errors):
                        groups.setdefault(tuple(row["track_ids"]), []).append(min(error / 0.01, 20.0))
                    table[i, j] = float(np.mean([np.median(v) for v in groups.values()]))
        costs[(a, b)] = table
    beam = [(0.0, {})]
    exhaustive = True
    for camera in order:
        expanded = []
        for score, assignment in beam:
            for index, candidate in enumerate(bank[camera]["candidates"]):
                value = score + candidate["unary_cost"] + route_cost[camera][index]
                for (a, b), table in costs.items():
                    if camera == a and b in assignment:
                        value += table[index, assignment[b]]
                    elif camera == b and a in assignment:
                        value += table[assignment[a], index]
                expanded.append((float(value), assignment | {camera: index}))
        expanded.sort(key=lambda v: (v[0], tuple(sorted(v[1].items()))))
        exhaustive &= len(expanded) <= beam_width
        beam = expanded[:beam_width]
    distinct = []
    for score, assignment in beam:
        signature = np.concatenate(
            [np.asarray(bank[c]["candidates"][assignment[c]]["signature"]).ravel() for c in sorted(cameras)]
        )
        if any(np.sqrt(np.mean((signature - v[2]) ** 2) * 2) < 18.0 for v in distinct):
            continue
        distinct.append((score, assignment, signature))
        if len(distinct) >= 10:
            break
    return [(s, a) for s, a, _ in distinct], exhaustive


def similarity(parameters):
    tx, ty, angle, log_scale = parameters
    c, s = np.cos(angle) * np.exp(log_scale), np.sin(angle) * np.exp(log_scale)
    return np.array([[c, -s, tx], [s, c, ty], [0.0, 0, 1.0]])


def refine_component(calibrations, rows, wall_mask=None, floor_probes=None):
    cameras = sorted(calibrations)
    anchor = cameras[0]
    others = [c for c in cameras if c != anchor]
    if not others or not rows:
        return deepcopy(calibrations), {"performed": False, "reason": "no_training_motion_constraints"}
    prepared = []
    by_pair = {}
    for row in rows:
        by_pair.setdefault(tuple(row["cameras"]), []).append(row)
    for (a, b), group in by_pair.items():
        x, cx, vx = projected_point_covariance([v["points"][0] for v in group], calibrations[a])
        y, cy, vy = projected_point_covariance([v["points"][1] for v in group], calibrations[b])
        noise = 0.01 * np.mean([np.hypot(*calibrations[c]["camera_size"]) for c in [a, b]])
        for row, pa, pb, ca, cb, valid in zip(group, x, y, cx, cy, vx & vy):
            if valid and np.linalg.det(ca + cb) > 1e-18:
                whitening = np.linalg.inv(np.linalg.cholesky(ca + cb)) / noise
                prepared.append((a, b, pa, pb, whitening, tuple(row["track_ids"])))
    if len(prepared) < 12:
        return deepcopy(calibrations), {
            "performed": False,
            "reason": "insufficient_projectable_training_pairs",
        }
    counts = {key: sum(v[-1] == key for v in prepared) for key in {v[-1] for v in prepared}}
    map_floors = {}
    if wall_mask is not None and floor_probes is not None:
        field = cv2.distanceTransform((wall_mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
        for camera in others:
            points, valid = project_calibration(np.asarray(floor_probes[camera]), calibrations[camera])
            map_floors[camera] = points[valid]

    def matrices(parameters):
        return {anchor: np.eye(3)} | {
            c: similarity(parameters[i * 4 : i * 4 + 4]) for i, c in enumerate(others)
        }

    def residuals(parameters):
        transforms = matrices(parameters)
        result = []
        for a, b, x, y, whitening, key in prepared:
            delta = transform_points(x[None], transforms[a])[0] - transform_points(y[None], transforms[b])[0]
            result.extend((whitening @ delta) / np.sqrt(counts[key]))
        for i, camera in enumerate(others):
            local = parameters[i * 4 : i * 4 + 4]
            result.extend(local / np.array([10.0, 10.0, 0.08, 0.1]))
            if camera in map_floors:
                points = transform_points(map_floors[camera], transforms[camera])
                distance = cv2.remap(
                    field,
                    points[:, 0:1].astype(np.float32),
                    points[:, 1:2].astype(np.float32),
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=30,
                )[:, 0]
                result.extend(distance / 3 / np.sqrt(max(1, len(distance))))
        return np.asarray(result)

    initial = np.zeros(4 * len(others))
    limit = np.tile([15.0, 15.0, 0.12, 0.15], len(others))
    optimized = least_squares(
        residuals,
        initial,
        bounds=(-limit, limit),
        loss="soft_l1",
        max_nfev=120,
        x_scale=np.tile([5.0, 5.0, 0.05, 0.05], len(others)),
    )
    transforms = matrices(optimized.x)
    result = deepcopy(calibrations)
    for camera in cameras:
        result[camera]["homography"] = (
            transforms[camera] @ np.asarray(calibrations[camera]["homography"])
        ).tolist()
        result[camera]["status"] = "motion_refined_unverified_absolute_location"
    return result, {
        "performed": True,
        "anchor_camera": anchor,
        "anchor_kind": "unverified_automatic_candidate_not_manual_ground_truth",
        "success": bool(optimized.success),
        "training_pairs": len(prepared),
        "parameters": {c: optimized.x[i * 4 : i * 4 + 4].tolist() for i, c in enumerate(others)},
        "initial_cost": float(np.sum(residuals(initial) ** 2)),
        "final_cost": float(np.sum(residuals(optimized.x) ** 2)),
        "maximum_translation_px": 15.0,
        "maximum_rotation_rad": 0.12,
        "maximum_absolute_log_scale": 0.15,
        "holdout_used": False,
    }


def solve(bank_report, pair_report, wall_mask=None, beam_width=2048, use_map_motion=False):
    bank = bank_report["cameras"]
    cameras = sorted(c for c in bank if bank[c]["candidates"])
    # Holdout acceptance must not decide which hypotheses/edges enter the optimizer.
    edges = [
        e
        for e in pair_report["edges"].values()
        if e.get("train_supported", False) and all(c in cameras for c in e["cameras"])
    ]
    components = connected_components(cameras, edges)
    result = {
        "method": "motion_constrained_joint_hypothesis_search_and_bounded_similarity_refinement",
        "manual_plan_landmarks_used": False,
        "global_track_IDs_used": False,
        "holdout_used_to_choose_hypotheses": False,
        "components": [],
        "calibrations": {},
        "motion_objective": "projection_covariance_normalized_pixel_uncertainty_not_raw_map_distance",
        "map_motion_prior_enabled": use_map_motion,
        "absolute_location_verified": False,
        "automatic_deployment_ready": False,
    }
    for component in components:
        current = [
            (edge, train_rows(pair_report, edge)) for edge in edges if set(edge["cameras"]) <= set(component)
        ]
        solutions, exhaustive = search_component(
            component, bank, current, beam_width, wall_mask if use_map_motion else None
        )
        if not solutions:
            continue
        score, assignment = solutions[0]
        chosen = {c: deepcopy(bank[c]["candidates"][assignment[c]]["calibration"]) for c in component}
        train = [v for _, rows in current for v in rows]
        refined, refinement = refine_component(
            chosen, train, wall_mask, {c: bank[c]["probes"] for c in component}
        )
        reasons = []
        if len(component) < 2 or not current:
            reasons.append("no_supported_training_motion_edge")
        if len(solutions) < 2:
            reasons.append("insufficient_distinct_place_alternatives")
        gap = solutions[1][0] - score if len(solutions) > 1 else None
        if gap is not None and gap < 0.25:
            reasons.append("ambiguous_absolute_place_hypotheses")
        evaluation = {}
        for edge, training in current:
            holdout = balanced_matches(
                [
                    v
                    for v in pair_report["pairs"]
                    if v["cameras"] == edge["cameras"]
                    and v["split"] == "holdout"
                    and all(p is not None for p in v["points"])
                ]
            )
            name = "-".join(edge["cameras"])
            evaluation[name] = {
                "train_before": error_summary(training, chosen),
                "train_after": error_summary(training, refined),
                "holdout_before": error_summary(holdout, chosen),
                "holdout_after": error_summary(holdout, refined),
                "independent_relative_geometry_accepted": edge["accepted"],
            }
            check = evaluation[name]["holdout_after"]
            if not edge["accepted"]:
                reasons.append(name + ":relative_holdout_failed")
            if (
                check["pairs"] < 8
                or check["unprojectable_pairs"]
                or check["median_px"] is None
                or check["median_px"] > 10
                or check["p95_px"] > 20
            ):
                reasons.append(name + ":map_projection_holdout_failed")
            if (
                check["median_scale_invariant_diagonal_fraction"] is None
                or check["median_scale_invariant_diagonal_fraction"] > 0.015
                or check["p95_scale_invariant_diagonal_fraction"] > 0.04
            ):
                reasons.append(name + ":scale_invariant_holdout_failed")
        result["components"].append(
            {
                "cameras": component,
                "status": "abstained" if reasons else "relative_candidate_absolute_location_unverified",
                "reasons": reasons,
                "chosen_ids": {c: bank[c]["candidates"][assignment[c]]["id"] for c in component},
                "joint_cost": score,
                "runner_up_cost_gap": gap,
                "exhaustive": exhaustive,
                "refinement": refinement,
                "edge_evaluation": evaluation,
                "top_assignments": [
                    {"cost": s, "ids": {c: bank[c]["candidates"][a[c]]["id"] for c in component}}
                    for s, a in solutions
                ],
                "baseline_calibrations": chosen,
            }
        )
        result["calibrations"].update(refined)
    result["candidate_calibrations"] = deepcopy(result["calibrations"])
    result["accepted_calibrations"] = {}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--plan-walls", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--use-map-motion",
        action="store_true",
        help="Experimental trajectory/wall prior; off for inaccurate plan masks",
    )
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    bank = json.loads(args.candidates.read_text())
    pairs = json.loads(args.pairs.read_text())
    result = solve(bank, pairs, cv2.imread(str(args.plan_walls), 0), use_map_motion=args.use_map_motion)
    (args.output / "joint.json").write_text(json.dumps(result, indent=2) + "\n")
    for component in result["components"]:
        print(component["cameras"], component["status"], component["reasons"], flush=True)


if __name__ == "__main__":
    main()
