"""Wall-distance and doorway hypothesis search, independent of reference coordinates."""

import heapq

import cv2
import numpy as np
from scipy.optimize import minimize


def dominant_angle(gray):
    lines = cv2.createLineSegmentDetector().detect(gray)[0]
    histogram = np.zeros(90)
    for line in [] if lines is None else lines[:, 0]:
        delta = line[2:] - line[:2]
        length = np.linalg.norm(delta)
        if length >= 20:
            angle = np.rad2deg(np.arctan2(delta[1], delta[0])) % 90
            histogram[int(round(angle)) % 90] += length
    return np.deg2rad(float(np.argmax(histogram)))


def floor_probes(recon, projection, images, floor_root):
    from pathlib import Path

    from density_geometry import get_density_coordinates
    from structural_features import processed_mask

    samples = []
    for i, image in enumerate(images):
        use = processed_mask(
            floor_root / (Path(image).stem + ".png"), recon["image_sizes"][i], recon["conf"][i].shape
        )
        use &= recon["conf"][i] >= np.quantile(recon["conf"][i], 0.45)
        xyz = recon["points_cam0"][i][use] @ projection["R_rect"].T
        samples.append(get_density_coordinates(xyz, projection))
    points = np.concatenate(samples)
    valid = np.isfinite(points).all(1) & (points >= 0).all(1) & (points < 512).all(1)
    points = points[valid]
    return (
        points[np.linspace(0, len(points) - 1, min(180, len(points))).astype(int)]
        if len(points)
        else np.empty((0, 2))
    )


def matrix_from_parameters(parameters):
    tx, ty, angle, log_scale = parameters
    scale = np.exp(log_scale)
    c, s = np.cos(angle) * scale, np.sin(angle) * scale
    return np.array([[c, -s, tx], [s, c, ty], [0, 0, 1.0]])


def door_scale_error(source_widths, target_widths, scale):
    source_widths = np.asarray(source_widths, float)
    target_widths = np.asarray(target_widths, float)
    known = np.isfinite(source_widths) & (source_widths > 0) & (target_widths > 0)
    if not known.any() or not np.isfinite(scale) or scale <= 0:
        return None
    return float(np.median(abs(np.log(source_widths[known] * scale / target_widths[known]))))


def rescore_and_propose(
    wall_image,
    door_points,
    floors,
    plan_walls,
    plan_doors,
    candidates,
    visibility=None,
    door_weight=0.3,
    explore_yaw=False,
    door_widths=None,
):
    height, width = plan_walls.shape
    yy, xx = np.where(wall_image < 220)
    if len(xx) < 30:
        return candidates
    selection = np.linspace(0, len(xx) - 1, min(450, len(xx))).astype(int)
    wall_points = np.column_stack((xx[selection], yy[selection])).astype(float)
    target_doors = np.asarray([d["center"] for d in plan_doors], dtype=float).reshape(-1, 2)
    distance = cv2.distanceTransform((plan_walls == 0).astype(np.uint8), cv2.DIST_L2, 3)
    floor_area = cv2.contourArea(cv2.convexHull(floors.astype(np.float32))) if len(floors) >= 3 else 0.0

    def assess(matrix):
        mapped = wall_points @ matrix[:2, :2].T + matrix[:2, 2]
        inside = ((mapped >= 0) & (mapped < [width, height])).all(1)
        if inside.mean() < 0.85 or floor_area * abs(np.linalg.det(matrix[:2, :2])) < 150:
            return 100.0, {}
        values = cv2.remap(
            distance,
            mapped[:, 0:1].astype(np.float32),
            mapped[:, 1:2].astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=30,
        )[:, 0]
        trimmed = np.sort(np.clip(values, 0, 25))[: max(1, int(0.8 * len(values)))]
        floor_fraction = 0.0
        if len(floors):
            projected = floors @ matrix[:2, :2].T + matrix[:2, 2]
            valid = ((projected >= 0) & (projected < [width, height])).all(1)
            if valid.mean() < 0.85:
                return 100.0, {}
            xy = np.rint(projected[valid]).astype(int)
            xy[:, 0] = np.clip(xy[:, 0], 0, width - 1)
            xy[:, 1] = np.clip(xy[:, 1], 0, height - 1)
            floor_fraction = float((plan_walls[xy[:, 1], xy[:, 0]] > 0).mean())
        door_error = 0.0
        width_error = None
        if len(door_points) and len(target_doors):
            projected = door_points @ matrix[:2, :2].T + matrix[:2, 2]
            door_distances = np.linalg.norm(projected[:, None] - target_doors[None], axis=-1)
            nearest = np.argmin(door_distances, axis=1)
            door_error = float(np.median(np.min(door_distances, axis=1)))
            if door_widths is not None:
                target_widths = np.array([plan_doors[index]["radius"] for index in nearest])
                scale = np.sqrt(abs(np.linalg.det(matrix[:2, :2])))
                width_error = door_scale_error(door_widths, target_widths, scale)
        cost = float(trimmed.mean() + 15 * floor_fraction + door_weight * min(door_error, 30))
        if width_error is not None:
            cost += 6 * width_error
        blocked = None
        if visibility is not None:
            from floor_ray_geometry import ray_blocking_fraction

            blocked = ray_blocking_fraction(matrix, visibility["origin"], visibility["ends"], plan_walls)
            cost += 12 * blocked
        return cost, {
            "wall_trimmed_distance_px": float(trimmed.mean()),
            "wall_median_distance_px": float(np.median(values)),
            "floor_wall_fraction": floor_fraction,
            "blocked_visibility_fraction": blocked,
            "door_width_absolute_log_ratio": width_error,
            "door_median_distance_px": door_error if len(door_points) and len(target_doors) else None,
        }

    def score_candidate(candidate):
        cost, metrics = assess(np.asarray(candidate["density_to_original"]))
        neural_score = candidate.get("neural_score", candidate["score"])
        score = float(np.exp(-cost / 6) * (0.75 + 0.25 * min(neural_score / 0.2, 1)))
        return candidate | {
            "neural_score": neural_score,
            "structural_cost": cost,
            "structural_metrics": metrics,
            "score": score,
        }

    ranked = [score_candidate(candidate) for candidate in candidates]
    source_corners = cv2.goodFeaturesToTrack(255 - wall_image, 4, 0.04, 30)
    target_corners = cv2.goodFeaturesToTrack(plan_walls, 48, 0.04, 8)
    pairs = []
    if len(door_points) and len(target_doors):
        pairs += [
            (a, b, "neural_door_to_drawing_door_proposal") for a in door_points[:6] for b in target_doors
        ]
    if source_corners is not None and target_corners is not None:
        pairs += [
            (a, b, "neural_wall_corner_to_drawing_corner_proposal")
            for a in source_corners[:, 0]
            for b in target_corners[:, 0]
        ]
    source_angle = dominant_angle(wall_image)
    target_angle = dominant_angle(255 - plan_walls)
    pool, serial = [], 0
    angles = list(target_angle - source_angle + np.arange(4) * np.pi / 2)
    if explore_yaw:
        angles += list(np.arange(24) * np.pi / 12)
    for angle in angles:
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        for scale in np.geomspace(0.08, 1.25, 16):
            linear = rotation * scale
            for source, target, kind in pairs:
                matrix = np.eye(3)
                matrix[:2, :2] = linear
                matrix[:2, 2] = target - linear @ source
                cost, metrics = assess(matrix)
                if cost >= 30:
                    continue
                item = {
                    "score": 0.0,
                    "neural_score": 0.0,
                    "density_to_original": matrix.tolist(),
                    "method": kind,
                    "fragment_box": None,
                    "inliers": None,
                    "source_points": [source.tolist()],
                    "plan_points": [target.tolist()],
                    "correspondence_type": "geometric_hypothesis_not_verified_correspondence",
                }
                serial += 1
                heapq.heappush(pool, (-cost, serial, item))
                if len(pool) > 120:
                    heapq.heappop(pool)
    from fragment_search import distinct_candidates

    geometric = distinct_candidates([score_candidate(item) for _, _, item in pool], maximum=16)
    refine = geometric[:8]
    if visibility is not None:
        refine += sorted(ranked, key=lambda item: -item["score"])[:8]
    for candidate in refine:
        matrix = np.array(candidate["density_to_original"])
        angle = np.arctan2(matrix[1, 0], matrix[0, 0])
        log_scale = np.log(np.sqrt(abs(np.linalg.det(matrix[:2, :2]))))
        initial = [matrix[0, 2], matrix[1, 2], angle, log_scale]
        bounds = [
            (initial[0] - 12, initial[0] + 12),
            (initial[1] - 12, initial[1] + 12),
            (angle - 0.16, angle + 0.16),
            (log_scale - 0.18, log_scale + 0.18),
        ]
        optimized = minimize(
            lambda p: assess(matrix_from_parameters(p))[0],
            initial,
            method="Powell",
            bounds=bounds,
            options={"maxiter": 25, "xtol": 0.01, "ftol": 0.005},
        )
        candidate = candidate | {
            "density_to_original": matrix_from_parameters(optimized.x).tolist(),
            "local_refinement": True,
        }
        ranked.append(score_candidate(candidate))
    ranked.extend(geometric)
    return sorted([item for item in ranked if item["structural_cost"] < 30], key=lambda item: -item["score"])
