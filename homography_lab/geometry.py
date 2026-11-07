"""Planar camera calibration, independent checks and conservative projection."""

import math
from datetime import UTC, datetime

import cv2
import numpy as np


def points(value):
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 2 or not np.isfinite(a).all():
        raise ValueError("Expected finite Nx2 coordinates")
    return a


def warp(value, matrix):
    a = points(value)
    homogeneous = np.column_stack((a, np.ones(len(a)))) @ np.asarray(matrix).T
    if not np.isfinite(homogeneous).all() or np.any(abs(homogeneous[:, 2]) < 1e-10):
        raise ValueError("Projection crosses the projective horizon")
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def validate_intrinsics(value, size):
    if value is None:
        return None
    k = np.asarray(value["camera_matrix"], dtype=float)
    d = np.asarray(value["distortion"], dtype=float).reshape(-1)
    if (
        k.shape != (3, 3)
        or not np.isfinite(k).all()
        or not np.isfinite(d).all()
        or k[0, 0] <= 0
        or k[1, 1] <= 0
        or not np.allclose(k[2], [0, 0, 1])
        or abs(k[0, 1]) > 1e-8
        or abs(k[1, 0]) > 1e-8
        or len(d) not in (4, 5, 8, 12, 14)
    ):
        raise ValueError("Invalid OpenCV pinhole intrinsics")
    if list(value["image_size"]) != list(size):
        raise ValueError("Intrinsics resolution must match the camera image")
    return {"camera_matrix": k.tolist(), "distortion": d.tolist(), "image_size": list(size)}


def undistort(value, intrinsics):
    a = points(value)
    if intrinsics is None:
        return a
    k = np.asarray(intrinsics["camera_matrix"])
    return cv2.undistortPoints(a[:, None], k, np.asarray(intrinsics["distortion"]), P=k)[:, 0]


def inverse_project(value, calibration):
    a = warp(value, np.linalg.inv(calibration["homography"]))
    intr = calibration.get("intrinsics")
    if intr is None:
        return a
    k = np.asarray(intr["camera_matrix"])
    normalized = np.column_stack(
        ((a[:, 0] - k[0, 2]) / k[0, 0], (a[:, 1] - k[1, 2]) / k[1, 1], np.ones(len(a)))
    )
    return cv2.projectPoints(normalized, np.zeros(3), np.zeros(3), k, np.asarray(intr["distortion"]))[0][:, 0]


def hull_area(a):
    return float(cv2.contourArea(cv2.convexHull(np.float32(a))))


def spread(a, size):
    area = hull_area(a)
    if area / math.prod(size) < 0.0001:
        raise ValueError("Points are collinear or occupy too small an area")
    # Thin almost-collinear sets are unstable even on a small plan.
    singular = np.linalg.svd(a - a.mean(axis=0), compute_uv=False)
    if singular[-1] / max(singular[0], 1e-12) < 0.015:
        raise ValueError("Point configuration is almost collinear")
    return area / math.prod(size)


def fit_calibration(
    pairs,
    camera_size,
    plan_size,
    *,
    threshold_px=4.0,
    max_check_error_px=6.0,
    intrinsics=None,
    meters_per_pixel=None,
):
    """Check points NEVER participate in RANSAC or final estimation."""
    for size in (camera_size, plan_size):
        if len(size) != 2 or any(type(v) is not int or v < 2 for v in size):
            raise ValueError("Image sizes must contain two positive integer dimensions")
    if any(not math.isfinite(x) or x <= 0 for x in (threshold_px, max_check_error_px)):
        raise ValueError("Error thresholds must be positive and finite")
    if meters_per_pixel is not None and (not math.isfinite(meters_per_pixel) or meters_per_pixel <= 0):
        raise ValueError("Metric scale must be positive")
    intrinsics = validate_intrinsics(intrinsics, camera_size)
    if any(p.get("role", "fit") not in ("fit", "check") for p in pairs):
        raise ValueError("Each point must have role fit or check")
    training = [p for p in pairs if p.get("role", "fit") == "fit"]
    checking = [p for p in pairs if p.get("role") == "check"]
    if len(training) < 4:
        raise ValueError("At least four floor-plane point pairs are required")
    raw = points([p["camera"] for p in pairs])
    dst_all = points([p["plan"] for p in pairs])
    for a, size in ((raw, camera_size), (dst_all, plan_size)):
        if np.any(a < 0) or np.any(a >= np.asarray(size)):
            raise ValueError("Landmark lies outside the declared image")
        distances = np.linalg.norm(a[:, None] - a[None, :], axis=2)
        np.fill_diagonal(distances, math.inf)
        if np.any(distances < 2):
            raise ValueError("Repeated/near-duplicate landmark; checks must be independent points")
    src = undistort([p["camera"] for p in training], intrinsics)
    dst = points([p["plan"] for p in training])
    camera_coverage, plan_coverage = spread(src, camera_size), spread(dst, plan_size)
    cv2.setRNGSeed(42)
    h, mask = cv2.findHomography(src, dst, cv2.RANSAC, threshold_px, maxIters=10000, confidence=0.999)
    if h is None or not np.isfinite(h).all() or np.linalg.matrix_rank(h) != 3:
        raise ValueError("Could not estimate a non-degenerate homography")
    inliers = mask.ravel().astype(bool)
    if inliers.sum() < 4:
        raise ValueError("Fewer than four geometrically consistent landmarks")
    spread(src[inliers], camera_size)
    spread(dst[inliers], plan_size)
    # Check that the projective horizon does not cross the calibrated region.
    denominator = np.column_stack((src[inliers], np.ones(inliers.sum()))) @ h[2]
    if np.min(denominator) * np.max(denominator) <= 0:
        raise ValueError("Projective horizon intersects the calibrated region")
    fit_errors = np.linalg.norm(warp(src, h) - dst, axis=1)
    check_errors = []
    if checking:
        predicted = warp(undistort([p["camera"] for p in checking], intrinsics), h)
        check_errors = np.linalg.norm(predicted - points([p["plan"] for p in checking]), axis=1).tolist()
    warnings = []
    if inliers.sum() < 6:
        warnings.append("Нужно не менее 6 согласованных обучающих точек для проверенной калибровки")
    if len(checking) < 2:
        warnings.append("Нужно минимум 2 отдельные контрольные точки, не участвующие в подборе матрицы")
    if inliers.mean() < 0.7:
        warnings.append("Слишком большая доля несогласованных соответствий")
    if check_errors and max(check_errors) > max_check_error_px:
        warnings.append("Ошибка на контрольных точках превышает допустимую")
    validated = (
        inliers.sum() >= 6
        and len(checking) >= 2
        and inliers.mean() >= 0.7
        and max(check_errors) <= max_check_error_px
    )
    quality = {
        "fit_count": len(training),
        "inlier_count": int(inliers.sum()),
        "inlier_ratio": float(inliers.mean()),
        "fit_rmse_px": float(np.sqrt(np.mean(fit_errors[inliers] ** 2))),
        "check_count": len(checking),
        "check_errors_px": check_errors,
        "check_max_error_px": max(check_errors) if check_errors else None,
        "camera_hull_fraction": camera_coverage,
        "plan_hull_fraction": plan_coverage,
        "warnings": warnings,
    }
    if meters_per_pixel is not None and check_errors:
        quality["check_max_error_m"] = max(check_errors) * meters_per_pixel
    result = {
        "schema_version": "homography-lab/1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "validated_on_check_points" if validated else "draft",
        "coordinate_convention": "raw_camera_pixels_to_plan_pixels",
        "camera_size": list(camera_size),
        "plan_size": list(plan_size),
        "homography": (h / h[2, 2] if abs(h[2, 2]) > 1e-10 else h).tolist(),
        "camera_points": [p["camera"] for p in training],
        "plan_points": [p["plan"] for p in training],
        "inliers": inliers.tolist(),
        "errors_plan_px": fit_errors.tolist(),
        "pairs": pairs,
        "quality": quality,
        "intrinsics": intrinsics,
        "lens_correction_applied": intrinsics is not None,
        "meters_per_pixel": meters_per_pixel,
        "calibrated_region_camera": cv2.convexHull(np.float32(src[inliers])).reshape(-1, 2).tolist(),
        "settings": {"ransac_threshold_px": threshold_px, "max_check_error_px": max_check_error_px},
    }
    return result


def project_point(point, calibration, *, allow_draft=False, allow_extrapolation=False):
    a = points([point])
    if np.any(a < 0) or np.any(a >= np.asarray(calibration["camera_size"])):
        return {"map_xy": None, "status": "outside_camera"}
    if calibration["status"] != "validated_on_check_points" and not allow_draft:
        return {"map_xy": None, "status": "unverified_calibration"}
    corrected = undistort(a, calibration.get("intrinsics"))
    polygon = np.float32(calibration["calibrated_region_camera"])
    extrapolated = cv2.pointPolygonTest(polygon, tuple(map(float, corrected[0])), False) < 0
    if extrapolated and not allow_extrapolation:
        return {"map_xy": None, "status": "outside_calibrated_region"}
    xy = warp(corrected, calibration["homography"])[0]
    if np.any(xy < 0) or np.any(xy >= np.asarray(calibration["plan_size"])):
        return {"map_xy": None, "status": "outside_plan"}
    return {"map_xy": xy.tolist(), "status": "extrapolated" if extrapolated else "projected"}


def project_observation(row, calibration, **kwargs):
    if row.get("camera_id") and calibration.get("camera_id") and row["camera_id"] != calibration["camera_id"]:
        raise ValueError("Observation belongs to a different camera")
    if (
        row.get("frame_width")
        and row.get("frame_height")
        and [row["frame_width"], row["frame_height"]] != calibration["camera_size"]
    ):
        raise ValueError("Observation resolution differs from calibration")
    if (
        row.get("footpoint_px") is None
        or row.get("footpoint_status") == "unknown"
        or row.get("touches_frame_bottom")
    ):
        projection = {"map_xy": None, "status": "unknown_footpoint"}
    else:
        projection = project_point(row["footpoint_px"], calibration, **kwargs)
    return row | {
        "map_xy": projection["map_xy"],
        "homography_status": projection["status"],
        "map_position_source": row.get("footpoint_status", "observed"),
    }
