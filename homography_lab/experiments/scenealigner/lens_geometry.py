"""Consistent simple-radial image/point transforms, including non-invertible domains."""

import numpy as np


def undistort_points(points, lens):
    source = np.asarray(points, dtype=float)
    if source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("Expected Nx2 points")
    k = float(lens["k1"])
    intrinsic = np.asarray(lens["K"], dtype=float)
    output = np.asarray(lens["new_K"], dtype=float)
    normalized = (source - intrinsic[:2, 2]) / np.diag(intrinsic)[:2]
    rd = np.linalg.norm(normalized, axis=1)
    valid = np.isfinite(source).all(1)
    model = lens.get("model", "simple_radial")
    if model == "simple_divisional":
        denominator = 1 + k * rd**2
        valid &= (denominator > 1e-9) & (1 - k * rd**2 > 0)
        corrected = normalized / np.maximum(denominator[:, None], 1e-9)
        result = corrected * np.diag(output)[:2] + output[:2, 2]
        result[~valid] = np.nan
        return result, valid
    if model != "simple_radial":
        raise ValueError("Unsupported lens model")
    if abs(k) < 1e-12:
        ru = rd
    else:
        if k < 0:
            limit = np.sqrt(-1 / (3 * k))
            valid &= rd < limit * 2 / 3 * (1 - 1e-9)
            upper = np.full_like(rd, limit)
        else:
            upper = rd.copy()
        lower = np.zeros_like(rd)
        for _ in range(48):
            middle = (lower + upper) / 2
            low = middle * (1 + k * middle**2) < rd
            lower = np.where(low, middle, lower)
            upper = np.where(low, upper, middle)
        ru = (lower + upper) / 2
    ratio = np.divide(ru, rd, out=np.ones_like(ru), where=rd > 1e-12)
    result = normalized * ratio[:, None] * np.diag(output)[:2] + output[:2, 2]
    result[~valid] = np.nan
    return result, valid


def distort_points(points, lens):
    points = np.asarray(points, dtype=float)
    old = np.asarray(lens["K"], dtype=float)
    new = np.asarray(lens["new_K"], dtype=float)
    normalized = (points - new[:2, 2]) / np.diag(new)[:2]
    r2 = np.sum(normalized**2, axis=1)
    k = float(lens["k1"])
    model = lens.get("model", "simple_radial")
    if model == "simple_divisional":
        discriminant = 1 - 4 * k * r2
        valid = np.isfinite(points).all(1) & (discriminant > 0)
        factor = 2 / (1 + np.sqrt(np.maximum(discriminant, 0)))
        result = normalized * factor[:, None] * np.diag(old)[:2] + old[:2, 2]
        result[~valid] = np.nan
        return result, valid
    if model != "simple_radial":
        raise ValueError("Unsupported lens model")
    factor = 1 + k * r2
    valid = np.isfinite(points).all(1) & (1 + 3 * k * r2 > 0) & (factor > 0)
    result = normalized * factor[:, None] * np.diag(old)[:2] + old[:2, 2]
    result[~valid] = np.nan
    return result, valid


def remap_grid(lens):
    width, height = map(int, lens["size"])
    yy, xx = np.mgrid[:height, :width]
    mapped, valid = distort_points(np.column_stack((xx.ravel(), yy.ravel())), lens)
    valid &= ((mapped >= 0) & (mapped < [width - 1, height - 1])).all(1)
    mapped[~valid] = -1
    return (
        mapped[:, 0].reshape(height, width).astype(np.float32),
        mapped[:, 1].reshape(height, width).astype(np.float32),
        valid.reshape(height, width),
    )


def project_calibration(points, prediction):
    source = np.asarray(points, dtype=float)
    valid = np.isfinite(source).all(1)
    if prediction.get("lens_correction_applied", False):
        space = prediction.get("homography_input_space")
        if space not in ("rectified_pixels", "ideal_pixels") or not prediction.get("lens"):
            raise ValueError("Corrected homography needs an explicit lens transform")
        observed = source.copy()
        source, valid = undistort_points(source, prediction["lens"])
        if space == "ideal_pixels":
            # Observed rays need not fit inside an arbitrarily cropped rectified raster.
            size = prediction["lens"]["size"]
            valid &= (observed >= 0).all(1) & (observed < size).all(1)
        else:
            size = prediction["lens"]["rectified_size"]
            valid &= (source >= 0).all(1) & (source < size).all(1)
    if prediction.get("floor_ray_denominator") is not None:
        ray_depth = np.column_stack((source, np.ones(len(source)))) @ np.asarray(
            prediction["floor_ray_denominator"]
        )
        valid &= ray_depth > prediction.get("minimum_floor_ray_denominator", 0.03)
    h = np.asarray(prediction["homography"], dtype=float)
    homogeneous = np.column_stack((source, np.ones(len(source)))) @ h.T
    valid &= np.isfinite(homogeneous).all(1) & (abs(homogeneous[:, 2]) > 1e-9)
    output = np.full_like(source, np.nan)
    output[valid] = homogeneous[valid, :2] / homogeneous[valid, 2:3]
    return output, valid
