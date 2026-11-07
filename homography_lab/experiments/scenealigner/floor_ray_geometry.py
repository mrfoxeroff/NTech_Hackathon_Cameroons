"""Floor-plane projection from intrinsic parameters and gravity, without predicted depth."""

import cv2
import numpy as np
if __package__:
    from .lens_geometry import undistort_points
else:
    from lens_geometry import undistort_points


def floor_homography(intrinsic, gravity):
    gravity = np.array(gravity, float, copy=True)
    gravity /= np.linalg.norm(gravity)
    right = np.array([1.0, 0, 0])
    if abs(right @ gravity) > 0.99:
        right = np.array([0.0, 0, 1.0])
    forward = np.cross(right, gravity)
    forward /= np.linalg.norm(forward)
    right = np.cross(gravity, forward)
    # Unit camera height; unknown physical height becomes the map similarity scale.
    return np.stack((right, -forward, gravity)) @ np.linalg.inv(np.asarray(intrinsic, float))


def project_floor(points, lens, homography, max_distance=25.0):
    ideal, valid = undistort_points(points, lens)
    h = np.column_stack((ideal, np.ones(len(ideal)))) @ homography.T
    valid &= np.isfinite(h).all(1) & (h[:, 2] > 0.03)
    result = np.full((len(h), 2), np.nan)
    result[valid] = h[valid, :2] / h[valid, 2:3]
    valid &= np.linalg.norm(result, axis=1) < max_distance
    result[~valid] = np.nan
    return result, valid


def fit_density_canvas(wall_points, floor_points, size=512):
    points = np.concatenate((wall_points, floor_points))
    lo, hi = np.quantile(points, [0.01, 0.99], axis=0)
    extent = max(hi - lo)
    if not np.isfinite(extent) or extent <= 1e-6:
        raise ValueError("Degenerate visible floor geometry")
    scale = 0.8 * size / extent
    center = (lo + hi) / 2
    matrix = np.array(
        [[scale, 0, size / 2 - scale * center[0]], [0, scale, size / 2 - scale * center[1]], [0, 0, 1.0]]
    )
    return matrix


def transform_points(points, matrix):
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def visible_floor_rays(points, maximum=24):
    # Farthest robust floor observation per angular bin is a free-space lower bound.
    angle = np.arctan2(points[:, 1], points[:, 0])
    radius = np.linalg.norm(points, axis=1)
    bins = np.floor((angle + np.pi) / (2 * np.pi) * maximum).astype(int).clip(0, maximum - 1)
    chosen = []
    for index in np.unique(bins):
        members = np.flatnonzero(bins == index)
        if len(members) < 3:
            continue
        target = np.quantile(radius[members], 0.9)
        chosen.append(points[members[np.argmin(abs(radius[members] - target))]])
    return np.asarray(chosen).reshape(-1, 2)


def ray_blocking_fraction(matrix, origin, ray_ends, plan_walls):
    if not len(ray_ends):
        return 0.0
    height, width = plan_walls.shape
    start = transform_points(np.asarray(origin).reshape(1, 2), matrix)[0]
    ends = transform_points(ray_ends, matrix)
    length = np.linalg.norm(ends - start, axis=1)
    if not np.isfinite(length).all():
        return 1.0
    # Fixed-count sampling can jump over thin plan walls in long rays.
    steps = max(96, int(np.ceil(min(float(length.max()), 2 * np.hypot(width, height)) / 0.75)) + 1)
    fractions = np.linspace(0.0, 0.95, steps)
    samples = start[None, None] + (ends - start)[:, None] * fractions[None, :, None]
    distance = length[:, None] * fractions[None]
    # A ceiling-mounted camera may project onto the wall itself; allow its first 3 px.
    relevant = (distance >= 3.0) & (length[:, None] >= 5.0)
    xy = np.rint(samples).astype(int)
    inside = ((xy >= 0) & (xy < [width, height])).all(-1)
    clipped = xy.copy()
    clipped[..., 0] = np.clip(clipped[..., 0], 0, width - 1)
    clipped[..., 1] = np.clip(clipped[..., 1], 0, height - 1)
    blocked = ((plan_walls[clipped[..., 1], clipped[..., 0]] > 0) | ~inside) & relevant
    return float(np.any(blocked, axis=1).mean())


def boundary_geometry(floor_mask, boundary_mask, lens, gravity):
    floor = floor_mask.astype(np.uint8)
    boundary = boundary_mask.astype(np.uint8)
    height, width = floor.shape
    k = max(3, round(max(width, height) * 0.015))
    near_boundary = cv2.distanceTransform((boundary == 0).astype(np.uint8), cv2.DIST_L2, 3) <= k
    floor_edge = (floor > 0) & (cv2.erode(floor, np.ones((5, 5), np.uint8)) == 0)
    floor_edge &= near_boundary
    margin = round(0.02 * max(width, height))
    floor_edge[:margin] = floor_edge[-margin:] = False
    floor_edge[:, :margin] = floor_edge[:, -margin:] = False
    y, x = np.where(floor_edge)
    source = np.column_stack((x[::5], y[::5])).astype(float)
    homography = floor_homography(lens["new_K"], gravity)
    wall_points, valid = project_floor(source, lens, homography)
    wall_points = wall_points[valid]
    source = source[valid]
    distance = cv2.distanceTransform((floor > 0).astype(np.uint8), cv2.DIST_L2, 5)
    yy, xx = np.mgrid[:height:16, :width:16]
    good = distance[yy, xx] >= max(6.0, 0.008 * max(width, height))
    samples = np.column_stack((xx[good], yy[good])).astype(float)
    floors, valid = project_floor(samples, lens, homography)
    floors = floors[valid]
    if len(wall_points) < 30 or len(floors) < 30:
        raise ValueError("Insufficient physical floor/boundary observations")
    canvas = fit_density_canvas(wall_points, floors)
    wall_pixels = transform_points(wall_points, canvas)
    floor_pixels = transform_points(floors, canvas)
    ray_ends = transform_points(visible_floor_rays(floors), canvas)
    origin = transform_points(np.zeros((1, 2)), canvas)[0]
    image = np.full((512, 512), 255, np.uint8)
    for point in np.rint(wall_pixels).astype(int):
        if ((point >= 0) & (point < 512)).all():
            cv2.circle(image, tuple(point), 1, 0, -1)
    image = cv2.erode(image, np.ones((3, 3), np.uint8))
    return {
        "image": image,
        "wall_points": wall_points,
        "floor_points": floors,
        "floor_density": floor_pixels,
        "camera_density": origin,
        "ray_endpoints_density": ray_ends,
        "image_to_density": canvas @ homography,
        "normalization": canvas,
        "floor_homography": homography,
        "source_boundary_pixels": source,
    }
