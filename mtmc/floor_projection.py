"""Project a camera pixel through a legacy matrix or a lab calibration."""

import numpy as np

from homography_lab.experiments.scenealigner.lens_geometry import project_calibration


def floor_point(point, calibration, size):
    if calibration is None:
        return None
    if isinstance(calibration, dict):
        polygon = calibration.get("region")
        if polygon is not None:
            import cv2

            if cv2.pointPolygonTest(np.float32(polygon), tuple(map(float, point)), False) < 0:
                return None
        points, valid = project_calibration([point], calibration)
        if not valid[0]:
            return None
        x, y = points[0]
    else:
        x, y, scale = calibration @ [*point, 1]
        if not np.isfinite([x, y, scale]).all() or abs(scale) < 1e-9:
            return None
        x, y = x / scale, y / scale
    if 0 <= x < size[0] and 0 <= y < size[1]:
        return float(x), float(y)
    return None
