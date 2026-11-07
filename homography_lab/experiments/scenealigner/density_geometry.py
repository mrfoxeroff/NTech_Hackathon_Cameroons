"""Ground-plane raster geometry independent of neural reconstruction packages."""

import numpy as np
from PIL import Image


def get_density_coordinates(points, projection):
    horizontal = np.asarray(points)[:, [0, 2]]
    origin = np.array([projection["min_x"], projection["min_z"]])
    padding = np.array([projection["cx"], projection["cz"]])
    pixels = (horizontal - origin) * projection["scale"] + padding
    pixels[:, 1] = projection["density_size"] - 1 - pixels[:, 1]
    return pixels.astype(np.float32)


def render_density_map(points, projection, gamma_val=0.5):
    size = int(projection["density_size"])
    pixels = get_density_coordinates(points, projection).astype(np.int32)
    pixels = pixels[((pixels >= 0) & (pixels < size)).all(axis=1)]
    counts = np.zeros((size, size), np.float32)
    if len(pixels):
        np.add.at(counts, (pixels[:, 1], pixels[:, 0]), 1)
    intensity = np.power(counts, gamma_val)
    peak = intensity.max()
    if peak > 0:
        intensity = intensity / peak * 255.0
    return Image.fromarray(np.clip(255.0 - intensity, 0, 255).astype(np.uint8)).convert("RGB")
