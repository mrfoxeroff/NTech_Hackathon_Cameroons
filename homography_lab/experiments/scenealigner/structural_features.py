"""Automatic wall/door proposals on a drawing and semantic 3D density maps."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _merged_wall_runs(mask, vertical=False):
    image = mask.T if vertical else mask
    found = cv2.HoughLinesP(image, 1, np.pi / 180, threshold=18, minLineLength=20, maxLineGap=2)
    lines = []
    for line in [] if found is None else found[:, 0]:
        x1, y1, x2, y2 = map(int, line)
        if abs(y2 - y1) <= 2 and abs(x2 - x1) >= 20:
            lines.append((round((y1 + y2) / 2), min(x1, x2), max(x1, x2)))
    clusters = []
    for y, x1, x2 in sorted(lines):
        cluster = next((c for c in clusters if abs(y - np.median(c["ys"])) <= 3), None)
        if cluster is None:
            cluster = {"ys": [], "runs": []}
            clusters.append(cluster)
        cluster["ys"].append(y)
        cluster["runs"].append((x1, x2))
    output = []
    for cluster in clusters:
        merged = []
        for a, b in sorted(cluster["runs"]):
            if merged and a <= merged[-1][1] + 3:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        output.append((int(np.median(cluster["ys"])), merged))
    return output


def plan_structure(image):
    gray = np.asarray(image.convert("L"))
    ink = (gray < 110).astype(np.uint8) * 255
    core = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 22), np.uint8))
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((22, 2), np.uint8))
    walls = cv2.bitwise_or(core, cv2.bitwise_or(horizontal, vertical))
    # Remove tiny text/furniture remnants while retaining long thin partitions.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(walls)
    keep = np.zeros(count, np.uint8)
    for i in range(1, count):
        keep[i] = (
            255
            if stats[i, cv2.CC_STAT_AREA] >= 35
            and max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) >= 16
            else 0
        )
    walls = keep[labels]
    distance_to_ink = cv2.distanceTransform((ink == 0).astype(np.uint8), cv2.DIST_L2, 3)
    ink_thickness = cv2.distanceTransform((ink > 0).astype(np.uint8), cv2.DIST_L2, 3)
    candidates = []
    height, width = gray.shape
    for swap in [False, True]:
        for coordinate, runs in _merged_wall_runs(walls, swap):
            for (_, end), (start, _) in zip(runs, runs[1:]):
                radius = start - end
                if not 5 <= radius <= 0.12 * min(width, height):
                    continue
                best = None
                sample = np.array([coordinate, end - 6] if swap else [end - 6, coordinate], int)
                thickness = max(0, float(ink_thickness[sample[1], sample[0]]) - 1)
                hypotheses = [
                    (end, radius),
                    (start, radius),
                    (end - thickness, radius + 2 * thickness),
                    (start + thickness, radius + 2 * thickness),
                ]
                for hinge, arc_radius in hypotheses:
                    center = np.array([coordinate, hinge] if swap else [hinge, coordinate], float)
                    for quadrant in range(4):
                        angles = np.deg2rad(np.linspace(quadrant * 90 + 20, quadrant * 90 + 70, 12))
                        arc = center + arc_radius * np.column_stack((np.cos(angles), np.sin(angles)))
                        x, y = np.rint(arc).astype(int).T
                        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
                        score = (
                            float((distance_to_ink[y[valid], x[valid]] <= 1.5).mean()) if valid.all() else 0.0
                        )
                        # An arc without a leaf can be floor hatching or furniture.
                        leaf_angle = quadrant * 90 if swap else quadrant * 90 + 90
                        a = np.deg2rad(leaf_angle)
                        leaf = center + np.linspace(0.25 * arc_radius, 0.9 * arc_radius, 12)[:, None] * [
                            np.cos(a),
                            np.sin(a),
                        ]
                        lx, ly = np.rint(leaf).astype(int).T
                        lv = (lx >= 0) & (lx < width) & (ly >= 0) & (ly < height)
                        leaf_score = (
                            float((distance_to_ink[ly[lv], lx[lv]] <= 1.5).mean()) if lv.all() else 0.0
                        )
                        if (
                            score >= 0.75
                            and leaf_score >= 0.6
                            and (best is None or score + leaf_score > best["score"])
                        ):
                            midpoint = np.array(
                                [coordinate, (end + start) / 2] if swap else [(end + start) / 2, coordinate]
                            )
                            best = {
                                "center": midpoint.tolist(),
                                "hinge": center.tolist(),
                                "radius": arc_radius,
                                "quadrant": quadrant,
                                "leaf_angle": leaf_angle,
                                "score": score + leaf_score,
                                "status": "unverified_wall_gap_arc_leaf_proposal",
                            }
                if best and all(
                    np.linalg.norm(np.array(best["center"]) - p["center"]) > 5 for p in candidates
                ):
                    candidates.append(best)
    result = np.full(gray.shape, 255, np.uint8)
    result[walls > 0] = 0
    for door in candidates:
        center = tuple(np.rint(door["hinge"]).astype(int))
        radius = int(door["radius"])
        cv2.ellipse(
            result,
            center,
            (radius, radius),
            0,
            door["quadrant"] * 90,
            (door["quadrant"] + 1) * 90,
            110,
            1,
            cv2.LINE_AA,
        )
        angle = np.deg2rad(door["leaf_angle"])
        end = tuple(np.rint(np.array(center) + radius * np.array([np.cos(angle), np.sin(angle)])).astype(int))
        cv2.line(result, center, end, 110, 1, cv2.LINE_AA)
    return Image.fromarray(result).convert("RGB"), walls, candidates


def processed_mask(path, image_size, shape):
    original = cv2.imread(str(path), 0)
    if original is None:
        raise ValueError(f"Missing semantic mask: {path}")
    width, height = map(int, image_size)
    full_h, full_w = shape
    px, py = (full_w - width) // 2, (full_h - height) // 2
    result = np.zeros(shape, np.uint8)
    result[py : py + height, px : px + width] = cv2.resize(
        original, (width, height), interpolation=cv2.INTER_NEAREST
    )
    return result > 128


def structural_density(recon, projection, images, semantic_root):
    from density_geometry import get_density_coordinates, render_density_map

    all_walls, all_doors, landmarks = [], [], []
    diagnostic = {"cameras": {}, "manual_landmarks_used": False}
    for i, path in enumerate(images):
        name = Path(path).stem
        points = recon["points_cam0"][i] @ projection["R_rect"].T
        confidence = recon["conf"][i]
        wall = processed_mask(
            semantic_root / "wall" / f"{name}.png", recon["image_sizes"][i], confidence.shape
        )
        door = processed_mask(
            semantic_root / "door" / f"{name}.png", recon["image_sizes"][i], confidence.shape
        )
        # Surface normals suppress horizontal furniture misclassified as a wall.
        dx = np.gradient(points, axis=1)
        dy = np.gradient(points, axis=0)
        normals = np.cross(dx, dy)
        lengths = np.linalg.norm(normals, axis=-1)
        vertical_surface = abs(normals[..., 1]) / np.maximum(lengths, 1e-9) < 0.55
        good = (
            np.isfinite(points).all(-1)
            & np.isfinite(confidence)
            & (confidence >= np.quantile(confidence, 0.45))
        )
        wall_points = points[wall & ~door & good & vertical_surface]
        door_points = points[door & good & vertical_surface]
        all_walls.append(wall_points)
        all_doors.append(door_points)
        count, components, stats, _ = cv2.connectedComponentsWithStats(door.astype(np.uint8))
        for index in range(1, count):
            if stats[index, cv2.CC_STAT_AREA] < 50:
                continue
            use = (components == index) & good
            if use.sum() < 25:
                continue
            center = get_density_coordinates(np.median(points[use], axis=0, keepdims=True), projection)[0]
            if (
                np.isfinite(center).all()
                and (center >= 0).all()
                and (center < projection["density_size"]).all()
                and all(np.linalg.norm(center - p) > 12 for p in landmarks)
            ):
                landmarks.append(center)
        diagnostic["cameras"][name] = {"wall_points": len(wall_points), "door_points": len(door_points)}
    walls = np.concatenate(all_walls)
    doors = np.concatenate(all_doors)
    if len(walls) < 50:
        raise ValueError("Insufficient neural wall pixels for structural matching")
    wall_image = render_density_map(walls, projection)
    door_image = (
        render_density_map(doors, projection) if len(doors) else Image.new("RGB", (512, 512), "white")
    )
    mixed = np.minimum(np.asarray(wall_image), 150 + np.asarray(door_image).astype(float) * 105 / 255).astype(
        np.uint8
    )
    diagnostic["door_centers_density"] = [p.tolist() for p in landmarks]
    diagnostic["wall_points"] = len(walls)
    return (
        Image.fromarray(mixed),
        np.asarray(wall_image.convert("L")),
        np.array(landmarks).reshape(-1, 2),
        diagnostic,
    )
