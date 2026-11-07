"""Absolute plan anchors and motion-validated camera overlaps for uploaded scenes."""

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from homography_lab.experiments.scenealigner.motion_export import exportable_calibrations
from homography_lab.experiments.scenealigner.motion_pairs import balanced_matches, mutual_appearance
from homography_lab.features import background, read_image
from homography_lab.geometry import warp
from local_tracking.pipeline import file_hash
from local_tracking.video import probe_video
from mtmc.floor_projection import floor_point
from tools.export_web_session import load_calibration

ROOT = Path(__file__).resolve().parents[1]


def markers(image):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    corners, ids, _ = cv2.aruco.ArucoDetector(dictionary).detectMarkers(image)
    if ids is None:
        return {}
    labels = ids.ravel().tolist()
    if len(set(labels)) != len(labels):
        raise ValueError("Одинаковые метки ArUco в одном изображении.")
    return {label: corner[0] for label, corner in zip(labels, corners, strict=True)}


def marker_calibration(source, target, camera_size, plan_size):
    ids = sorted(source.keys() & target.keys())
    if len(ids) < 3:
        return None
    camera = np.concatenate([source[k] for k in ids])
    plan = np.concatenate([target[k] for k in ids])
    if (
        cv2.contourArea(cv2.convexHull(camera)) < np.prod(camera_size) * 0.015
        or cv2.contourArea(cv2.convexHull(plan)) < np.prod(plan_size) * 0.005
    ):
        return None
    limit = max(3, np.hypot(*plan_size) * 0.006)
    for index in range(len(ids)):
        test = np.arange(len(camera)) // 4 == index
        fitted, _ = cv2.findHomography(camera[~test], plan[~test], 0)
        if fitted is None or np.max(np.linalg.norm(warp(camera[test], fitted) - plan[test], axis=1)) > limit:
            return None
    matrix, _ = cv2.findHomography(camera, plan, 0)
    if matrix is None or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) != 3:
        return None
    return {
        "camera_size": list(camera_size),
        "plan_size": list(plan_size),
        "homography": matrix.tolist(),
        "region": cv2.convexHull(camera).reshape(-1, 2).tolist(),
        "method": "floor_aruco_holdout",
    }


def lab_calibrations(path, plan, videos, plan_size):
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    accepted = exportable_calibrations(result)
    if not accepted:
        return {}
    if result.get("plan_sha256") != file_hash(plan):
        raise ValueError("Лабораторная калибровка относится к другому плану.")
    calibrations = {}
    for camera, data in accepted.items():
        name = camera.replace("cam_", "camera_", 1) if camera.startswith("cam_") else camera
        if name not in videos or name in calibrations:
            raise ValueError(f"Неизвестная или повторная камера в лабораторном экспорте: {camera}")
        if result.get("source_sha256", {}).get(camera) != file_hash(videos[name]):
            raise ValueError(f"Лабораторная калибровка относится к другой записи: {camera}")
        video = probe_video(videos[name])
        if data["camera_size"] != [video.width, video.height] or data["plan_size"] != list(plan_size):
            raise ValueError(f"Размеры лабораторной калибровки не совпадают: {camera}")
        floor_point((video.width / 2, video.height / 2), data, plan_size)
        calibrations[name] = data | {"method": "accepted_motion_lab"}
    return calibrations


def office_calibrations(plan_path, videos, plan_size, known_maps):
    if file_hash(plan_path) != file_hash(ROOT / "office.jpg"):
        return {}
    by_digest = {
        entry["source_sha256"]: camera
        for bundle in known_maps
        for camera, entry in bundle["cameras"].items()
    }
    calibrated = {}
    for name, path in videos.items():
        camera = by_digest.get(file_hash(path))
        if camera is None:
            continue
        video = probe_video(path)
        source = {"width": video.width, "height": video.height}
        matrix = load_calibration(ROOT, camera, source, plan_size)
        calibrated[name] = {
            "camera_size": [video.width, video.height],
            "plan_size": list(plan_size),
            "homography": matrix.tolist(),
            "method": "verified_office_calibration",
            "office_camera": camera.replace("cam_", "camera_", 1),
        }
    return calibrated


def office_overlaps(calibrated, verified_pairs):
    verified = {frozenset(pair) for pair in verified_pairs}
    office_cameras = {name: data["office_camera"] for name, data in calibrated.items() if "office_camera" in data}
    return [
        [left, right]
        for left, right in combinations(sorted(office_cameras), 2)
        if frozenset((office_cameras[left], office_cameras[right])) in verified
    ]


def localize(plan_path, videos, report=None, *, known_maps=()):
    plan = read_image(plan_path)
    size = plan.shape[1::-1]
    calibrated = office_calibrations(plan_path, videos, size, known_maps)
    if report:
        calibrated.update(lab_calibrations(report, plan_path, videos, size))
    target = markers(plan) if len(calibrated) < len(videos) else {}
    if len(target) >= 3:
        for name, video in videos.items():
            if name in calibrated:
                continue
            try:
                frame, _, info = background(video, count=5)
                if info["camera_motion_detected"]:
                    continue
                found = marker_calibration(markers(frame), target, frame.shape[1::-1], size)
            except (ValueError, cv2.error):
                continue
            if found is not None:
                calibrated[name] = found
    return calibrated, size


def motion_matches(observations):
    frames = defaultdict(lambda: defaultdict(dict))
    for row in observations:
        name = row["camera_id"]
        if (
            not row.get("embedding")
            or row.get("footpoint_status") != "observed"
            or row.get("footpoint_px") is None
        ):
            continue
        frames[row["timestamp_ms"] // 250][name][row["local_id"]] = row
    candidates = defaultdict(list)
    for cameras in frames.values():
        for a, b in combinations(sorted(cameras), 2):
            left, right = list(cameras[a].values()), list(cameras[b].values())
            for i, j, score, _ in mutual_appearance(left, right, minimum=0.65):
                x, y = left[i], right[j]
                if abs(x["timestamp_ms"] - y["timestamp_ms"]) > 180:
                    continue
                candidates[a, b].append(
                    {
                        "track_ids": [x["local_id"], y["local_id"]],
                        "timestamp_sec": (x["timestamp_ms"] + y["timestamp_ms"]) / 2000,
                        "appearance_cosine": score,
                        "foot_confidences": [x["confidence"], y["confidence"]],
                        "points": [x["footpoint_px"], y["footpoint_px"]],
                    }
                )
    return {pair: balanced_matches(matches) for pair, matches in sorted(candidates.items())}


def transfer_to_plan(samples, anchor, anchor_index, camera_size, plan_size):
    train, holdout = [], []
    for row in samples:
        points = row["points"]
        projected = floor_point(points[anchor_index], anchor, plan_size)
        if projected is not None:
            target = train if int(row["timestamp_sec"] // 10) % 2 == 0 else holdout
            target.append((points[1 - anchor_index], projected, tuple(row["track_ids"])))
    if len(train) < 12 or len(holdout) < 8 or len({r[2] for r in train + holdout}) < 2:
        return None
    source = np.float32([r[0] for r in train])
    plan = np.float32([r[1] for r in train])
    limit = min(14, max(3, np.hypot(*plan_size) * 0.014))
    matrix, mask = cv2.findHomography(source, plan, cv2.RANSAC, limit / 2)
    if (
        matrix is None
        or mask is None
        or not np.isfinite(matrix).all()
        or np.linalg.matrix_rank(matrix) != 3
        or mask.sum() < 0.8 * len(train)
    ):
        return None
    hull = cv2.convexHull(source[mask.ravel().astype(bool)])
    if cv2.contourArea(hull) < np.prod(camera_size) * 0.012:
        return None
    checked = [r for r in holdout if cv2.pointPolygonTest(hull, tuple(map(float, r[0])), False) >= 0]
    if len(checked) < 8 or len(checked) < 0.6 * len(holdout):
        return None
    try:
        errors = np.linalg.norm(warp([r[0] for r in checked], matrix) - [r[1] for r in checked], axis=1)
    except ValueError:
        return None
    if np.median(errors) > limit / 2 or np.percentile(errors, 95) > limit:
        return None
    return {
        "camera_size": list(camera_size),
        "plan_size": list(plan_size),
        "homography": matrix.tolist(),
        "region": hull.reshape(-1, 2).tolist(),
        "method": "motion_transfer_holdout",
    }


def extend_calibrations(matches, calibrated, camera_sizes, plan_size):
    proposals = defaultdict(list)
    for pair, samples in matches.items():
        present = [i for i, name in enumerate(pair) if name in calibrated]
        if len(present) != 1:
            continue
        anchor = present[0]
        target = pair[1 - anchor]
        proposal = transfer_to_plan(
            samples, calibrated[pair[anchor]], anchor, camera_sizes[target], plan_size
        )
        if proposal is not None:
            proposals[target].append(proposal)
    # Only transfer directly from an absolute anchor. Multiple competing
    # transforms require a separate joint validation before deployment.
    return calibrated | {name: values[0] for name, values in proposals.items() if len(values) == 1}


def motion_overlaps(matches, calibrated, plan_size):
    overlaps = []
    limit = min(18, max(3, np.hypot(*plan_size) * 0.02))
    for pair, balanced in matches.items():
        if any(name not in calibrated for name in pair):
            continue
        positions = [
            (
                [
                    floor_point(point, calibrated[name], plan_size)
                    for point, name in zip(row["points"], pair, strict=True)
                ],
                row["timestamp_sec"],
            )
            for row in balanced
        ]
        comparable = [(a, b, t) for (a, b), t in positions if a is not None and b is not None]
        support = [t for a, b, t in comparable if np.linalg.norm(np.subtract(a, b)) <= limit]
        if len(support) >= 8 and len(support) >= 0.8 * len(comparable) and max(support) - min(support) >= 8:
            overlaps.append(list(pair))
    return overlaps
