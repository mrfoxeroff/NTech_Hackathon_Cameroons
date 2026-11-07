"""Structural landmark proposals; photo/drawing matches are never trusted blindly."""

import hashlib
import json
from pathlib import Path

import av
import cv2
import numpy as np

if __package__:
    from .geometry import fit_calibration
else:
    from geometry import fit_calibration


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_image(path):
    image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def write_image(path, image):
    ok, data = cv2.imencode(Path(path).suffix, image)
    if not ok:
        raise ValueError("Image encoding failed")
    data.tofile(path)


def small(image):
    scale = min(1.0, 1280 / max(image.shape[:2]))
    return cv2.resize(image, None, fx=scale, fy=scale), scale


def motion_between(first, second):
    a, _ = small(first)
    b = cv2.resize(second, (a.shape[1], a.shape[0]))
    orb = cv2.ORB_create(2000)
    ka, da = orb.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None)
    kb, db = orb.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)
    if da is None or db is None:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    matches = [m for m in matches if m.distance < 45]
    if len(matches) < 20:
        return None
    x = np.float32([ka[m.queryIdx].pt for m in matches])
    y = np.float32([kb[m.trainIdx].pt for m in matches])
    matrix, mask = cv2.estimateAffinePartial2D(x, y, method=cv2.RANSAC, ransacReprojThreshold=2)
    if matrix is None or mask.mean() < 0.4:
        return None
    corners = np.float32([[0, 0], [a.shape[1], 0], [0, a.shape[0]], [a.shape[1], a.shape[0]]])
    moved = corners @ matrix[:, :2].T + matrix[:, 2]
    return float(np.median(np.linalg.norm(moved - corners, axis=1)) / np.hypot(*a.shape[:2]))


def background(path, count=7):
    frames, times = [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.thread_count = 2
        first = next(container.decode(video=0))
        if first.pts is None:
            raise ValueError("Video must have source timestamps")
        origin = float(first.pts * first.time_base)
        duration = (
            float(stream.duration * stream.time_base)
            if stream.duration
            else float(container.duration / av.time_base)
        )
        if duration <= 0:
            raise ValueError("Could not determine video duration")
        for time in np.linspace(0.01, 0.95, count) * duration:
            container.seek(int((time + origin) / stream.time_base), stream=stream, backward=True)
            for frame in container.decode(video=0):
                if frame.pts is None:
                    continue
                actual = float(frame.pts * frame.time_base) - origin
                if actual >= time:
                    frames.append(frame.to_ndarray(format="bgr24"))
                    times.append(actual)
                    break
    if len(frames) < 3 or any(f.shape != frames[0].shape for f in frames):
        raise ValueError("Need at least three same-resolution frames for a background")
    motions = [motion_between(frames[0], f) for f in frames[1:]]
    moving = any(m is not None and m > 0.003 for m in motions)
    # Never create an unaligned median background for a moving camera.
    representative = frames[0] if moving else np.median(np.stack(frames), axis=0).astype(np.uint8)
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames])
    stable = np.uint8(np.percentile(abs(gray.astype(float) - np.median(gray, axis=0)), 80, axis=0) < 24) * 255
    return (
        representative,
        stable,
        {
            "sample_media_seconds": times,
            "camera_motion_detected": moving,
            "relative_image_motion": motions,
            "background_method": "first_sample_moving_camera" if moving else "temporal_median",
            "stable_pixel_fraction": float((stable > 0).mean()),
        },
    )


def structural_features(image, stable_mask=None):
    reduced, scale = small(image)
    gray = cv2.cvtColor(reduced, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(2, (8, 8)).apply(gray)
    mask = cv2.resize(stable_mask, (gray.shape[1], gray.shape[0])) if stable_mask is not None else None
    found = cv2.createLineSegmentDetector().detect(gray)[0]
    lines = [] if found is None else [line.ravel() for line in found]
    minimum = 0.035 * np.hypot(*gray.shape)
    lines = sorted(
        [line for line in lines if np.linalg.norm(line[:2] - line[2:]) >= minimum],
        key=lambda line: np.linalg.norm(line[:2] - line[2:]),
        reverse=True,
    )[:120]
    proposed = []
    for line in lines:
        for xy in (line[:2], line[2:]):
            x, y = np.rint(xy).astype(int)
            if mask is not None and not mask[min(y, mask.shape[0] - 1), min(x, mask.shape[1] - 1)]:
                continue
            if not any(np.linalg.norm(xy - p) < 14 for p in proposed):
                proposed.append(xy)
    corners = cv2.goodFeaturesToTrack(gray, 120, 0.035, 18, mask=mask)
    for point in [] if corners is None else corners[:, 0]:
        if not any(np.linalg.norm(point - p) < 14 for p in proposed):
            proposed.append(point)
    return {
        "segments": [(line / scale).tolist() for line in lines],
        "candidates": [
            {"id": i + 1, "xy": (p / scale).tolist(), "kind": "unverified_structural_corner"}
            for i, p in enumerate(proposed[:180])
        ],
        "note": "Candidates may belong to furniture, text or ceiling. Confirm floor-plane identity manually.",
    }


def automatic_proposal(camera, plan, stable_mask=None):
    left, sl = small(camera)
    right, sr = small(plan)
    detector = cv2.SIFT_create(nfeatures=3500)
    mask = cv2.resize(stable_mask, (left.shape[1], left.shape[0])) if stable_mask is not None else None
    a, x = detector.detectAndCompute(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), mask)
    b, y = detector.detectAndCompute(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY), None)
    matches = []
    if x is not None and y is not None and len(x) >= 2 and len(y) >= 2:
        matcher = cv2.BFMatcher()
        forward = [
            p
            for pair in matcher.knnMatch(x, y, k=2)
            if len(pair) == 2
            for p, q in [pair]
            if p.distance < 0.65 * q.distance
        ]
        backward = {
            p.queryIdx: p.trainIdx
            for pair in matcher.knnMatch(y, x, k=2)
            if len(pair) == 2
            for p, q in [pair]
            if p.distance < 0.65 * q.distance
        }
        matches = [m for m in forward if backward.get(m.trainIdx) == m.queryIdx]
    report = {
        "method": "mutual_sift_ratio_065",
        "tentative_matches": len(matches),
        "status": "needs_landmark_confirmation",
        "pairs": [],
        "reason": "Photo and schematic have different appearance; no reliable automatic correspondence.",
    }
    if len(matches) < 10:
        return report
    pairs = [
        {
            "id": str(i + 1),
            "camera": (np.array(a[m.queryIdx].pt) / sl).tolist(),
            "plan": (np.array(b[m.trainIdx].pt) / sr).tolist(),
            "role": "fit",
            "kind": "automatic_unverified",
        }
        for i, m in enumerate(matches)
    ]
    try:
        calibration = fit_calibration(pairs, camera.shape[1::-1], plan.shape[1::-1])
        q = calibration["quality"]
        if q["inlier_count"] >= 10 and q["inlier_ratio"] >= 0.7 and q["camera_hull_fraction"] >= 0.03:
            report.update(
                status="unverified_proposal",
                pairs=[p for p, good in zip(pairs, calibration["inliers"]) if good],
                reason="Geometric candidate only: physical floor landmarks and independent checks are required.",
            )
    except ValueError as error:
        report["reason"] = str(error)
    return report


def prepare(plan_path, videos, images, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new output directory")
    if set(videos) & set(images) or not videos and not images:
        raise ValueError("Provide distinct camera IDs and at least one camera")
    for name in set(videos) | set(images):
        if not name or any(not (ch.isalnum() or ch in "-_") for ch in name):
            raise ValueError("Camera IDs may contain letters, digits, hyphens and underscores")
    output.mkdir(parents=True)
    plan = read_image(plan_path)
    write_image(output / "plan.png", plan)
    state = {
        "status": "running",
        "plan": {
            "url": "/media/plan.png",
            "size": list(plan.shape[1::-1]),
            "sha256": digest(plan_path),
            "features": structural_features(plan),
        },
        "cameras": {},
    }
    path = output / "session.json"
    try:
        for name, source in (videos | images).items():
            if name in videos:
                image, stable, metadata = background(source)
            else:
                image, stable, metadata = (
                    read_image(source),
                    None,
                    {"background_method": "provided_frame", "camera_motion_detected": False},
                )
            write_image(output / f"{name}.jpg", image)
            proposal = automatic_proposal(image, plan, stable)
            state["cameras"][name] = {
                "url": f"/media/{name}.jpg",
                "size": list(image.shape[1::-1]),
                "source": str(Path(source).resolve()),
                "source_sha256": digest(source),
                "reference_image_sha256": digest(output / f"{name}.jpg"),
                "features": structural_features(image, stable),
                "automatic": proposal,
                "background": metadata,
            }
            print(
                name,
                "candidates:",
                len(state["cameras"][name]["features"]["candidates"]),
                "automatic:",
                proposal["status"],
                "matches:",
                proposal["tentative_matches"],
                flush=True,
            )
        state["status"] = "complete"
    except BaseException:
        state["status"] = "failed"
        raise
    finally:
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return state
