import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from homography_lab.experiments.scenealigner.motion_export import exportable_calibrations
from local_tracking.pipeline import file_hash
from mtmc.floor_projection import floor_point
from tools.upload_clock import align_videos, parse_stamp, sampled_anchors
from tools.upload_geometry import (
    localize,
    marker_calibration,
    markers,
    motion_matches,
    motion_overlaps,
    office_overlaps,
    transfer_to_plan,
)


def test_office_geometry_follows_the_recording_hash_and_requires_the_matching_plan(tmp_path, monkeypatch):
    plan = tmp_path / "office.jpg"
    assert cv2.imwrite(str(plan), np.full((320, 400), 255, np.uint8))
    other_plan = tmp_path / "other.jpg"
    assert cv2.imwrite(str(other_plan), np.full((320, 400), 127, np.uint8))
    files = [tmp_path / f"upload_{n}.mkv" for n in range(3)]
    for n, path in enumerate(files):
        path.write_bytes(bytes([n]))
    known = [{"cameras": {f"cam_{n}": {"source_sha256": file_hash(files[n])} for n in (1, 2)}}]
    for n, x_offset in ((1, 10), (2, 30)):
        (tmp_path / f"cam_{n}_calibration.json").write_text(
            json.dumps(
                {
                    "camera_size": [400, 320],
                    "plan_size": [400, 320],
                    "homography": [[1, 0, x_offset], [0, 1, 0], [0, 0, 1]],
                }
            ),
            encoding="utf-8",
        )
    monkeypatch.setattr("tools.upload_geometry.ROOT", tmp_path)
    monkeypatch.setattr("tools.upload_geometry.probe_video", lambda _: SimpleNamespace(width=400, height=320))
    videos = dict(zip(("camera_1", "camera_2", "camera_3"), (files[2], files[0], files[1]), strict=True))

    calibrations, size = localize(plan, videos, known_maps=known)
    assert set(calibrations) == {"camera_1", "camera_3"}
    assert floor_point((100, 100), calibrations["camera_1"], size) == (130, 100)
    assert floor_point((100, 100), calibrations["camera_3"], size) == (110, 100)
    assert office_overlaps(calibrations, [["camera_1", "camera_2"]]) == [["camera_1", "camera_3"]]
    assert localize(other_plan, videos, known_maps=known)[0] == {}


def test_osd_uses_date_and_preserves_media_gap_as_separate_axis(tmp_path, monkeypatch):
    first = parse_stamp("2026-10-08 Thu 23:59:58")
    assert parse_stamp("2026-10-09 00:00:00") - first == 2
    assert parse_stamp("2026-02-30 12:00:00") is None
    times = [i / 20 + (10 if i >= 40 else 0) for i in range(100)]
    readings = [(i, first + i // 20) for i in range(0, 100, 5)]
    anchors = sampled_anchors(times, readings, 20)
    before = next(a for a in anchors if a[0] == times[39])
    after = next(a for a in anchors if a[0] == times[40])
    assert after[0] - before[0] > 10
    assert after[1] - before[1] == pytest.approx(0.05, abs=0.001)
    paths = {name: tmp_path / name for name in ("one", "two")}
    for p in paths.values():
        p.touch()
    monkeypatch.setattr(
        "tools.upload_clock.scan_clock",
        lambda path, roi: [(t, seconds + (2 if path == paths["two"] else 0)) for t, seconds in anchors],
    )
    _, aligned = align_videos(paths, (0, 0, 1, 1), [])
    assert aligned["two"][0]["timestamp_sec"] - aligned["one"][0]["timestamp_sec"] == 2


def test_marker_calibration_requires_spatially_consistent_anchors():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    plan = np.full((320, 400), 255, np.uint8)
    for i, (x, y) in enumerate(((45, 45), (290, 55), (260, 230), (75, 240))):
        plan[y : y + 36, x : x + 36] = cv2.aruco.generateImageMarker(dictionary, i, 36)
    shift = np.float32([[1, 0, 10], [0, 1, 15]])
    image = cv2.warpAffine(plan, shift, (400, 320), borderValue=255)
    target = markers(plan)
    source = markers(image)
    calibration = marker_calibration(source, target, (400, 320), (400, 320))
    assert floor_point((210, 160), calibration, (400, 320)) == pytest.approx((200, 145), abs=0.1)
    source[0] += 70
    assert marker_calibration(source, target, (400, 320), (400, 320)) is None
    assert floor_point((210, 160), None, (400, 320)) is None


def test_motion_edge_needs_repeated_and_close_appearance_matches():
    def row(camera, t, x):
        return {
            "camera_id": camera,
            "timestamp_ms": t * 1000,
            "local_id": 1,
            "confidence": 0.9,
            "footpoint_status": "observed",
            "footpoint_px": (x, 100),
            "embedding": [1.0, 0.0],
        }

    observations = [
        row(camera, i, x) for i in range(12) for camera, x in (("a", 100), ("b", 104), ("c", 200))
    ]
    calibrations = {camera: {"homography": np.eye(3).tolist()} for camera in ("a", "b", "c")}
    assert motion_overlaps(motion_matches(observations), calibrations, (400, 400)) == [["a", "b"]]


def test_lab_diagnostics_are_never_promoted():
    assert (
        exportable_calibrations(
            {
                "absolute_location_verified": False,
                "candidate_calibrations": {"cam_1": {"homography": np.eye(3).tolist()}},
            }
        )
        == {}
    )


def test_motion_transfers_an_absolute_anchor_only_with_independent_support():
    anchor = {"homography": np.eye(3).tolist()}
    points = [(x, y) for x in (100, 150, 200, 250) for y in (80, 135, 190, 245)]
    samples = []
    for start in (0, 10):
        samples.extend(
            {
                "timestamp_sec": start + i / 2,
                "track_ids": [i % 2, i % 2],
                "points": [(x, y), (x + 20, y + 30)],
            }
            for i, (x, y) in enumerate(points)
        )
    fit = transfer_to_plan(samples, anchor, 0, (400, 400), (400, 400))
    assert floor_point((180, 200), fit, (400, 400)) == pytest.approx((160, 170))
    assert transfer_to_plan(samples[:16], anchor, 0, (400, 400), (400, 400)) is None
