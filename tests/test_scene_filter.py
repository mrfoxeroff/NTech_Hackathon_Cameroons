import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from ultralytics.engine.results import Results

from local_tracking.config import Settings
from local_tracking.scene_filter import SceneFilter
from local_tracking.tracker import LocalTracker


def test_reported_camera3_frame4026_reflection_removed_without_removing_physical_people():
    root = Path(__file__).resolve().parents[1]
    case = json.loads((root / "tests/fixtures/camera3_frame4026.json").read_text())
    camera = next(
        c
        for c in yaml.safe_load((root / "configs/cameras.yaml").read_text())["cameras"]
        if c["camera_id"] == "camera_3"
    )
    region = next(r for r in camera["scene_regions"] if r["name"] == "curved_mirror")
    masks = np.zeros((len(case["detections"]), case["height"], case["width"]), np.uint8)
    for mask, detection in zip(masks, case["detections"]):
        cv2.fillPoly(mask, [np.array(p, np.int32) for p in detection["polygons"]], 1)
    im = np.zeros((case["height"], case["width"], 3), np.uint8)
    source = result(im, masks)
    old = SceneFilter([{**region, "min_overlap": 0.85}], case["width"], case["height"])
    assert len(old.apply(source, im)[0].boxes) == 3  # Reproduces the original escape.
    current = SceneFilter([region], case["width"], case["height"])
    filtered, rejected = current.apply(source, im)
    assert len(filtered.boxes) == 2
    assert [r["source_detection_index"] for r in rejected] == [0]
    assert np.array_equal(filtered.masks.data, masks[1:])


@pytest.mark.parametrize("key", ["corridor_reflection", "corridor_border_reflection"])
def test_camera3_corridor_reflection_removed_but_person_behind_open_door_kept(key):
    root = Path(__file__).resolve().parents[1]
    case = json.loads((root / "tests/fixtures/camera3_frame4026.json").read_text())
    camera = next(
        c
        for c in yaml.safe_load((root / "configs/cameras.yaml").read_text())["cameras"]
        if c["camera_id"] == "camera_3"
    )
    region = next(r for r in camera["scene_regions"] if r["name"] == "corridor_glass_reflections")
    masks = np.zeros((2, case["height"], case["width"]), np.uint8)
    for mask, detection in zip(masks, [case[key], case["detections"][2]]):
        cv2.fillPoly(mask, [np.array(p, np.int32) for p in detection["polygons"]], 1)
    im = np.zeros((case["height"], case["width"], 3), np.uint8)
    filtered, rejected = SceneFilter([region], case["width"], case["height"]).apply(result(im, masks), im)
    if key == "corridor_border_reflection":
        # The user accepts diagnostic reflections without IDs. This 0.4766
        # candidate survives geometry but remains below ByteTrack's low cutoff.
        assert len(filtered.boxes) == 2 and not rejected
        filtered.boxes.data[0, 4] = case[key]["confidence"]
        config = yaml.safe_load((root / "configs/cameras.yaml").read_text())
        settings = Settings(
            recording_id="case",
            output_root=root,
            model=root / "unused.pt",
            cameras=[],
            confidence=config["confidence"],
            tracker=config["tracker"],
        )
        tracker = LocalTracker(settings, fps=25)
        for step in range(3):
            assigned = tracker.update(filtered.boxes.cpu().numpy(), im, step / 25)
            assert 0 not in assigned and 1 in assigned
        return
    assert len(filtered.boxes) == 1
    assert [r["source_detection_index"] for r in rejected] == [0]
    assert np.array_equal(filtered.masks.data[0], masks[1])


def result(image, masks):
    boxes = np.array([[0, 0, 40, 40, 0.9, 0]] * len(masks), np.float32)
    return Results(image, path="test", names={0: "person"}, boxes=boxes, masks=np.asarray(masks))


@pytest.mark.parametrize("mode", ["screen", "reflection"])
def test_screen_containment_preserves_person_extending_outside_screen(mode):
    im = np.zeros((40, 40, 3), np.uint8)
    masks = np.zeros((2, 40, 40), np.uint8)
    masks[0, 2:18, 2:18] = 1
    masks[1, 2:38, 22:35] = 1
    filt = SceneFilter(
        [dict(name="tv", mode=mode, polygon=[[0, 0], [1, 0], [1, 0.5], [0, 0.5]], min_overlap=0.9)],
        40,
        40,
    )
    filtered, rejected = filt.apply(result(im, masks), im)
    assert len(filtered.boxes) == 1 and len(filtered.masks) == 1
    assert np.array_equal(filtered.masks.data[0], masks[1])
    assert rejected[0]["source_detection_index"] == 0


def test_static_furniture_removed_but_stationary_foreground_person_kept(tmp_path):
    background = np.full((40, 40, 3), 90, np.uint8)
    path = tmp_path / "background.png"
    cv2.imwrite(str(path), background)
    filt = SceneFilter(
        [
            dict(
                name="table",
                mode="static_object",
                reference=str(path),
                polygon=[[0, 0], [1, 0], [1, 1], [0, 1]],
            )
        ],
        40,
        40,
    )
    masks = np.ones((1, 40, 40), np.uint8)
    assert len(filt.apply(result(background, masks), background)[0].boxes) == 0
    # Uniform illumination changes should not turn furniture into a person.
    brighter = background + 30
    assert len(filt.apply(result(brighter, masks), brighter)[0].boxes) == 0
    foreground = background.copy()
    foreground[5:30, 5:20] = [10, 220, 30]
    assert len(filt.apply(result(foreground, masks), foreground)[0].boxes) == 1
    assert len(filt.apply(result(foreground, masks), foreground)[0].boxes) == 1
