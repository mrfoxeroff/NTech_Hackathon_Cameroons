import cv2
import numpy as np
from ultralytics.engine.results import Results

from local_tracking.scene_filter import SceneFilter


def result(image, masks):
    boxes = np.array([[0, 0, 40, 40, 0.9, 0]] * len(masks), np.float32)
    return Results(image, path="test", names={0: "person"}, boxes=boxes, masks=np.asarray(masks))


def test_screen_containment_preserves_person_extending_outside_screen():
    im = np.zeros((40, 40, 3), np.uint8)
    masks = np.zeros((2, 40, 40), np.uint8)
    masks[0, 2:18, 2:18] = 1
    masks[1, 2:38, 22:35] = 1
    filt = SceneFilter(
        [dict(name="tv", mode="screen", polygon=[[0, 0], [1, 0], [1, 0.5], [0, 0.5]], min_overlap=0.9)],
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
