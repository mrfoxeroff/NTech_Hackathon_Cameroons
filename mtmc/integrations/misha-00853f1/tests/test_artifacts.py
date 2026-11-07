from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from local_tracking.artifacts import ObservationWriter, TrackSummary, observation_geometry


class Array:
    """Minimal tensor adapter: tests exercise exporting, without loading a neural model."""

    def __init__(self, value):
        self.value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self):
        return self.value


def result(boxes, masks, scores=None):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    return SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=Array(boxes),
            conf=Array(scores if scores is not None else np.full(len(boxes), 0.9)),
        ),
        masks=None if masks is None else SimpleNamespace(data=Array(masks)),
    )


def frame(index=0, time=0.0, image=None):
    if image is None:
        image = np.zeros((12, 16, 3), dtype=np.uint8)
        image[:, :, :] = [25, 90, 170]
    return SimpleNamespace(index=index, timestamp_sec=time, image=image)


def read_image(path, flags=cv2.IMREAD_UNCHANGED):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)


def test_geometry_clips_to_source_and_keeps_fractional_footpoint():
    bbox, roi, foot, borders = observation_geometry([-3.4, 2.2, 18.0, 12.7], 16, 12)
    assert bbox == [0.0, 2.2, 16, 12]
    assert roi == [0, 2, 16, 12]
    assert foot == [8, 12]
    assert borders == {"left": True, "top": False, "right": True, "bottom": True}
    bbox, roi, foot, borders = observation_geometry([2.2, 1.2, 12.1, 10.4], 16, 12)
    assert roi == [2, 1, 13, 11]
    assert foot == pytest.approx([7.15, 10.4])
    assert not any(borders.values())


@pytest.mark.parametrize(
    "bbox",
    [
        [1, 2, 1, 5],
        [5, 2, 3, 8],
        [18, 0, 20, 5],
        [0, 0, float("nan"), 5],
        [0, 0, 5, float("inf")],
        [0, 1, 2],
    ],
)
def test_geometry_rejects_invalid_or_empty_boxes(bbox):
    with pytest.raises(ValueError):
        observation_geometry(bbox, 16, 12)


def test_mask_roi_reconstructs_source_coordinates_and_crop_is_clean(tmp_path):
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    image[:] = [25, 90, 170]
    image[5:8, 6:10] = [200, 40, 30]
    clean = image.copy()
    mask = np.zeros((12, 16), dtype=np.uint8)
    mask[3:9, 4:11] = 1
    writer = ObservationWriter(tmp_path, crop_interval_sec=0.5, save_masks=True)
    observations = writer.build(
        result([[2.2, 1.2, 12.1, 10.4]], [mask]),
        {0: 7},
        frame(image=image),
        "recording",
        "camera",
    )
    assert len(observations) == 1
    obs = observations[0]
    assert obs["local_track_id"] == 7
    assert obs["observation_id"] == "recording:camera:0:0"
    assert obs["mask_origin_px"] == [2, 1]
    assert obs["mask_size"] == [11, 10]
    roi_mask = read_image(tmp_path / obs["mask_ref"])
    assert roi_mask.shape == (10, 11)
    assert set(np.unique(roi_mask)) == {0, 255}
    restored = np.zeros_like(mask)
    restored[1:11, 2:13] = roi_mask
    np.testing.assert_array_equal(restored, mask * 255)
    points = np.asarray(obs["silhouette_polygons"][0])
    assert points.min(axis=0).tolist() == [4, 3]
    assert points.max(axis=0).tolist() == [10, 8]
    np.testing.assert_array_equal(image, clean)
    _, encoded = cv2.imencode(".jpg", clean[1:11, 2:13])
    expected = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    np.testing.assert_array_equal(read_image(tmp_path / obs["crop_ref"]), expected)
    assert obs["crop_quality"]["width"] == 11
    assert obs["crop_quality"]["height"] == 10
    assert obs["crop_quality"]["truncated"] is False


def test_crop_interval_is_per_track_and_untracked_detections_remain_separate(tmp_path):
    mask = np.ones((12, 16), np.uint8)
    detections = result([[2, 2, 10, 10], [3, 2, 12, 10]], [mask, mask])
    writer = ObservationWriter(tmp_path, crop_interval_sec=0.5, save_masks=False)
    first = writer.build(detections, {0: 7}, frame(0, 0), "r", "c")
    second = writer.build(detections, {0: 7}, frame(1, 0.2), "r", "c")
    third = writer.build(detections, {0: 7}, frame(2, 0.5), "r", "c")
    assert first[0]["crop_ref"] and third[0]["crop_ref"]
    assert second[0]["crop_ref"] is None
    assert second[0]["crop_quality"] is None
    assert all(batch[1]["local_track_id"] is None for batch in [first, second, third])
    assert len({batch[1]["crop_ref"] for batch in [first, second, third]}) == 3
    assert first[0]["mask_ref"] is None
    assert first[0]["silhouette_polygons"]
    assert not (tmp_path / "masks").exists()


def test_empty_detections_produce_empty_observations(tmp_path):
    writer = ObservationWriter(tmp_path, 0.5, True)
    assert writer.build(result([], None), {}, frame(), "r", "c") == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("masks", [None, [], [np.zeros((6, 8), np.uint8)]])
def test_mask_mismatch_is_not_silently_resized_or_misassigned(tmp_path, masks):
    writer = ObservationWriter(tmp_path, 0.5, True)
    with pytest.raises(RuntimeError):
        writer.build(result([[2, 2, 10, 10]], masks), {0: 1}, frame(), "r", "c")


def observation(local_id, name, score, with_crop=True):
    return {
        "local_track_id": local_id,
        "observation_id": name,
        "crop_ref": f"crops/{name}.jpg" if with_crop else None,
        "crop_quality": {"selection_score": score} if with_crop else None,
    }


def test_summary_preserves_gaps_excludes_null_and_selects_top_k():
    summary = TrackSummary(selected_count=2)
    summary.add(observation(7, "a", 10), 1.0, 0)
    summary.add(observation(7, "b", 30), 1.1, 1)
    summary.add(observation(None, "untracked", 100), 1.2, 2)
    summary.add(observation(7, "c", 20), 1.3, 3)
    summary.add(observation(7, "d", 0, with_crop=False), 1.4, 4)
    tracks = summary.export()
    assert len(tracks) == 1
    track = tracks[0]
    assert track["observation_count"] == 4
    assert track["first_seen_sec"] == 1.0
    assert track["last_seen_sec"] == 1.4
    assert track["visibility_intervals"] == [
        {"start_sec": 1.0, "end_sec": 1.1},
        {"start_sec": 1.3, "end_sec": 1.4},
    ]
    assert [c["observation_id"] for c in track["selected_crops"]] == ["b", "c"]
    assert not any(key.startswith("_") for key in track)


def test_summary_uses_processed_steps_not_original_frame_stride():
    summary = TrackSummary(1)
    summary.add(observation(1, "frame_0", 1), 0.0, 0)
    summary.add(observation(1, "frame_3", 1), 0.1, 1)
    assert summary.export()[0]["visibility_intervals"] == [{"start_sec": 0.0, "end_sec": 0.1}]


def test_source_gap_splits_visibility_even_when_id_survives():
    summary = TrackSummary(1)
    summary.add(observation(1, "before_gap", 1), 0.0, 0)
    summary.add(observation(1, "after_gap", 1), 0.8, 1, source_gap=True)
    assert summary.export()[0]["visibility_intervals"] == [
        {"start_sec": 0.0, "end_sec": 0.0},
        {"start_sec": 0.8, "end_sec": 0.8},
    ]
