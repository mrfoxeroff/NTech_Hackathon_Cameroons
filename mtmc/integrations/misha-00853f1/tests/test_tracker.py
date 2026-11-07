"""Integration checks against the installed Ultralytics BYTETracker API."""

import numpy as np
import pytest
from ultralytics.engine.results import Boxes

from local_tracking.config import CameraConfig, Settings
from local_tracking.tracker import LocalTracker


@pytest.fixture
def settings(tmp_path):
    return Settings(
        recording_id="test",
        output_root=tmp_path / "out",
        model=tmp_path / "model.pt",
        cameras=[CameraConfig("cam_01", tmp_path / "video.mkv")],
        lost_track_sec=0.5,
    )


def boxes(*coordinates):
    data = np.asarray(
        [[*xyxy, 0.95, 0] for xyxy in coordinates],
        dtype=np.float32,
    ).reshape(-1, 6)
    return Boxes(data, orig_shape=(240, 320))


IMAGE = np.zeros((240, 320, 3), dtype=np.uint8)


def test_real_bytetracker_keeps_identity_through_empty_frame(settings):
    tracker = LocalTracker(settings, fps=10)
    first = tracker.update(boxes([10, 10, 50, 110]), IMAGE, 0.0)
    assert first == {0: 1}
    assert tracker.update(boxes(), IMAGE, 0.1) == {}
    assert tracker.update(boxes([12, 10, 52, 110]), IMAGE, 0.2) == {0: 1}


def test_separate_camera_trackers_export_independent_id_sequences(settings):
    first_camera = LocalTracker(settings, fps=10)
    second_camera = LocalTracker(settings, fps=10)
    assert first_camera.update(boxes([10, 10, 50, 110]), IMAGE, 0.0) == {0: 1}
    assert second_camera.update(boxes([150, 10, 190, 110]), IMAGE, 0.0) == {0: 1}
    assert first_camera.update(boxes([11, 10, 51, 110]), IMAGE, 0.1) == {0: 1}
    assert second_camera.update(boxes([151, 10, 191, 110]), IMAGE, 0.1) == {0: 1}


def test_elapsed_time_splits_identity_even_if_backend_reuses_track(settings):
    tracker = LocalTracker(settings, fps=10)
    assert tracker.update(boxes([10, 10, 50, 110]), IMAGE, 0.0) == {0: 1}
    assert tracker.update(boxes([10, 10, 50, 110]), IMAGE, 0.6) == {0: 2}


def test_detection_indices_survive_unconfirmed_tracks_and_reordering(settings):
    tracker = LocalTracker(settings, fps=10)
    assert tracker.update(boxes([10, 10, 50, 110]), IMAGE, 0.0) == {0: 1}
    # New B is detection 0, existing A moves to detection 1. B is not yet
    # confirmed, so tracker output order cannot be used as a detection index.
    second = tracker.update(boxes([150, 10, 190, 110], [12, 10, 52, 110]), IMAGE, 0.1)
    assert second == {1: 1}
    third = tracker.update(boxes([151, 10, 191, 110], [14, 10, 54, 110]), IMAGE, 0.2)
    assert third[1] == 1
    assert third[0] != third[1]


def test_buffer_is_in_processed_frame_steps(settings):
    settings.stride = 2
    tracker = LocalTracker(settings, fps=30)
    assert tracker.buffer_steps == 8  # ceil(0.5 * (30 / 2))
    assert tracker.backend.max_frames_lost == 8
