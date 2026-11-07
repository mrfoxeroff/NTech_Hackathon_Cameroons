import pytest

from local_tracking.config import CameraConfig


def test_clock_interpolates_and_extrapolates_monotonically(tmp_path):
    camera = CameraConfig(
        "cam",
        tmp_path / "video.mkv",
        0.25,
        [
            {"video_time_sec": 1.0, "timestamp_sec": 2.0},
            {"video_time_sec": 11.0, "timestamp_sec": 13.0},
            {"video_time_sec": 21.0, "timestamp_sec": 23.0},
        ],
    )
    assert camera.common_time(0.0) == pytest.approx(1.15)
    assert camera.common_time(6.0) == pytest.approx(7.75)
    assert camera.common_time(16.0) == pytest.approx(18.25)
    assert camera.common_time(22.0) == pytest.approx(24.25)


def test_offset_only_remains_available(tmp_path):
    assert CameraConfig("cam", tmp_path / "v", 3.5).common_time(10.0) == 13.5
