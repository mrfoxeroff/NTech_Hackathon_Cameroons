from fractions import Fraction
from types import SimpleNamespace

import av
import numpy as np
import pytest

from local_tracking.video import iter_video, probe_video


@pytest.fixture
def sample_video(tmp_path):
    path = tmp_path / "sample.mkv"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("ffv1", rate=10)
        stream.width, stream.height = 64, 48
        stream.pix_fmt = "yuv420p"
        for i in range(8):
            image = np.full((48, 64, 3), i * 20, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts = i + 100
            frame.time_base = Fraction(1, 10)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def test_decode_normalizes_nonzero_pts_and_keeps_bgr_shape(sample_video):
    info = probe_video(sample_video)
    assert (info.width, info.height, info.fps) == (64, 48, 10)
    assert info.first_frame_time_sec == pytest.approx(10.0)
    frames = list(iter_video(sample_video, time_offset_sec=2.5))
    assert len(frames) == 8
    assert [f.index for f in frames] == list(range(8))
    assert [f.video_time_sec for f in frames] == pytest.approx([i / 10 for i in range(8)])
    assert frames[0].timestamp_sec == pytest.approx(2.5)
    assert [f.media_time_sec for f in frames] == pytest.approx([10 + i / 10 for i in range(8)])
    assert all(f.timestamp_source == "pts" for f in frames)
    assert frames[0].image.shape == (48, 64, 3)
    assert frames[0].image.dtype == np.uint8


def test_half_open_bounds_and_stride_use_source_index(sample_video):
    frames = list(iter_video(sample_video, start_sec=0.15, end_sec=0.6, stride=2))
    assert [f.index for f in frames] == [2, 4]
    assert [f.video_time_sec for f in frames] == pytest.approx([0.2, 0.4])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stride": 0},
        {"stride": True},
        {"stride": 1.5},
        {"start_sec": -1},
        {"start_sec": float("nan")},
        {"end_sec": 0},
        {"end_sec": float("inf")},
        {"time_offset_sec": float("nan")},
    ],
)
def test_invalid_arguments(sample_video, kwargs):
    with pytest.raises(ValueError):
        list(iter_video(sample_video, **kwargs))


def _fake_open(monkeypatch, path, frames, rate=10, has_video=True):
    stream = SimpleNamespace(
        average_rate=rate,
        guessed_rate=rate,
        base_rate=rate,
        time_base=Fraction(1, 10),
    )

    class Container:
        streams = SimpleNamespace(video=[stream] if has_video else [])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def decode(self, selected_stream):
            return iter(frames)

    monkeypatch.setattr("local_tracking.video.av.open", lambda _: Container())
    path.touch()


def _fake_frame(pts):
    return SimpleNamespace(
        pts=pts,
        time_base=Fraction(1, 10),
        to_ndarray=lambda **_: np.zeros((2, 2, 3), dtype=np.uint8),
    )


def test_missing_pts_is_labelled_and_preserves_timeline(tmp_path, monkeypatch):
    path = tmp_path / "fake.mkv"
    _fake_open(monkeypatch, path, [_fake_frame(None), _fake_frame(101), _fake_frame(None)])
    frames = list(iter_video(path))
    assert [f.video_time_sec for f in frames] == pytest.approx([0, 0.1, 0.2])
    assert [f.timestamp_source for f in frames] == ["index_fps", "pts", "index_fps"]
    assert [f.media_time_sec for f in frames] == [None, 10.1, None]


@pytest.mark.parametrize("pts", [[0, 1, 1], [0, 2, 1]])
def test_rejects_duplicate_and_backwards_timestamps_even_if_skipped(tmp_path, monkeypatch, pts):
    path = tmp_path / "fake.mkv"
    _fake_open(monkeypatch, path, [_fake_frame(p) for p in pts])
    with pytest.raises(ValueError, match="Non-increasing"):
        list(iter_video(path, stride=3))


def test_rejects_empty_video(tmp_path, monkeypatch):
    path = tmp_path / "fake.mkv"
    _fake_open(monkeypatch, path, [])
    with pytest.raises(ValueError, match="no decodable"):
        list(iter_video(path))


def test_rejects_missing_video_stream(tmp_path, monkeypatch):
    path = tmp_path / "fake.mkv"
    _fake_open(monkeypatch, path, [], has_video=False)
    with pytest.raises(ValueError, match="no video stream"):
        list(iter_video(path))


def test_rejects_invalid_fps(tmp_path, monkeypatch):
    path = tmp_path / "fake.mkv"
    _fake_open(monkeypatch, path, [_fake_frame(0)], rate=0)
    with pytest.raises(ValueError, match="frame rate"):
        list(iter_video(path))


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        probe_video(tmp_path / "missing.mkv")
