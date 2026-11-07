"""Integration through decoder, real ByteTrack and exporters without downloading weights."""

import json
from fractions import Fraction

import av
import numpy as np
import pytest

from local_tracking.config import CameraConfig, Settings
from local_tracking.pipeline import process_camera


def fixture_settings(tmp_path):
    path = tmp_path / "source.mkv"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("ffv1", rate=10)
        stream.width, stream.height, stream.pix_fmt = 64, 64, "bgr0"
        for i in range(4):
            frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), 120, np.uint8), format="bgr24")
            frame.pts, frame.time_base = i, Fraction(1, 10)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    weights = tmp_path / "fake.pt"
    weights.write_bytes(b"test fixture, not neural weights")
    camera = CameraConfig("cam_1", path, 2.0)
    return Settings("test", tmp_path / "output", weights, [camera], save_preview=True), camera


class FakeSegmenter:
    def __init__(self, fail=False):
        self.step = 0
        self.fail = fail

    def predict(self, image, **kwargs):
        import torch
        from ultralytics.engine.results import Results

        step = self.step
        self.step += 1
        if self.fail and step == 1:
            raise RuntimeError("injected inference failure")
        if step == 2:
            boxes = torch.zeros((0, 6))
            masks = None
        else:
            boxes = torch.tensor([[10.0, 5.0, 30.0, 60.0, 0.95, 0.0]])
            masks = torch.zeros((1, 64, 64))
            masks[0, 5:60, 10:30] = 1
        return [Results(image, path="fixture", names={0: "person"}, boxes=boxes, masks=masks)]


def test_complete_pipeline_exports_real_observations_and_gaps(tmp_path):
    settings, camera = fixture_settings(tmp_path)
    metrics = process_camera(settings, camera, FakeSegmenter(), 0, {"device": "cpu"})
    root = settings.output_root / "test" / "cam_1"
    manifest = json.loads((root / "manifest.json").read_text())
    frames = [json.loads(line) for line in (root / "frames.jsonl").read_text().splitlines()]
    tracks = json.loads((root / "local_tracks.json").read_text())["tracks"]
    assert manifest["status"] == "complete"
    assert metrics["processed_frames"] == 4 and metrics["empty_frames"] == 1
    assert frames[0]["timestamp_sec"] == 2.0
    assert frames[2]["observations"] == []  # No Kalman prediction as a detection.
    assert frames[0]["observations"][0]["local_track_id"] == frames[3]["observations"][0]["local_track_id"]
    assert len(tracks) == 1 and len(tracks[0]["visibility_intervals"]) == 2
    obs = frames[0]["observations"][0]
    assert obs["footpoint_px"] == [20.0, 60.0]
    assert (root / obs["mask_ref"]).is_file() and (root / obs["crop_ref"]).is_file()
    with av.open(str(root / "preview.mp4")) as container:
        assert len(list(container.decode(video=0))) >= 3
    from local_tracking.validate import validate_output

    assert validate_output(root)["valid"] is True
    with pytest.raises(FileExistsError):
        process_camera(settings, camera, FakeSegmenter(), 0, {"device": "cpu"})


def test_pipeline_preserves_media_time_and_aligns_to_clock(tmp_path):
    from local_tracking.validate import validate_output

    settings, camera = fixture_settings(tmp_path)
    camera.time_offset_sec = 0.0
    camera.clock_anchors = [
        {"video_time_sec": 0.0, "timestamp_sec": 4.0},
        {"video_time_sec": 1.0, "timestamp_sec": 5.1},
    ]
    process_camera(settings, camera, FakeSegmenter(), 0, {"device": "cpu"})
    root = settings.output_root / "test" / "cam_1"
    frames = [json.loads(line) for line in (root / "frames.jsonl").read_text().splitlines()]
    assert frames[-1]["video_time_sec"] == pytest.approx(0.3)
    assert frames[-1]["timestamp_sec"] == pytest.approx(4.33)
    assert frames[-1]["media_time_sec"] == pytest.approx(0.3)
    assert validate_output(root)["valid"]


def test_failed_run_is_marked_and_partial_outputs_remain_readable(tmp_path):
    settings, camera = fixture_settings(tmp_path)
    with pytest.raises(RuntimeError, match="injected inference failure"):
        process_camera(settings, camera, FakeSegmenter(fail=True), 0, {"device": "cpu"})
    root = settings.output_root / "test" / "cam_1"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"]["type"] == "RuntimeError"
    assert len((root / "frames.jsonl").read_text().splitlines()) == 1
    assert json.loads((root / "metrics.json").read_text())["processed_frames"] == 1


def test_hash_failure_leaves_failed_manifest_and_preserves_original_error(tmp_path, monkeypatch):
    settings, camera = fixture_settings(tmp_path)
    error = OSError("injected hash read failure")

    def broken_hash(path):
        root = settings.output_root / "test" / "cam_1"
        assert json.loads((root / "manifest.json").read_text())["status"] == "running"
        raise error

    monkeypatch.setattr("local_tracking.pipeline.file_hash", broken_hash)
    with pytest.raises(OSError) as caught:
        process_camera(settings, camera, FakeSegmenter(), 0, {"device": "cpu"})
    assert caught.value is error
    root = settings.output_root / "test" / "cam_1"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"]["message"] == "injected hash read failure"
    assert json.loads((root / "metrics.json").read_text())["processed_frames"] == 0


@pytest.mark.parametrize("inference_fails", [False, True])
def test_preview_finalize_failure_marks_failed_and_preserves_processing_error(
    tmp_path,
    monkeypatch,
    inference_fails,
):
    from local_tracking.artifacts import PreviewWriter

    settings, camera = fixture_settings(tmp_path)
    original_close = PreviewWriter.close

    def broken_close(self):
        original_close(self)  # Always release the real encoder before failing.
        raise OSError("injected preview finalization failure")

    monkeypatch.setattr(PreviewWriter, "close", broken_close)
    expected = RuntimeError if inference_fails else OSError
    message = "injected inference failure" if inference_fails else "injected preview finalization failure"
    with pytest.raises(expected, match=message):
        process_camera(settings, camera, FakeSegmenter(fail=inference_fails), 0, {"device": "cpu"})
    root = settings.output_root / "test" / "cam_1"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"]["message"] == message
    assert manifest["finalization_errors"] == [
        {
            "type": "OSError",
            "message": "injected preview finalization failure",
        }
    ]
    assert (root / "local_tracks.json").is_file()
    assert (root / "metrics.json").is_file()
