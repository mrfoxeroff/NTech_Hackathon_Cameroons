from pathlib import Path

import pytest
import yaml

from local_tracking.config import CameraConfig, Settings, load_settings, safe_id


@pytest.fixture
def settings(tmp_path):
    video = tmp_path / "video.mkv"
    video.touch()
    return Settings(
        recording_id="office_01",
        output_root=tmp_path / "output",
        model=tmp_path / "weights.pt",
        cameras=[CameraConfig("camera_1", video)],
    )


def test_valid_configuration_does_not_require_downloaded_weights(settings):
    settings.validate()
    assert not settings.model.exists()


@pytest.mark.parametrize("value", ["../escape", "a/b", "a\\b", "", ".", "camera one", None, 5])
def test_identifiers_cannot_escape_output_directory(value):
    with pytest.raises(ValueError):
        safe_id(value)


def test_relatives_resolve_from_yaml_location_not_cwd(tmp_path, monkeypatch):
    source = tmp_path / "dataset" / "camera.mkv"
    source.parent.mkdir()
    source.touch()
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config = config_dir / "cameras.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "recording_id": "office_01",
                "output_root": "../results",
                "model": "../weights/model.pt",
                "cameras": [
                    {"camera_id": "camera_1", "video": "../dataset/camera.mkv", "time_offset_sec": -0.5}
                ],
            }
        ),
        encoding="utf-8",
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    actual = load_settings(config)
    assert actual.cameras[0].video == source.resolve()
    assert actual.model == (tmp_path / "weights" / "model.pt").resolve()
    assert actual.output_root == (tmp_path / "results").resolve()
    assert actual.cameras[0].time_offset_sec == -0.5


def test_duplicate_camera_ids_rejected(settings):
    settings.cameras.append(CameraConfig("camera_1", settings.cameras[0].video))
    with pytest.raises(ValueError, match="unique"):
        settings.validate()


def test_selected_cameras_do_not_require_unselected_recordings(tmp_path):
    (tmp_path / "present.mkv").touch()
    config = tmp_path / "cameras.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "recording_id": "test",
                "output_root": "output",
                "model": "model.pt",
                "cameras": [
                    {"camera_id": "camera_1", "video": "present.mkv"},
                    {"camera_id": "camera_2", "video": "missing.mkv"},
                ],
            }
        ),
        encoding="utf-8",
    )
    assert [camera.camera_id for camera in load_settings(config, camera_ids={"camera_1"}).cameras] == [
        "camera_1"
    ]
    with pytest.raises(ValueError, match="Video does not exist: .*missing.mkv"):
        load_settings(config)


def test_missing_video_rejected(settings):
    settings.cameras[0].video = Path("missing-source-video.mkv")
    with pytest.raises(ValueError, match="does not exist"):
        settings.validate()


def test_empty_camera_list_rejected(settings):
    settings.cameras = []
    with pytest.raises(ValueError, match="At least one"):
        settings.validate()


@pytest.mark.parametrize(
    "field,value",
    [
        ("stride", 0),
        ("stride", True),
        ("imgsz", -1),
        ("preview_max_width", 0),
        ("selected_crops_per_track", 0),
        ("confidence", 0),
        ("confidence", float("nan")),
        ("iou", 1.1),
        ("lost_track_sec", 0),
        ("lost_track_sec", float("inf")),
        ("crop_interval_sec", -1),
        ("crop_interval_sec", float("nan")),
    ],
)
def test_invalid_processing_parameters_rejected(settings, field, value):
    setattr(settings, field, value)
    with pytest.raises(ValueError):
        settings.validate()


def test_nonfinite_camera_offset_rejected(settings):
    settings.cameras[0].time_offset_sec = float("nan")
    with pytest.raises(ValueError, match="offset"):
        settings.validate()


@pytest.mark.parametrize(
    "confidence,tracker",
    [
        (0.2, {"track_low_thresh": 0.1}),
        (0.1, {"track_low_thresh": 0.4, "track_high_thresh": 0.3}),
        (0.1, {"track_high_thresh": 0.4, "new_track_thresh": 0.3}),
        (0.1, {"match_thresh": 1.1}),
        (0.1, {"track_low_thresh": float("nan")}),
    ],
)
def test_thresholds_preserve_low_confidence_recovery(settings, confidence, tracker):
    settings.confidence = confidence
    settings.tracker = tracker
    with pytest.raises(ValueError):
        settings.validate()


def test_unknown_tracker_option_rejected(settings):
    settings.tracker = {"track_bufffer": 30}
    with pytest.raises(ValueError, match="Unknown tracker"):
        settings.validate()
