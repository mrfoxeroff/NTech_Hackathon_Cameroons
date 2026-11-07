import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from motion_candidates import validate_bank
from motion_config import LAB, OFFICE, load_config
from motion_export import exportable_calibrations


def config_file(tmp_path):
    folder = tmp_path / "configuration"
    folder.mkdir()
    data = {
        "sample_fps": 4,
        "clock_templates": "clock",
        "cameras": {"cam_1": {"video": "videos/camera_new_1.mkv", "floor_mask": "floor.png"}},
        "windows": [
            {"id": "train", "start": 0, "end": 5, "split": "train"},
            {"id": "test", "start": 10, "end": 15, "split": "holdout"},
        ],
    }
    path = folder / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path, data


def test_configuration_paths_do_not_depend_on_current_directory(tmp_path, monkeypatch):
    path, _ = config_file(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    config = load_config(path)
    assert Path(config["cameras"]["cam_1"]["video"]) == path.parent / "videos/camera_new_1.mkv"
    assert Path(config["clock_templates"]) == path.parent / "clock"


def test_video_root_can_be_changed_without_changing_model_assets(tmp_path):
    path, _ = config_file(tmp_path)
    root = tmp_path / "recordings"
    config = load_config(path, video_root=root)
    assert Path(config["cameras"]["cam_1"]["video"]) == root / "camera_new_1.mkv"
    assert Path(config["cameras"]["cam_1"]["floor_mask"]) == path.parent / "floor.png"


def test_overlapping_training_and_holdout_are_rejected(tmp_path):
    path, data = config_file(tmp_path)
    data["windows"][1]["start"] = 3
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        load_config(path)


def test_window_names_cannot_escape_the_output_directory(tmp_path):
    path, data = config_file(tmp_path)
    data["windows"][0]["id"] = "../../other"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="safe directory"):
        load_config(path)


def test_bundled_bank_is_automatic_and_rejects_manual_or_singular_inputs():
    report = json.loads((OFFICE / "automatic-candidates.json").read_text(encoding="utf-8"))
    validate_bank(report)
    modified = deepcopy(report)
    modified["manual_plan_landmarks_used"] = True
    with pytest.raises(ValueError, match="automatic"):
        validate_bank(modified)
    modified = deepcopy(report)
    next(iter(modified["cameras"].values()))["candidates"][0]["calibration"]["homography"] = np.zeros(
        (3, 3)
    ).tolist()
    with pytest.raises(ValueError, match="homography"):
        validate_bank(modified)


def test_legacy_diagnostic_calibrations_cannot_be_exported():
    result = {
        "calibrations": {"cam_1": {"homography": np.eye(3).tolist()}},
        "candidate_calibrations": {"cam_1": {}},
        "absolute_location_verified": False,
        "automatic_deployment_ready": False,
    }
    assert exportable_calibrations(result) == {}


def test_ready_flag_cannot_override_rejected_components():
    result = {
        "absolute_location_verified": True,
        "automatic_deployment_ready": True,
        "components": [{"status": "abstained"}],
        "accepted_calibrations": {},
    }
    with pytest.raises(ValueError, match="conflicts"):
        exportable_calibrations(result)


def test_verified_supported_calibration_can_be_exported():
    accepted = {"cam_1": {"homography": np.eye(3).tolist()}}
    result = {
        "absolute_location_verified": True,
        "automatic_deployment_ready": True,
        "components": [{"status": "accepted", "cameras": ["cam_1"]}],
        "accepted_calibrations": accepted,
    }
    assert exportable_calibrations(result) == accepted
    result["components"] = []
    with pytest.raises(ValueError, match="component"):
        exportable_calibrations(result)


def test_preflight_reports_missing_inputs_without_creating_output(tmp_path):
    config, _ = config_file(tmp_path)
    output = tmp_path / "not-created"
    run = subprocess.run(
        [
            sys.executable,
            str(LAB / "experiments/scenealigner/motion_run.py"),
            "--config",
            str(config),
            "--output",
            str(output),
            "--weights-root",
            str(tmp_path / "no-weights"),
            "--check-inputs",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 1, run.stderr
    report = json.loads(run.stdout)
    assert not report["ready"] and report["missing_files"]
    assert not output.exists()


def test_clone_demo_runs_elsewhere_without_neural_models(tmp_path):
    output = tmp_path / "result"
    subprocess.run(
        [sys.executable, str(LAB / "demo.py"), "--output", str(output)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    assert not protocol["neural_models_loaded"] and not protocol["video_inference_executed"]
    assert protocol["exported_cameras"] == []
    result = json.loads((output / "joint.json").read_text(encoding="utf-8"))
    assert result["accepted_calibrations"] == {}
    assert result["candidate_calibrations"]
