"""Exercise the real HTTP calibration/export flow with synthetic ground truth."""

import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from test_geometry import CAMERA, PLAN, sample


def test_http_fit_check_project_save_and_media_allowlist(tmp_path):
    state = {
        "status": "complete",
        "plan": {"size": PLAN, "sha256": "synthetic-plan"},
        "cameras": {
            "test_camera": {
                "url": "/media/test_camera.jpg",
                "size": CAMERA,
                "source_sha256": "synthetic-camera",
                "background": {"camera_motion_detected": False},
            }
        },
    }
    (tmp_path / "session.json").write_text(json.dumps(state), encoding="utf-8")
    process = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).with_name("app.py")),
            "serve",
            "--session",
            str(tmp_path),
            "--port",
            "0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        url = process.stdout.readline().strip()
        assert url.startswith("http://127.0.0.1:")

        def post(route, **extra):
            payload = json.dumps({"camera": "test_camera"} | extra).encode()
            request = urllib.request.Request(
                url + "/api/" + route, data=payload, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                return json.load(response)

        result = post("fit", pairs=sample())
        assert result["status"] == "validated_on_check_points"
        assert post("project", point=[330, 200])["map_xy"] is not None
        saved = post("save")
        stored = json.loads(Path(saved["file"]).read_text(encoding="utf-8"))
        assert stored["source_sha256"] == "synthetic-camera"
        assert stored["quality"]["check_count"] == 3
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(url + "/media/session.json", timeout=10)
        assert error.value.code == 404
        # A failed independent check never becomes an approved export.
        bad = sample()
        bad[-1]["plan"] = [550, 420]
        assert post("fit", pairs=bad)["status"] == "draft"
        assert post("save")["calibration"]["status"] == "draft"
    finally:
        process.terminate()
        process.communicate(timeout=10)


def test_cli_calibration_and_jsonl_projection_preserve_unknown_observations(tmp_path):
    app = str(Path(__file__).with_name("app.py"))
    source = tmp_path / "landmarks.json"
    source.write_text(json.dumps({"pairs": sample(), "camera_size": CAMERA, "plan_size": PLAN}))
    calibration = tmp_path / "calibration.json"
    subprocess.run(
        [sys.executable, app, "fit", "--input", str(source), "--output", str(calibration)],
        check=True,
        capture_output=True,
    )
    rows = [
        {"id": 1, "footpoint_px": [330, 200], "footpoint_status": "observed"},
        {"id": 2, "footpoint_px": None},
        {"id": 3, "footpoint_px": [10, 10]},
    ]
    observations, result = tmp_path / "input.jsonl", tmp_path / "output.jsonl"
    observations.write_text("\n".join(map(json.dumps, rows)))
    subprocess.run(
        [
            sys.executable,
            app,
            "project",
            "--calibration",
            str(calibration),
            "--input",
            str(observations),
            "--output",
            str(result),
        ],
        check=True,
        capture_output=True,
    )
    output = [json.loads(line) for line in result.read_text().splitlines()]
    assert [r["id"] for r in output] == [1, 2, 3]
    assert output[0]["map_xy"] is not None
    assert output[1]["map_xy"] is None and output[2]["map_xy"] is None


def test_neural_dashboard_preserves_negative_independent_review(tmp_path):
    manifest = {"status": "complete", "cameras": {"one": {"status": "automatic_candidate"}}}
    audit = {"cameras": {"one": {"localization_valid": False}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "evaluation.json").write_text(json.dumps(audit))
    process = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).with_name("neural_viewer.py")),
            "--output",
            str(tmp_path),
            "--port",
            "0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        url = process.stdout.readline().strip()
        with urllib.request.urlopen(url + "/api/result", timeout=10) as response:
            payload = json.load(response)
        assert payload["cameras"]["one"]["review"]["localization_valid"] is False
        assert json.loads((tmp_path / "manifest.json").read_text()) == manifest
    finally:
        process.terminate()
        process.communicate(timeout=10)
