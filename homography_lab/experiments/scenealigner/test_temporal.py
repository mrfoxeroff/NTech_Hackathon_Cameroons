import copy
import json
import sys

import numpy as np
import pytest
from PIL import Image
from temporal_evaluate import accuracy_pass, transferred_reference
from temporal_frames import choose_stress
from temporal_registration import fit_registration


def test_foreground_selection_keeps_brief_events_without_duplicate_frames():
    probes = [
        {"actual_seconds": time, "foreground_fraction": score}
        for time, score in [(60.0, 0.9), (65.0, 0.8), (70.0, 0.7), (200.0, 0.6), (300.0, 0.1)]
    ]
    selected = choose_stress(probes, [60.0, 180.0])
    assert [item["actual_seconds"] for item in selected] == [65.0, 200.0]
    assert selected[1]["actual_seconds"] - selected[0]["actual_seconds"] >= 60


def test_subpixel_camera_shift_is_recovered_with_held_out_features():
    rng = np.random.default_rng(3)
    source = rng.uniform([10, 10], [630, 470], (200, 2)).astype(np.float32)
    target = source + [0.04, -0.03]
    target[:25] = rng.uniform([0, 0], [640, 480], (25, 2))
    result = fit_registration(source, target.astype(np.float32), (480, 640), 0.5)
    assert result["accepted"]
    assert np.allclose(np.asarray(result["matrix"])[:, 2], [0.08, -0.06], atol=0.001)
    assert result["holdout_fraction_within_2px_reduced"] > 0.7


def test_camera_movement_invalidates_frozen_calibration_stability_claim():
    rng = np.random.default_rng(1)
    source = rng.uniform([10, 10], [630, 470], (200, 2)).astype(np.float32)
    result = fit_registration(source, source + [50.0, 0.0], (480, 640), 0.5)
    assert not result["accepted"]
    assert result["relative_motion"] > 0.003


def test_reference_transfer_does_not_refit_or_mutate_labels():
    reference = {"camera_points": [[10.0, 20.0], [30.0, 40.0]], "plan_points": [[1.0, 2.0], [3.0, 4.0]]}
    before = copy.deepcopy(reference)
    updated = transferred_reference(reference, [[1, 0, 2], [0, 1, -3]])
    assert reference == before
    assert updated["camera_points"] == [[12.0, 17.0], [32.0, 37.0]]
    assert updated["plan_points"] == reference["plan_points"]


@pytest.mark.parametrize(
    "median,p95,invalid,passes",
    [(7.4, 19.7, 0, True), (7.4, 21.3, 0, False), (7.4, 19.7, 1, False), (None, None, 1, False)],
)
def test_temporal_criteria_include_tail_errors_and_unprojectable_points(median, p95, invalid, passes):
    assert (
        accuracy_pass(
            {"median_error_px": median, "p95_error_px": p95, "unprojectable_reference_inliers": invalid}
        )
        == passes
    )


def test_failures_controls_and_oracle_do_not_inflate_temporal_accuracy(tmp_path, monkeypatch):
    from temporal_evaluate import main

    image = tmp_path / "frame.png"
    Image.new("L", (200, 200), 127).save(image)

    def write(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    points = [[20, 20], [100, 20], [100, 100], [20, 100], [60, 60]]
    reference = write(
        "reference.json",
        {
            "camera_size": [200, 200],
            "plan_size": [300, 300],
            "camera_points": points,
            "plan_points": points,
            "inliers": [True] * 5,
        },
    )
    base = {"camera_size": [200, 200], "plan_size": [300, 300], "homography": np.eye(3).tolist()}
    frozen = write("frozen.json", base)
    wrong = write("wrong.json", base | {"homography": [[1, 0, 50], [0, 1, 0], [0, 0, 1]]})
    cases = [
        {
            "id": "case_01",
            "kind": "uniform",
            "actual_seconds": 10,
            "status": "complete",
            "prediction": str(wrong),
            "image": str(image),
        },
        {
            "id": "case_02",
            "kind": "foreground_stress",
            "actual_seconds": 20,
            "status": "failed",
            "error": "no floor",
            "image": str(image),
        },
        {"id": "control", "kind": "control_cached_search", "status": "complete", "prediction": str(frozen)},
    ]
    protocol = write("inference/protocol.json", {"status": "complete", "cases": cases})
    write(
        "inference/case_01/search/search.json",
        {
            "candidates": [
                {"density_to_original": np.eye(3).tolist()},
                {"density_to_original": [[1, 0, -50], [0, 1, 0], [0, 0, 1]]},
            ]
        },
    )
    write(
        "inference/control/search/search.json", {"candidates": [{"density_to_original": np.eye(3).tolist()}]}
    )
    samples = write("samples.json", {"cases": cases[:2]})
    registration = write(
        "registration.json",
        {
            "cases": {case["id"]: {"accepted": True, "matrix": [[1, 0, 0], [0, 1, 0]]} for case in cases[:2]},
            "probes": [],
        },
    )
    output = tmp_path / "evaluation.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--inference",
            str(protocol.parent),
            "--samples",
            str(samples),
            "--registration",
            str(registration),
            "--reference",
            str(reference),
            "--frozen-calibration",
            str(frozen),
            "--background",
            str(image),
            "--output",
            str(output),
        ],
    )
    main()
    result = json.loads(output.read_text())
    assert result["summary"]["new_frames"] == 2
    assert result["summary"]["fresh_predictions"] == 1
    assert result["summary"]["fresh_pass_count"] == 0
    assert result["summary"]["oracle_any_pass_count_diagnostic"] == 1
    assert len(result["controls"]) == 1
