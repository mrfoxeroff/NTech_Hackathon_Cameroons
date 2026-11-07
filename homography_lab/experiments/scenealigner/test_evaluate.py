import copy

import numpy as np
import pytest
from evaluate import evaluate_camera


def examples():
    h = np.array([[0.4, 0, 20], [0, 0.5, 30], [0, 0, 1]])
    src = np.array([[10, 10], [100, 10], [100, 100], [10, 100], [50, 40]], dtype=float)
    dst = (np.column_stack((src, np.ones(len(src)))) @ h.T)[:, :2]
    sizes = {"camera_size": [200, 200], "plan_size": [300, 300]}
    return sizes | {"homography": h.tolist()}, sizes | {
        "camera_points": src.tolist(),
        "plan_points": dst.tolist(),
        "inliers": [True] * 5,
    }


def test_evaluation_does_not_change_or_fit_prediction():
    prediction, reference = examples()
    before = copy.deepcopy(prediction)
    reference["plan_points"][0][0] += 50
    result = evaluate_camera(prediction, reference)
    assert prediction == before
    assert result["errors_px"][0] == pytest.approx(50)
    assert result["fraction_within_10px"] == 0.8


def test_reference_outliers_are_reported_explicitly():
    prediction, reference = examples()
    reference["plan_points"][0][0] += 50
    reference["inliers"][0] = False
    result = evaluate_camera(prediction, reference)
    assert result["excluded_reference_outliers"] == 1
    assert result["median_error_px"] == 0
    assert result["errors_px"][0] == 50


def test_mismatched_sizes_are_rejected():
    prediction, reference = examples()
    prediction["plan_size"] = [600, 600]
    with pytest.raises(ValueError, match="resolution"):
        evaluate_camera(prediction, reference)


def test_missing_predictions_are_reported_instead_of_silently_dropped(tmp_path, monkeypatch):
    import json
    import sys

    from evaluate import main

    _, reference = examples()
    refs = tmp_path / "refs"
    preds = tmp_path / "preds"
    refs.mkdir()
    preds.mkdir()
    (refs / "cam_1_calibration.json").write_text(json.dumps(reference))
    output = tmp_path / "evaluation.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate", "--reference", str(refs), "--predictions", str(preds), "--output", str(output)],
    )
    main()
    result = json.loads(output.read_text())
    assert result["missing_predictions"] == ["cam_1"]
    assert result["coverage_ratio"] == 0
    assert result["status"] == "no_predictions"
