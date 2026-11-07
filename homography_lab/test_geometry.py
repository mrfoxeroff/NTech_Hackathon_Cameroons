import copy

import cv2
import numpy as np
import pytest

from geometry import fit_calibration, inverse_project, project_observation, project_point, warp

H = np.array([[0.4, 0.04, 20], [-0.02, 0.5, 30], [0.0002, -0.0001, 1]])
CAMERA, PLAN = [640, 480], [600, 500]


def sample():
    fit = [[40, 40], [300, 40], [600, 40], [50, 230], [590, 230], [40, 430], [300, 430], [600, 430]]
    checks = [[170, 150], [420, 350], [350, 250]]
    return [
        {"camera": p, "plan": q.tolist(), "role": role}
        for role, rows in [("fit", fit), ("check", checks)]
        for p, q in zip(rows, warp(rows, H))
    ]


def test_independent_checks_recover_projective_mapping():
    c = fit_calibration(sample(), CAMERA, PLAN)
    assert c["status"] == "validated_on_check_points"
    assert c["quality"]["check_max_error_px"] < 1e-3
    p = [330, 200]
    assert np.allclose(project_point(p, c)["map_xy"], warp([p], H)[0], atol=1e-3)
    assert np.allclose(inverse_project(warp([p], H), c), [p], atol=1e-3)


def test_ransac_rejects_wrong_landmark_but_keeps_independent_checks():
    pairs = sample()
    pairs[3]["plan"] = [490, 450]
    c = fit_calibration(pairs, CAMERA, PLAN)
    assert c["quality"]["inlier_count"] == 7
    assert c["inliers"][3] is False
    assert c["quality"]["check_max_error_px"] < 0.001


def test_bad_check_does_not_move_fit_and_prevents_verified_status():
    pairs = sample()
    good = fit_calibration(pairs, CAMERA, PLAN)
    pairs[-1]["plan"] = [550, 420]
    bad = fit_calibration(pairs, CAMERA, PLAN)
    assert np.allclose(good["homography"], bad["homography"])
    assert bad["status"] == "draft"
    assert project_point([330, 200], bad)["status"] == "unverified_calibration"


def test_four_exact_points_are_only_a_draft():
    pairs = [sample()[i] for i in (0, 2, 5, 7)]
    c = fit_calibration(pairs, CAMERA, PLAN)
    assert c["status"] == "draft"
    assert c["quality"]["fit_rmse_px"] < 0.001


@pytest.mark.parametrize("pairs", [[], [{"camera": [x, 100], "plan": [x, 40]} for x in (30, 100, 300, 400)]])
def test_insufficient_or_collinear_points_are_rejected(pairs):
    with pytest.raises(ValueError):
        fit_calibration(pairs, CAMERA, PLAN)


def test_duplicate_control_is_not_an_independent_validation_point():
    pairs = sample()
    pairs[-1] = copy.deepcopy(pairs[0]) | {"role": "check"}
    with pytest.raises(ValueError, match="duplicate"):
        fit_calibration(pairs, CAMERA, PLAN)


def test_projection_rejects_extrapolation_by_default():
    c = fit_calibration(sample(), CAMERA, PLAN)
    assert project_point([10, 10], c)["status"] == "outside_calibrated_region"
    assert project_point([10, 10], c, allow_extrapolation=True)["status"] == "extrapolated"


def test_hidden_feet_do_not_become_map_coordinates():
    c = fit_calibration(sample(), CAMERA, PLAN)
    r = {"frame_index": 7, "footpoint_px": None, "bbox_xyxy": [20, 20, 100, 200]}
    output = project_observation(r, c)
    assert output["map_xy"] is None
    assert output["frame_index"] == 7 and "map_xy" not in r
    output = project_observation(r | {"footpoint_px": [100, 200], "footpoint_status": "unknown"}, c)
    assert output["map_xy"] is None


def test_observed_and_estimated_feet_retain_their_provenance():
    c = fit_calibration(sample(), CAMERA, PLAN)
    for status in ("observed", "estimated"):
        output = project_observation({"footpoint_px": [100, 200], "footpoint_status": status}, c)
        assert output["map_xy"] is not None and output["map_position_source"] == status


def test_known_lens_distortion_is_used_for_fit_and_projection():
    k = np.array([[420.0, 0, 320], [0, 420, 240], [0, 0, 1]])
    d = np.array([-0.12, 0.02, 0, 0, 0])
    pairs = sample()
    for pair in pairs:
        x, y = pair["camera"]
        point = np.array([[(x - 320) / 420, (y - 240) / 420, 1.0]])
        pair["camera"] = cv2.projectPoints(point, np.zeros(3), np.zeros(3), k, d)[0][0, 0].tolist()
    intr = {"camera_matrix": k.tolist(), "distortion": d.tolist(), "image_size": CAMERA}
    c = fit_calibration(pairs, CAMERA, PLAN, intrinsics=intr)
    assert c["status"] == "validated_on_check_points" and c["lens_correction_applied"]
    predicted = project_point(pairs[-1]["camera"], c)["map_xy"]
    assert np.allclose(predicted, pairs[-1]["plan"], atol=0.02)
    assert np.allclose(inverse_project([predicted], c)[0], pairs[-1]["camera"], atol=0.02)


def test_same_fit_cannot_validate_itself_by_changing_role():
    pairs = [p for p in sample() if p["role"] == "fit"]
    c = fit_calibration(pairs, CAMERA, PLAN)
    assert c["status"] == "draft" and c["quality"]["check_count"] == 0


def test_camera_and_resolution_mismatch_are_rejected():
    c = fit_calibration(sample(), CAMERA, PLAN) | {"camera_id": "one"}
    with pytest.raises(ValueError, match="camera"):
        project_observation({"camera_id": "two", "footpoint_px": [100, 200]}, c)
    with pytest.raises(ValueError, match="resolution"):
        project_observation({"frame_width": 1280, "frame_height": 960}, c)
