import numpy as np
from analytic_place_search import select_place
from door_landmarks import door_floor_anchor
from floor_ray_geometry import floor_homography, project_floor, ray_blocking_fraction
from lens_geometry import project_calibration
from structural_matching import door_scale_error


def identity_lens(k=0):
    return {
        "model": "simple_divisional",
        "K": np.eye(3),
        "new_K": np.eye(3),
        "k1": k,
        "size": [640, 480],
        "rectified_size": [640, 480],
    }


def test_floor_projection_matches_independent_ray_plane_intersection():
    gravity = np.array([0.0, 2.0, 2.0])
    before = gravity.copy()
    h = floor_homography(np.eye(3), gravity)
    points = np.array([[0.2, 1.0], [-0.3, 0.5], [0, 0]])
    floor, valid = project_floor(points, identity_lens(), h)
    g = gravity / np.linalg.norm(gravity)
    rays = np.column_stack((points, np.ones(3)))
    xyz = rays / (rays @ g)[:, None]
    forward = np.cross([1.0, 0, 0], g)
    forward /= np.linalg.norm(forward)
    expected = np.column_stack((xyz[:, 0], -xyz @ forward))
    assert valid.all() and np.allclose(floor, expected)
    assert np.array_equal(gravity, before)


def test_horizon_and_behind_camera_are_not_floor_observations():
    h = floor_homography(np.eye(3), [0.0, 1.0, 0.0])
    floor, valid = project_floor(np.array([[0.0, 1.0], [0, 0], [0, -1.0], [100, 1]]), identity_lens(), h)
    assert valid.tolist() == [True, False, False, False]
    assert np.isnan(floor[~valid]).all()


def test_free_space_rays_reject_wall_crossings_but_allow_open_doorway():
    walls = np.zeros((100, 100), np.uint8)
    walls[:, 50] = 255
    origin, ends = np.array([20.0, 50.0]), np.array([[80.0, 50.0], [40, 50]])
    assert ray_blocking_fraction(np.eye(3), origin, ends, walls) == 0.5
    walls[45:56, 50] = 0
    assert ray_blocking_fraction(np.eye(3), origin, ends, walls) == 0.0
    assert ray_blocking_fraction(np.eye(3), origin, np.array([[120.0, 50.0]]), walls) == 1.0


def test_long_visibility_ray_does_not_skip_thin_wall():
    walls = np.zeros((100, 1000), np.uint8)
    walls[:, 123] = 255
    assert ray_blocking_fraction(np.eye(3), [5.0, 50.0], np.array([[900.0, 50.0]]), walls) == 1.0


def test_observed_ideal_rays_can_be_outside_rectified_canvas():
    lens = {
        "model": "simple_divisional",
        "k1": -0.7,
        "K": [[700, 0, 320], [0, 700, 240], [0, 0, 1]],
        "new_K": [[700, 0, 320], [0, 700, 240], [0, 0, 1]],
        "size": [640, 480],
        "rectified_size": [640, 480],
    }
    prediction = {
        "lens": lens,
        "lens_correction_applied": True,
        "homography_input_space": "ideal_pixels",
        "homography": np.eye(3),
    }
    _, valid = project_calibration(np.array([[639.0, 479.0], [640, 400]]), prediction)
    assert valid.tolist() == [True, False]
    prediction.update(floor_ray_denominator=[0.0, 1.0, -300.0])
    _, valid = project_calibration(np.array([[320.0, 200.0], [320, 400.0]]), prediction)
    assert valid.tolist() == [False, True]


def test_distinct_places_with_close_scores_cannot_be_silently_selected():
    metrics = {
        "blocked_visibility_fraction": 0.0,
        "floor_wall_fraction": 0.0,
        "wall_trimmed_distance_px": 1.0,
    }
    candidates = [
        {"score": 1.0, "structural_metrics": metrics},
        {"score": 0.95, "structural_metrics": metrics},
    ]
    signatures = [np.array([[0.0, 0], [1, 1]]), np.array([[100.0, 0], [101, 1]])]
    decision = select_place(candidates, signatures, 6)
    assert decision["status"] == "abstained" and decision["selected_rank"] is None
    assert "insufficient_margin_between_distinct_places" in decision["reasons"]


def test_numerical_variants_of_one_place_do_not_hide_different_runner_up():
    metrics = {
        "blocked_visibility_fraction": 0.0,
        "floor_wall_fraction": 0.0,
        "wall_trimmed_distance_px": 1.0,
    }
    candidates = [{"score": score, "structural_metrics": metrics} for score in [1.0, 0.99, 0.5]]
    signatures = [np.full((3, 2), offset) for offset in [0.0, 0.1, 100.0]]
    decision = select_place(candidates, signatures, 6)
    assert decision["distinct_runner_up_rank"] == 3
    assert decision["status"] == "candidate_requires_validation"
    assert not decision["accuracy_certified"]


def test_single_hypothesis_without_competing_search_support_is_not_certified():
    candidate = {
        "score": 1.0,
        "structural_metrics": {
            "blocked_visibility_fraction": 0.0,
            "floor_wall_fraction": 0.0,
            "wall_trimmed_distance_px": 1.0,
        },
    }
    assert select_place([candidate], [np.zeros((3, 2))], 6)["selected_rank"] is None


def test_door_anchor_requires_visible_floor_near_detected_door_bottom():
    floor = np.zeros((480, 640), np.uint8)
    floor[320:] = 255
    anchor = door_floor_anchor([200, 100, 350, 320], floor)
    assert np.allclose(anchor, [275.0, 320.0])
    assert door_floor_anchor([200, 100, 350, 250], floor) is None
    assert door_floor_anchor([200, 100, 350, 320], np.zeros_like(floor)) is None


def test_door_size_penalty_rejects_collapsed_scale_without_absolute_meter_measurement():
    assert door_scale_error([100.0], [20.0], 0.2) == 0.0
    assert np.isclose(door_scale_error([100.0], [20.0], 0.02), np.log(10.0))
    assert np.isclose(door_scale_error([100.0], [20.0], 2.0), np.log(10.0))
    assert door_scale_error([None], [20.0], 0.2) is None
