import cv2
import numpy as np
import pytest
from check_lens import residuals
from evaluate import evaluate_camera
from lens_geometry import distort_points, project_calibration, undistort_points
from PIL import Image
from structural_features import plan_structure


def lens(model="simple_divisional", k=-0.2):
    return {
        "model": model,
        "k1": k,
        "size": [640, 480],
        "rectified_size": [640, 480],
        "K": [[700.0, 0, 320], [0, 700.0, 240], [0, 0, 1]],
        "new_K": [[580.0, 0, 320], [0, 580.0, 240], [0, 0, 1]],
    }


@pytest.mark.parametrize("model", ["simple_radial", "simple_divisional"])
@pytest.mark.parametrize("k", [-0.2, 0.0, 0.15])
def test_lens_point_round_trip_with_changed_focal_length(model, k):
    config = lens(model, k)
    ideal = np.array([[80.0, 60.0], [550, 410], [320, 240], [180, 350]])
    distorted, valid_forward = distort_points(ideal, config)
    recovered, valid_inverse = undistort_points(distorted, config)
    assert valid_forward.all() and valid_inverse.all()
    assert np.allclose(recovered, ideal, atol=1e-6)


def test_invalid_radial_branch_is_not_silently_projected():
    config = lens("simple_radial", -0.5)
    config["K"] = [[100.0, 0, 0], [0, 100.0, 0], [0, 0, 1]]
    result, valid = undistort_points(np.array([[100.0, 100.0], [0, 0]]), config)
    assert valid.tolist() == [False, True]
    assert np.isnan(result[0]).all()


def test_calibration_evaluation_uses_nonlinear_lens_before_homography():
    config = lens()
    ideal = np.array([[100.0, 100.0], [500, 100], [500, 380], [100, 380]])
    original, _ = distort_points(ideal, config)
    matrix = np.array([[0.4, 0, 15], [0, 0.5, 10], [0, 0, 1.0]])
    expected = cv2.perspectiveTransform(ideal[:, None], matrix)[:, 0]
    prediction = {
        "camera_size": [640, 480],
        "plan_size": [612, 658],
        "lens": config,
        "lens_correction_applied": True,
        "homography_input_space": "rectified_pixels",
        "homography": matrix.tolist(),
    }
    reference = {
        "camera_size": [640, 480],
        "plan_size": [612, 658],
        "camera_points": original.tolist(),
        "plan_points": expected.tolist(),
        "inliers": [True] * 4,
    }
    result = evaluate_camera(prediction, reference)
    assert result["median_error_px"] < 1e-6
    assert result["unprojectable_reference_inliers"] == 0


def test_corrected_points_outside_observed_frame_are_counted_as_unprojectable():
    config = lens("simple_divisional", -0.7)
    config["new_K"] = config["K"]
    prediction = {
        "lens": config,
        "lens_correction_applied": True,
        "homography_input_space": "rectified_pixels",
        "homography": np.eye(3),
    }
    _, valid = project_calibration(np.array([[639.0, 479.0], [320, 240]]), prediction)
    assert valid.tolist() == [False, True]


def test_lens_correction_straightens_a_synthetic_physical_edge():
    config = lens("simple_divisional", -0.4)
    ideal = np.column_stack((np.full(100, 100.0), np.linspace(50, 430, 100)))
    original, _ = distort_points(ideal, config)
    rms_before, _ = residuals([original])
    rms_after, _ = residuals([original], config)
    assert rms_before[0] > 0.1
    assert rms_after[0] < 1e-7


def test_no_door_is_invented_in_unbroken_walls():
    image = np.full((200, 200, 3), 255, np.uint8)
    cv2.rectangle(image, (20, 30), (180, 170), (0, 0, 0), 5)
    _, walls, doors = plan_structure(Image.fromarray(image))
    assert (walls > 0).sum() > 1000
    assert not doors


def test_wall_gap_arc_and_leaf_produce_door_proposal():
    image = np.full((220, 220, 3), 255, np.uint8)
    cv2.line(image, (20, 60), (80, 60), (0, 0, 0), 3)
    cv2.line(image, (100, 60), (200, 60), (0, 0, 0), 3)
    cv2.line(image, (80, 60), (80, 80), (0, 0, 0), 1)
    cv2.ellipse(image, (80, 60), (20, 20), 0, 0, 90, (0, 0, 0), 1)
    _, _, doors = plan_structure(Image.fromarray(image))
    assert any(np.linalg.norm(np.array(door["center"]) - [90, 60]) < 8 for door in doors)


@pytest.mark.parametrize("door_depth, expected_count", [(5.0, 1), (600.0, 0)])
def test_door_outside_density_frame_cannot_anchor_a_plan_candidate(tmp_path, door_depth, expected_count):
    from structural_features import structural_density

    for name in ["wall", "door"]:
        (tmp_path / name).mkdir()
    wall = np.full((160, 160), 255, np.uint8)
    door = np.zeros_like(wall)
    door[40:120, 40:120] = 255
    cv2.imwrite(str(tmp_path / "wall/camera.png"), wall)
    cv2.imwrite(str(tmp_path / "door/camera.png"), door)
    yy, xx = np.mgrid[:20, :20]
    points = np.stack((xx * 0.1, yy * 0.1, np.full_like(xx, 5, dtype=float)), axis=-1)
    points[5:15, 5:15, 2] = door_depth
    recon = {"points_cam0": points[None], "conf": np.ones((1, 20, 20)), "image_sizes": np.array([[20, 20]])}
    projection = {
        "R_rect": np.eye(3),
        "min_x": 0.0,
        "min_z": 0.0,
        "scale": 1.0,
        "cx": 0.0,
        "cz": 0.0,
        "density_size": 512,
    }
    _, _, centers, _ = structural_density(recon, projection, [tmp_path / "camera.jpg"], tmp_path)
    assert len(centers) == expected_count
