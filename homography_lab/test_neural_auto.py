import numpy as np
from PIL import Image

from geometry import warp
from neural_auto import estimate, letterbox, original_coordinates, sample_correspondences


def synthetic_mapping():
    _, camera = letterbox(Image.new("RGB", (640, 480)))
    _, plan = letterbox(Image.new("RGB", (600, 500)))
    yy, xx = np.mgrid[:512, :512]
    source = original_coordinates(np.stack((xx, yy), -1), camera)
    h = np.array([[0.4, 0.04, 20], [-0.02, 0.5, 30], [0.0002, -0.0001, 1]])
    target = warp(source.reshape(-1, 2), h).reshape(512, 512, 2)
    return target * plan["ratio"] + plan["offset"], camera, plan


def test_letterbox_coordinates_are_reversible():
    _, transform = letterbox(Image.new("RGB", (2592, 1944)))
    point = np.array([1800.0, 1200.0])
    assert np.allclose(
        original_coordinates(point * transform["ratio"] + transform["offset"], transform), point
    )


def test_automatic_correspondences_need_no_manual_points_but_are_not_ground_truth():
    mapping, camera, plan = synthetic_mapping()
    pairs = sample_correspondences(mapping, np.ones((480, 640)), plan, camera)
    calibration, report = estimate(pairs, camera["size"], plan["size"])
    assert report["status"] == "automatic_candidate"
    assert calibration["status"] == "automatic_candidate"
    assert not report["manual_landmarks_used"] and not report["physical_accuracy_verified"]


def test_floor_segmentation_prevents_using_non_floor_pixels():
    mapping, camera, plan = synthetic_mapping()
    assert sample_correspondences(mapping, np.zeros((480, 640)), plan, camera) == []


def test_collapsed_network_predictions_are_rejected():
    mapping, camera, plan = synthetic_mapping()
    mapping[:] = [256, 256]
    pairs = sample_correspondences(mapping, np.ones((480, 640)), plan, camera)
    calibration, report = estimate(pairs, camera["size"], plan["size"])
    assert calibration is None and report["status"] == "rejected"
