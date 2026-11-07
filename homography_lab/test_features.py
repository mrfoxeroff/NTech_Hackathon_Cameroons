import numpy as np

from features import automatic_proposal, motion_between, structural_features


def test_blank_images_cannot_produce_automatic_calibration():
    image = np.full((200, 300, 3), 255, np.uint8)
    proposal = automatic_proposal(image, image)
    assert proposal["status"] == "needs_landmark_confirmation"
    assert proposal["pairs"] == []
    assert structural_features(image)["candidates"] == []


def test_camera_motion_is_detected_independently_of_people_tracking():
    import cv2

    rng = np.random.default_rng(42)
    image = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    shifted = cv2.warpAffine(image, np.float32([[1, 0, 20], [0, 1, 0]]), (640, 480))
    assert motion_between(image, image) < 0.001
    assert motion_between(image, shifted) > 0.003
