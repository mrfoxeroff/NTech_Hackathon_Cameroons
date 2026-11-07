import json

import numpy as np
from PIL import Image
from run_experiment import recover_homographies


def test_predicted_3d_floor_recovers_homography_without_reference_json(tmp_path):
    image = tmp_path / "camera.jpg"
    Image.new("RGB", (640, 480)).save(image)
    masks = tmp_path / "masks"
    masks.mkdir()
    Image.new("L", (640, 480), 255).save(masks / "camera.png")
    yy, xx = np.mgrid[:518, :518]
    xyz = np.stack((xx * 640 / 518, np.ones_like(xx) * 2, (yy - 63) * 480 / 392), -1).astype(float)
    recon = {"points_cam0": xyz[None], "conf": np.ones((1, 518, 518)), "image_sizes": np.array([[518, 392]])}
    alignment = {
        "proj_params": {
            "R_rect": np.eye(3),
            "min_x": 0,
            "min_z": 0,
            "scale": 1,
            "cx": 0,
            "cz": 0,
            "density_size": 512,
        },
        "M": np.array([[0.3, 0, 20], [0, -0.4, 240]]),
    }
    result = recover_homographies(recon, alignment, [image], masks, (600, 500), (1, 0, 0), tmp_path)
    assert result["camera"]["status"] == "automatic_candidate"
    calibration = json.loads((tmp_path / "camera-calibration.json").read_text())
    expected = np.array([[0.3, 0, 20], [0, 0.4, 35.6], [0, 0, 1]])
    assert np.allclose(calibration["homography"], expected, atol=0.001)
    assert calibration["manual_landmarks_used"] is False
