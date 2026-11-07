import cv2
import numpy as np
import pytest
from fragment_search import compose_to_original, distinct_candidates, fit_candidate, rotation_matrix, windows
from grouped_experiment import validate_groups
from neural_overlap import assess_pair, connected_groups
from verify_overlap import independent_support


def test_overlap_accepts_shared_3d_scene_and_rejects_unrelated_points():
    rng = np.random.default_rng(7)
    xyz = rng.uniform([-2, -1.5, 4], [2, 1.5, 8], (250, 3))
    a = xyz[:, :2] / xyz[:, 2:] * 500 + [320, 240]
    moved = xyz + [0.8, 0.1, 0.2]
    b = moved[:, :2] / moved[:, 2:] * 500 + [320, 240]
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    good, _ = assess_pair(a, b, (480, 640), (480, 640))
    bad, _ = assess_pair(
        a, rng.uniform([0, 0], [640, 480], a.shape).astype(np.float32), (480, 640), (480, 640)
    )
    assert good["accepted_overlap_edge"]
    assert good["holdout_inlier_ratio"] > 0.95
    assert not bad["accepted_overlap_edge"]


def test_collinear_matches_do_not_establish_overlap():
    x = np.column_stack((np.linspace(50, 500, 80), np.ones(80) * 200)).astype(np.float32)
    result, _ = assess_pair(x, x + [30, 0], (480, 640), (480, 640))
    assert not result["accepted_overlap_edge"]


def test_graph_uses_only_confirmed_edges_and_preserves_singletons():
    pairs = {
        "a": {"cameras": ["a", "b"], "accepted_overlap_edge": True},
        "b": {"cameras": ["b", "c"], "accepted_overlap_edge": False},
    }
    assert connected_groups(["a", "b", "c"], pairs) == [["a", "b"], ["c"]]
    validate_groups([["a", "b"], ["c"]], ["a", "b", "c"])
    with pytest.raises(ValueError):
        validate_groups([["a", "b"], ["b", "c"]], ["a", "b", "c"])


def test_crop_and_rotation_coordinates_round_trip():
    crop = np.array([[2.0, 0, -120], [0, 2, -60], [0, 0, 1]])
    affine = np.array([[0.8, -0.2, 50], [0.2, 0.8, 20]])
    source = np.array([[80.0, 100.0, 1.0], [200.0, 360.0, 1.0]])
    for turn in range(4):
        rotation = rotation_matrix(turn)
        result = compose_to_original(affine, crop, rotation)
        target = (source @ rotation.T) @ np.vstack((affine, [0, 0, 1])).T @ np.linalg.inv(crop).T
        assert np.allclose(source @ result.T, target)


def test_fragment_fit_rejects_collapsed_soft_positions():
    yy, xx = np.mgrid[60:460:60, 60:460:60]
    source = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32)
    assert fit_candidate(source, np.full_like(source, 256), np.ones(len(source))) is None
    affine = np.array([[0.6, -0.1, 100], [0.1, 0.6, 40]])
    target = cv2.transform(source[:, None], affine)[:, 0]
    result = fit_candidate(source, target, np.ones(len(source)) * 0.8)
    assert result is not None
    assert np.allclose(result["affine"], affine, atol=0.001)


def test_generic_windows_cover_edges_and_stay_inside_plan():
    candidates = windows(612, 658)
    coverage = np.zeros((658, 612), bool)
    for x, y, w, h in candidates:
        assert 0 <= x < x + w <= 612 and 0 <= y < y + h <= 658
        if (w, h) != (612, 658):
            coverage[y : y + h, x : x + w] = True
    assert coverage.all()


def test_cross_method_support_distinguishes_random_and_consistent_matches():
    rng = np.random.default_rng(6)
    x = rng.uniform([30, 30], [600, 440], (70, 2))
    y = x + [20, 0]
    fundamental = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0.0]])
    good = independent_support(fundamental, x, y)
    bad = independent_support(fundamental, x, rng.uniform([30, 30], [600, 440], x.shape))
    assert good["inliers"] == 70 and good["binomial_tail_approx"] < 1e-4
    assert bad["inliers"] < 8


def test_candidate_deduplication_preserves_different_rotations_at_same_position():
    matrices = [
        np.eye(3),
        np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]]),
        np.array([[0, -1, 512], [1, 0, 0], [0, 0, 1]]),
    ]
    candidates = [
        {"score": 3 - i, "density_to_original": matrix.tolist()} for i, matrix in enumerate(matrices)
    ]
    kept = distinct_candidates(candidates)
    assert len(kept) == 2
    assert kept[1]["density_to_original"] == matrices[2].tolist()
