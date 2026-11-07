from copy import deepcopy

import cv2
import numpy as np
import pytest
from floor_ray_geometry import transform_points
from motion_body_geometry import fit_body_edge
from motion_candidates import propagate_candidates, verify_sources
from motion_floor import admissible_floor
from motion_foreground import motion_pixels
from motion_geometry_matching import resolve_training
from motion_joint import motion_map_cost, refine_component, scale_invariant_errors, solve
from motion_observations import visible_foot
from motion_pairs import fit_relative_edge, matching_floor_points, mutual_appearance


def synthetic():
    k = np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1.0]])
    lens = {
        "model": "simple_divisional",
        "k1": 0.0,
        "K": k.tolist(),
        "new_K": k.tolist(),
        "gravity": [0.0, 0, 1.0],
        "size": [640, 480],
        "rectified_size": [640, 480],
    }
    angle = 0.2
    scale = 0.85
    m = np.array(
        [
            [np.cos(angle) * scale, -np.sin(angle) * scale, 0.1],
            [np.sin(angle) * scale, np.cos(angle) * scale, -0.08],
            [0, 0, 1.0],
        ]
    )
    image_transform = k @ m @ np.linalg.inv(k)
    rng = np.random.default_rng(42)
    rows = []
    for split, count, start in [("train", 40, 0), ("holdout", 20, 100)]:
        for index in range(count):
            t = start + index
            person = index % 2
            a = rng.uniform([170, 130], [450, 350])
            b = transform_points(a[None], image_transform)[0]
            rows.append(
                {
                    "cameras": ["a", "b"],
                    "track_ids": [f"a:{person}", f"b:{person}"],
                    "frame_ids": [f"a:{t}", f"b:{t}"],
                    "timestamp_sec": float(t),
                    "points": [a.tolist(), b.tolist()],
                    "split": split,
                    "window": split,
                    "foot_confidences": [0.95, 0.95],
                    "appearance_cosine": 0.9,
                }
            )
    return rows, lens, image_transform


def calibration(h):
    return {"camera_size": [640, 480], "plan_size": [612, 658], "homography": np.asarray(h).tolist()}


def bank_for(rows, image_transform, equal_prior=False):
    probes = np.array([[170.0, 130], [450, 130], [450, 350], [170, 350]])
    ha = np.array([[0.4, 0, 50], [0, 0.4, 40], [0, 0, 1.0]])
    hb = ha @ np.linalg.inv(image_transform)

    def translated(h, dx):
        d = np.eye(3)
        d[0, 2] = dx
        return d @ h

    def item(name, h, cost):
        return {
            "id": name,
            "calibration": calibration(h),
            "unary_cost": cost,
            "signature": transform_points(probes, h).tolist(),
        }

    return {
        "cameras": {
            "a": {
                "probes": probes.tolist(),
                "candidates": [
                    item("a-correct", ha, 0),
                    item("a-shifted", translated(ha, 80), 0 if equal_prior else 1),
                ],
            },
            "b": {
                "probes": probes.tolist(),
                "candidates": [
                    item("b-wrong", translated(hb, -50), 0),
                    item("b-correct", hb, 0.2),
                    item("b-shifted", translated(hb, 80), 0.2 if equal_prior else 1),
                ],
            },
        }
    }


def report_for(rows, lens):
    edge = fit_relative_edge(rows, lens, lens) | {"cameras": ["a", "b"]}
    return {"pairs": rows, "edges": {"a-b": edge}}


def test_floor_motion_recovers_relative_geometry_on_unseen_times():
    rows, lens, expected = synthetic()
    edge = fit_relative_edge(rows, lens, lens)
    assert edge["accepted"] and edge["train_supported"]
    h = np.asarray(edge["ideal_a_to_b"])
    h /= h[2, 2]
    assert np.allclose(h, expected, atol=1e-4)
    assert edge["holdout"]["p95_symmetric_px"] < 1e-3


def test_holdout_failure_does_not_silently_become_training_data():
    rows, lens, _ = synthetic()
    before = fit_relative_edge(rows, lens, lens)
    for row in rows:
        if row["split"] == "holdout":
            row["points"][1][0] += 140
    after = fit_relative_edge(rows, lens, lens)
    assert after["train_supported"] and not after["accepted"]
    assert before["floor_similarity"] == after["floor_similarity"]


def test_stationary_people_cannot_establish_camera_geometry():
    rows, lens, _ = synthetic()
    for row in rows:
        row["points"] = [[320.0, 240.0], [330.0, 240.0]]
    edge = fit_relative_edge(rows, lens, lens)
    assert not edge["accepted"] and edge["reason"] == "insufficient_motion_extent"


def test_only_one_trajectory_is_insufficient_even_with_many_frames():
    rows, lens, _ = synthetic()
    for row in rows:
        row["track_ids"] = ["a:one", "b:one"]
    assert not fit_relative_edge(rows, lens, lens)["train_supported"]


def test_joint_motion_can_overrule_wrong_independent_top_candidate():
    rows, lens, image_transform = synthetic()
    pairs = report_for(rows, lens)
    result = solve(bank_for(rows, image_transform), pairs)
    assert result["components"][0]["chosen_ids"] == {"a": "a-correct", "b": "b-correct"}
    assert not result["absolute_location_verified"] and not result["automatic_deployment_ready"]


def test_changing_holdout_positions_cannot_change_selected_hypothesis_or_refinement():
    rows, lens, image_transform = synthetic()
    pairs = report_for(rows, lens)
    bank = bank_for(rows, image_transform)
    original = solve(bank, pairs)
    modified = deepcopy(pairs)
    for row in modified["pairs"]:
        if row["split"] == "holdout":
            row["points"][1][0] += 200
    other = solve(bank, modified)
    assert original["components"][0]["chosen_ids"] == other["components"][0]["chosen_ids"]
    assert original["calibrations"] == other["calibrations"]
    assert any("holdout_failed" in r for r in other["components"][0]["reasons"])


def test_common_map_translation_remains_ambiguous_from_motion_alone():
    rows, lens, image_transform = synthetic()
    result = solve(bank_for(rows, image_transform, equal_prior=True), report_for(rows, lens))
    assert "ambiguous_absolute_place_hypotheses" in result["components"][0]["reasons"]


def test_refinement_keeps_anchor_fixed_and_cannot_collapse_the_map():
    rows, lens, image_transform = synthetic()
    bank = bank_for(rows, image_transform)
    source = {
        c: deepcopy(bank["cameras"][c]["candidates"][0 if c == "a" else 1]["calibration"]) for c in ["a", "b"]
    }
    source["b"]["homography"][0][2] += 6
    refined, diagnostic = refine_component(source, [v for v in rows if v["split"] == "train"])
    assert np.array_equal(refined["a"]["homography"], source["a"]["homography"])
    assert diagnostic["performed"] and diagnostic["final_cost"] < diagnostic["initial_cost"]
    assert abs(diagnostic["parameters"]["b"][3]) <= 0.15


def test_identical_clothing_without_appearance_margin_is_rejected():
    a = [{"embedding": [1.0, 0.0]}, {"embedding": [1.0, 0.0]}]
    b = [{"embedding": [1.0, 0.0]}]
    assert not mutual_appearance(a, b)


def test_invisible_or_occluded_ankles_are_not_replaced_by_bbox_bottom():
    floor = np.full((480, 640), 255, np.uint8)
    k = np.zeros((17, 3))
    k[15] = [240.0, 420.0, 0.8]
    k[16] = [260.0, 420.0, 0.3]
    foot, reason = visible_foot([200, 100, 300, 450], k, floor)
    assert reason is None and np.allclose(foot, [240.0, 420.0])
    k[15, 2] = 0.3
    assert visible_foot([200, 100, 300, 450], k, floor)[0] is None
    k[15, 2] = 0.8
    assert visible_foot([200, 100, 300, 450], k, np.zeros_like(floor))[0] is None
    assert visible_foot([200, 100, 300, 450], k, floor, [[230, 300, 290, 460]])[0] is None


def test_common_map_shrink_cannot_improve_motion_objective():
    rows, lens, image_transform = synthetic()
    bank = bank_for(rows, image_transform)
    source = {c: deepcopy(bank["cameras"][c]["candidates"][0]["calibration"]) for c in ["a", "b"]}
    before = scale_invariant_errors(rows, source)
    tiny = deepcopy(source)
    for c in tiny:
        tiny[c]["homography"] = (np.diag([0.01, 0.01, 1.0]) @ np.array(tiny[c]["homography"])).tolist()
    after = scale_invariant_errors(rows, tiny)
    assert np.allclose(before, after, rtol=1e-7)


def test_uncertain_floor_is_not_a_wall_but_confident_screen_is_excluded():
    mask = admissible_floor(np.array([0.35, 0.02, 0.0]), np.array([0.55, 0.95, 0.0]))
    assert mask.tolist() == [True, False, True]


def test_joint_training_association_resolves_ambiguous_identity_proposals():
    rows, lens, _ = synthetic()
    proposals = []
    for row in rows:
        if row["split"] != "train":
            continue
        actual = deepcopy(row)
        actual["ground"] = [[(p[0] - 320) / 500, (p[1] - 240) / 500] for p in row["points"]]
        proposals.append(actual)
        wrong = deepcopy(actual)
        wrong["track_ids"][1] += "wrong"
        wrong["ground"][1] = [
            actual["ground"][1][0] + 0.4 * np.sin(row["timestamp_sec"]),
            actual["ground"][1][1] + 0.6,
        ]
        proposals.append(wrong)
    chosen, hypotheses = resolve_training(proposals)
    assert len(chosen) >= 35 and hypotheses
    assert sum("wrong" in r["track_ids"][1] for r in chosen) <= 2
    assert len({tuple(r["frame_ids"]) for r in chosen}) == len(chosen)


def body_synthetic():
    rng = np.random.default_rng(81)
    g = np.array([0.0, 0.8, 0.6])
    k = np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1.0]])
    rotation = cv2.Rodrigues(g * 0.2)[0]
    translation = np.array([0.35, 0.02, 0.05])
    lens = {
        "model": "simple_divisional",
        "k1": 0.0,
        "K": k.tolist(),
        "new_K": k.tolist(),
        "gravity": g.tolist(),
        "size": [640, 480],
        "rectified_size": [640, 480],
    }

    def pixel(point):
        p = k @ point
        return p[:2] / p[2]

    rows = []
    for split, count, start in [("train", 40, 0), ("holdout", 20, 100)]:
        for index in range(count):
            z = rng.uniform(1.1, 1.8)
            foot = np.array([rng.uniform(-0.3, 0.3), (1 - 0.6 * z) / 0.8, z])
            a, b = np.zeros((17, 3)), np.zeros((17, 3))
            for joint, height in zip(
                [5, 6, 11, 12, 13, 14, 15, 16], [0.45, 0.45, 0.3, 0.3, 0.15, 0.15, 0.0, 0.0]
            ):
                offset = np.array([-0.04 if joint % 2 else 0.04, 0.0, 0.0])
                point = foot - g * height + offset
                a[joint] = np.r_[pixel(point), 0.99]
                b[joint] = np.r_[pixel(rotation @ point + translation), 0.99]
            t = start + index
            person = index % 2
            rows.append(
                {
                    "cameras": ["a", "b"],
                    "track_ids": [f"a:{person}", f"b:{person}"],
                    "frame_ids": [f"a:{t}", f"b:{t}"],
                    "timestamp_sec": float(t),
                    "points": [pixel(foot).tolist(), pixel(rotation @ foot + translation).tolist()],
                    "keypoints": [a.tolist(), b.tolist()],
                    "split": split,
                    "window": split,
                    "foot_confidences": [0.99, 0.99],
                    "appearance_cosine": 0.95,
                }
            )
    expected = k @ (rotation + np.outer(translation, g)) @ np.linalg.inv(k)
    return rows, lens, expected


def test_body_pose_and_sparse_foot_scale_recover_a_physical_floor_homography():
    rows, lens, expected = body_synthetic()
    edge = fit_body_edge(rows, lens, lens)
    assert edge["accepted"], edge["reason"]
    matrix = np.array(edge["ideal_a_to_b"])
    matrix /= matrix[2, 2]
    expected /= expected[2, 2]
    assert np.allclose(matrix, expected, atol=1e-3)


def test_body_joints_cannot_be_treated_as_floor_when_feet_are_hidden():
    rows, lens, _ = body_synthetic()
    for row in rows:
        row["points"] = [None, None]
    edge = fit_body_edge(rows, lens, lens)
    assert not edge["train_supported"]
    assert "insufficient_visible_floor_scale_pairs" in edge["reason"]


def test_moving_track_cannot_cut_through_wall_but_can_use_doorway():
    rows = [
        {
            "cameras": ["a", "b"],
            "points": [[x, 50.0], [x, 50.0]],
            "track_ids": ["a:1", "b:1"],
            "timestamp_sec": i,
        }
        for i, x in enumerate([30.0, 70.0])
    ]
    walls = np.zeros((100, 100), np.uint8)
    walls[:, 50] = 255
    assert motion_map_cost(rows, "a", calibration(np.eye(3)), walls) > 0
    walls[45:56, 50] = 0
    assert motion_map_cost(rows, "a", calibration(np.eye(3)), walls) == 0


def test_different_visible_ankles_cannot_be_declared_the_same_ground_point():
    _, lens, _ = body_synthetic()
    a = np.zeros((17, 3))
    b = a.copy()
    a[15] = [300, 350, 0.9]
    b[16] = [320, 350, 0.9]
    x = {"camera": "a", "foot": [300.0, 350.0], "keypoints": a}
    y = {"camera": "b", "foot": [320.0, 350.0], "keypoints": b}
    assert matching_floor_points(x, y, {"a": lens, "b": lens}) is None


def test_motion_transfer_composes_lens_space_in_the_right_order():
    rows, lens, image_transform = synthetic()
    bank = bank_for(rows, image_transform)
    parent = bank["cameras"]["a"]["candidates"][0]
    parent.update(source="synthetic_automatic_model", rank=1, source_score=1.0)
    parent["calibration"].update(
        lens=lens, lens_correction_applied=True, homography_input_space="ideal_pixels"
    )
    for c in bank["cameras"]:
        for item in bank["cameras"][c]["candidates"]:
            item.setdefault("source", "synthetic_automatic_model")
            item.setdefault("rank", 1)
            item.setdefault("source_score", 1.0)
    edge = report_for(rows, lens)
    count = propagate_candidates(bank, edge, {"a": lens, "b": lens})
    assert count["b"] > 0
    transferred = next(v for v in bank["cameras"]["b"]["candidates"] if v.get("motion_source_camera") == "a")
    expected = np.array(parent["calibration"]["homography"]) @ np.linalg.inv(image_transform)
    actual = np.array(transferred["calibration"]["homography"])
    assert np.allclose(actual / actual[2, 2], expected / expected[2, 2], atol=1e-5)


def test_reusing_automatic_candidates_for_a_different_video_is_rejected():
    with pytest.raises(ValueError, match="another source video"):
        verify_sources(
            {"cameras": {"a": {"source_sha256": "new"}}}, {"cameras": {"a": {"video_sha256": "old"}}}
        )


def test_illumination_change_is_not_human_motion_but_a_new_object_is():
    background = np.full((100, 100, 3), 80, np.uint8)
    brighter = background + 40
    assert not motion_pixels(brighter, background).any()
    brighter[30:70, 30:70] = 220
    assert motion_pixels(brighter, background)[35:65, 35:65].mean() > 0.95
