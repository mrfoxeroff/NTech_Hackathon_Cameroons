"""Regression tests for motion stitching and ambiguous cross-camera identity."""

import unittest
from dataclasses import replace

from associate import DEFAULTS, Observation, appearance, associate, joint_overlap_choice, split_tracklets


CFG = DEFAULTS | {"stitch_local_fragments": True, "contextual_appearance": True,
                  "resolve_group_conflicts": True, "joint_overlap_matching": True,
                  "overlap_pairs": [["a", "b"]]}


def fragment(camera, ident, start, end, x=0, vx=0.1, embedding=(1.0, 0.0)):
    return [Observation.parse({"camera_id": camera, "local_id": ident,
                               "frame_index": t // 100, "timestamp_ms": t,
                               "embedding": embedding,
                               "bbox_xyxy": [x + vx * t, 0, x + vx * t + 50, 200]})
            for t in range(start, end + 1, 100)]


class StitchTests(unittest.TestCase):
    def test_short_occlusion_with_motion(self):
        obs = fragment("a", 1, 0, 500) + fragment("a", 2, 1000, 1500)
        self.assertEqual(len(associate(obs, CFG)), 1)

    def test_missing_boxes_preserves_fragments(self):
        obs = [replace(o, bbox=None) for o in fragment("a", 1, 0, 500) + fragment("a", 2, 1000, 1500)]
        self.assertEqual(len(associate(obs, CFG)), 2)

    def test_no_appearance_is_not_identity_evidence(self):
        obs = [replace(o, reid=None) for o in fragment("a", 1, 0, 500) + fragment("a", 2, 1000, 1500)]
        self.assertEqual(len(associate(obs, CFG)), 2)

    def test_similar_clothes_wrong_position(self):
        obs = fragment("a", 1, 0, 500) + fragment("a", 2, 1000, 1500, x=400)
        self.assertEqual(len(associate(obs, CFG)), 2)

    def test_wrong_appearance_same_motion(self):
        obs = fragment("a", 1, 0, 500) + fragment("a", 2, 1000, 1500, embedding=(0.0, 1.0))
        self.assertEqual(len(associate(obs, CFG)), 2)

    def test_crossing_same_clothes_uses_bidirectional_motion(self):
        obs = fragment("a", 1, 0, 500) + fragment("a", 2, 0, 500, x=200, vx=-0.1)
        obs += fragment("a", 3, 1000, 1500) + fragment("a", 4, 1000, 1500, x=200, vx=-0.1)
        groups = associate(obs, CFG)
        self.assertEqual([{o.local_id for o in g} for g in groups], [{1, 3}, {2, 4}])

    def test_ambiguous_reappearance_abstains(self):
        obs = fragment("a", 1, 0, 500)
        obs += fragment("a", 2, 1000, 1500, x=-5) + fragment("a", 3, 1000, 1500, x=5)
        self.assertEqual(len(associate(obs, CFG)), 3)

    def test_confirmed_exit_is_final(self):
        obs = fragment("a", 1, 0, 500)
        obs[-1] = replace(obs[-1], exit_after=True)
        obs += fragment("a", 2, 1000, 1500)
        self.assertEqual(len(associate(obs, CFG)), 2)

    def test_long_absence_is_not_reentry(self):
        obs = fragment("a", 1, 0, 500) + fragment("a", 2, 2200, 2700)
        self.assertEqual(len(associate(obs, CFG)), 2)

    def test_single_frame_duplicate_boundary(self):
        obs = fragment("a", 1, 0, 500) + fragment("a", 2, 500, 1000)
        result = associate(obs, CFG)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), len(obs))

    def test_overlapping_similar_people_remain_separate(self):
        obs = fragment("a", 1, 0, 700) + fragment("a", 2, 400, 1100)
        self.assertEqual(len(associate(obs, CFG)), 2)

    def test_two_cameras_identical_clothing_abstains(self):
        obs = sum((fragment(cam, ident, 0, 600, x=ident * 100)
                   for cam in ("a", "b") for ident in (1, 2)), [])
        self.assertEqual(len(associate(obs, CFG)), 4)

    def test_joint_evidence_breaks_local_tie(self):
        ts = split_tracklets(fragment("a", 1, 0, 600) + fragment("a", 2, 0, 600)
                             + fragment("b", 1, 0, 600) + fragment("b", 2, 0, 600), 750)
        # Individually 0->2 vs 0->3 is ambiguous; 1 can only match 3.
        edges = [(0.80, 0, 2, None, None), (0.79, 0, 3, None, None), (0.78, 1, 3, None, None)]
        chosen = joint_overlap_choice(edges, list(range(4)), {i: [i] for i in range(4)}, ts,
                                      {frozenset(("a", "b"))}, CFG)
        self.assertIn(chosen, [(0, 2), (1, 3)])

    def test_joint_tied_assignments_abstain(self):
        ts = split_tracklets(fragment("a", 1, 0, 600) + fragment("a", 2, 0, 600)
                             + fragment("b", 1, 0, 600) + fragment("b", 2, 0, 600), 750)
        edges = [(0.8, i, j, None, None) for i in (0, 1) for j in (2, 3)]
        chosen = joint_overlap_choice(edges, list(range(4)), {i: [i] for i in range(4)}, ts,
                                      {frozenset(("a", "b"))}, CFG)
        self.assertIsNone(chosen)

    def test_disjoint_fragments_can_share_other_camera(self):
        obs = fragment("a", 1, 0, 400) + fragment("a", 2, 700, 1100, x=500)
        obs += fragment("b", 1, 0, 1100)
        self.assertEqual(len(associate(obs, CFG)), 1)

    def test_transitive_group_cannot_cross_confirmed_exit(self):
        obs = fragment("a", 1, 0, 400)
        obs[-1] = replace(obs[-1], exit_after=True)
        obs += fragment("a", 2, 700, 1100) + fragment("b", 1, 0, 1100)
        self.assertGreaterEqual(len(associate(obs, CFG)), 2)
        for group in associate(obs, CFG):
            self.assertFalse({1, 2} <= {o.local_id for o in group if o.camera == "a"})

    def test_solver_limit_abstains_in_ambiguous_component(self):
        obs = sum((fragment(cam, ident, 0, 600) for cam in ("a", "b") for ident in (1, 2)), [])
        self.assertEqual(len(associate(obs, CFG | {"joint_max_edges": 1})), 4)

    def test_legacy_config_does_not_stitch(self):
        obs = fragment("a", 1, 0, 500) + fragment("a", 2, 1000, 1500)
        self.assertEqual(len(associate(obs, {})), 2)

    def test_context_ranks_but_does_not_change_admission_threshold(self):
        # A shared pose can look different even when the full gallery passes.
        left = fragment("a", 1, 0, 400, embedding=(1.0, 0.0))
        left += fragment("a", 1, 500, 1900, embedding=(0.0, 1.0))
        right = fragment("b", 1, 0, 400, embedding=(0.0, 1.0))
        a, b = split_tracklets(left + right, 750)
        cfg = CFG | {"reid_min_similarity": 0.75}
        score = appearance(a, b, cfg)
        self.assertGreaterEqual(score, 0)
        self.assertLess(score, cfg["reid_min_similarity"])

    def test_invalid_geometry(self):
        with self.assertRaises(ValueError):
            Observation.parse({"camera_id": "a", "local_id": 1, "frame_index": 0,
                               "timestamp_ms": 0, "bbox_xyxy": [10, 0, 0, 100]})


if __name__ == "__main__":
    unittest.main()
