"""Regression coverage for duplicated IDs and local tracker identity switches."""
from dataclasses import replace
import math

import numpy as np

from associate import DEFAULTS, Observation, associate, candidate, split_tracklets
from refine_cross_camera import fit_geometry, geometry_assign, key, reid_fallback, refine
from stabilize_tracklets import make_tracklet, merge_duplicates
from test_associate import fragment


CFG = DEFAULTS | {"overlap_pairs": [["a", "b"]], "reid_min_similarity": .30,
                  "resolve_group_conflicts": True, "temporal_evidence_ranking": True,
                  "merge_local_duplicates": True}


def test_sustained_match_outranks_brief_rival_without_relaxing_threshold():
    a = fragment("a", 1, 0, 5000, embedding=(1, 0))
    b = fragment("b", 1, 0, 5000, embedding=(.8, .6))
    rival = fragment("b", 2, 2000, 2300, x=400, embedding=(1, 0))
    groups = associate(a + b + rival, CFG)
    joined = next(g for g in groups if any(o.camera == "a" for o in g))
    assert {(o.camera, o.local_id) for o in joined} == {("a", 1), ("b", 1)}
    bad = make_tracklet(fragment("b", 3, 0, 10000, embedding=(.1, .995)))
    assert candidate(make_tracklet(a), bad, {frozenset(("a", "b"))}, set(), CFG) is None


def test_duplicate_boxes_preserve_all_observations():
    a = fragment("a", 1, 0, 1000, vx=0)
    b = fragment("a", 2, 300, 500, x=2, vx=0)
    ts = merge_duplicates(split_tracklets(a + b, 750), CFG)
    assert len(ts) == 1
    assert {key(o) for o in ts[0].observations} == {key(o) for o in a + b}


def test_people_crossing_once_are_not_duplicate_tracks():
    a = fragment("a", 1, 0, 1000, vx=.1)
    b = fragment("a", 2, 0, 1000, x=100, vx=-.1)
    assert len(merge_duplicates(split_tracklets(a + b, 750), CFG)) == 2


def test_single_detection_is_not_duplicate_evidence():
    rows = fragment("a", 1, 0, 1000, vx=0) + fragment("a", 2, 500, 500, vx=0)
    assert len(merge_duplicates(split_tracklets(rows, 750), CFG)) == 2


def test_source_boundary_and_exit_block_duplicate_merge():
    a = fragment("a", 1, 0, 1000, vx=0)
    b = [replace(o, source_segment=1) for o in fragment("a", 2, 300, 500, vx=0)]
    assert len(merge_duplicates(split_tracklets(a + b, 750), CFG)) == 2
    b = [replace(o, source_segment=0, exit_after=True) for o in b]
    assert len(merge_duplicates([make_tracklet(a), make_tracklet(b)], CFG)) == 2


def test_geometry_corrects_swapped_local_ids_and_abstains_at_boundary():
    a = fragment("a", 1, 0, 2000, x=0, vx=0)
    b = fragment("a", 2, 0, 2000, x=500, vx=0)
    x = fragment("b", 10, 0, 900, x=0, vx=0) + fragment("b", 10, 1000, 2000, x=500, vx=0)
    y = fragment("b", 20, 0, 900, x=500, vx=0) + fragment("b", 20, 1000, 2000, x=0, vx=0)
    assignments, _ = geometry_assign([make_tracklet(a), make_tracklet(b)],
                                     [make_tracklet(x), make_tracklet(y)], np.eye(3), {}, CFG)
    assert assignments[key(x[5])] == 0 and assignments[key(x[16])] == 1
    assert assignments[key(y[5])] == 1 and assignments[key(y[16])] == 0
    assert assignments[key(x[9])] is None and assignments[key(x[10])] is None
    for p, q in zip(x, y):
        assert assignments[key(p)] is None or assignments[key(q)] is None or assignments[key(p)] != assignments[key(q)]


def test_geometry_requires_two_independent_long_matches():
    ref = [make_tracklet(fragment("a", 1, 0, 10000))]
    other = [make_tracklet(fragment("b", 1, 0, 10000))]
    h, seeds, report = fit_geometry(ref, other, CFG)
    assert h is None and not seeds and not report["accepted"]


def test_geometry_validates_held_out_time_bins():
    reference, target = [], []
    for ident, embedding in [(1, (1, 0)), (2, (0, 1))]:
        a = fragment("a", ident, 0, 10000, x=ident * 500, vx=.03, embedding=embedding)
        a = [replace(o, bbox=(o.bbox[0], 100 * math.sin(o.time / 1500),
                             o.bbox[2], 100 * math.sin(o.time / 1500) + 200)) for o in a]
        b = [replace(o, camera="b", bbox=tuple(v + (120 if k % 2 == 0 else 70)
                                              for k, v in enumerate(o.bbox))) for o in a]
        reference.append(make_tracklet(a))
        target.append(make_tracklet(b))
    h, seeds, report = fit_geometry(reference, target, CFG)
    assert h is not None and report["accepted"] and seeds == {0: 0, 1: 1}
    assert report["validation_pairs"] >= 12
    assert report["validation_median_box_heights"] < .0001


def test_missing_secondary_embeddings_do_not_break_existing_local_groups():
    obs = fragment("a", 1, 0, 500) + fragment("a", 2, 1000, 1500)
    rows = [{} for _ in obs]
    groups = reid_fallback(obs, rows, CFG | {"stitch_local_fragments": True})
    assert len(groups) == 1 and len(groups[0]) == len(obs)


def test_explicit_exit_bypasses_geometry():
    rows = [{"camera_id": cam, "local_id": 1, "frame_index": i, "timestamp_ms": i * 100,
             "embedding": [1, 0], "confirmed_global_exit": i == 3}
            for cam in ("a", "b") for i in range(5)]
    groups, pending, report = refine(rows, CFG, reference_camera="a")
    assert report["reason"] == "explicit_exit_or_source_discontinuity"
    assert all(not any(o.exit_after and any(p.time > o.time for p in group) for o in group) for group in groups)
    assert sum(map(len, groups)) + len(pending) == len(rows)


def test_osnet_rescue_needs_multiple_features_and_preserves_confirmed_matches():
    obs = fragment("a", 1, 0, 600, embedding=(1, 0)) + fragment("b", 1, 0, 600, embedding=(0, 1))
    rows = [{"embedding_osnet": [1, 0]} for _ in obs]
    assert len(reid_fallback(obs, rows, CFG)) == 1
    # A single crop on each side cannot create a rescue match.
    rows = [{"embedding_osnet": [1, 0]} if o.frame == 0 else {} for o in obs]
    assert len(reid_fallback(obs, rows, CFG)) == 2
