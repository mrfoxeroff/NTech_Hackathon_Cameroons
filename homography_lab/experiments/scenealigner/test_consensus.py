import numpy as np
from temporal_consensus import cluster_candidates, floor_probes


def candidate(frame, x, score=0.5):
    return {"case_id": frame, "score": score, "signature": [[x, 0.0], [x + 1.0, 1.0]]}


def test_many_candidates_from_one_frame_cannot_outvote_three_distinct_frames():
    items = [candidate("a", 100 + i * 0.01, 0.99) for i in range(10)]
    items += [candidate("b", 0), candidate("c", 0.1), candidate("d", -0.1)]
    result = cluster_candidates(items, 1.0, 4)
    assert result[0]["support_frames"] == 3
    assert set(result[0]["support_case_ids"]) == {"b", "c", "d"}
    assert result[0]["total_frames"] == 4


def test_every_retained_member_is_within_final_medoid_radius():
    items = [candidate(str(i), x) for i, x in enumerate([-0.9, 0.0, 0.9, 1.6, 2.3])]
    result = cluster_candidates(items, 1.0, 5)
    assert result
    assert all(item["max_rms_px"] <= 1.0 + 1e-9 for item in result)
    assert all(len(set(item["support_case_ids"])) == item["support_frames"] for item in result)


def test_probes_are_derived_from_largest_floor_component_and_avoid_holes():
    mask = np.zeros((640, 800), np.uint8)
    mask[50:590, 80:720] = 255
    mask[200:400, 300:500] = 0
    mask[0:20, 0:20] = 255
    points = floor_probes(mask, count=16).astype(int)
    assert len(np.unique(points, axis=0)) == 16
    assert (mask[points[:, 1], points[:, 0]] == 255).all()
    assert (points[:, 0] > 50).all() and (points[:, 1] > 30).all()
