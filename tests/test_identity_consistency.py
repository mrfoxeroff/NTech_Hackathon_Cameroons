from mtmc.associate import DEFAULTS, Observation, split_tracklets
from mtmc.identity_consistency import repair_geometry


def test_sustained_geometry_switch_repairs_two_dark_tracks_without_following_one_bad_frame():
    def detection(camera, local_id, frame, x):
        return Observation.parse(
            {
                "camera_id": camera,
                "local_id": local_id,
                "frame_index": frame,
                "timestamp_ms": 100 * frame,
                "map_xy": [x, 10],
            }
        )

    anchor_a = [detection("camera_1", 1, frame, 5) for frame in range(12)]
    anchor_b = [detection("camera_1", 2, frame, 25) for frame in range(12)]
    target_a = [detection("camera_2", 7, frame, 24 if frame == 3 or frame >= 7 else 6) for frame in range(12)]
    target_b = [detection("camera_2", 8, frame, 6 if frame == 3 or frame >= 7 else 24) for frame in range(12)]
    config = DEFAULTS | {"overlap_pairs": [["camera_1", "camera_2"]], "sync_tolerance_ms": 60}
    base = [anchor_a + target_a, anchor_b + target_b]
    fragments = split_tracklets(anchor_a + anchor_b + target_a + target_b, config["max_local_gap_ms"])
    groups, pending, changed = repair_geometry(base, fragments, "camera_1", config)
    assert not pending
    assert changed == 10
    assert set(groups[0]) == set(anchor_a + target_a[:7] + target_b[7:])
    assert set(groups[1]) == set(anchor_b + target_b[:7] + target_a[7:])
