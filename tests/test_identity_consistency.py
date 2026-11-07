from mtmc.associate import DEFAULTS, Observation, split_tracklets
from mtmc.identity_consistency import appearance_switches, nested_duplicates, repair_geometry


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


def test_short_torso_duplicate_is_removed_but_adjacent_short_person_is_kept():
    def detection(local_id, frame, box, confidence):
        return Observation.parse(
            {
                "camera_id": "camera_2",
                "local_id": local_id,
                "frame_index": frame,
                "timestamp_ms": 120 * frame,
                "bbox_xyxy": box,
                "confidence": confidence,
            }
        )

    person = [detection(1, frame, [100, 50, 160, 220], 0.94) for frame in range(7)]
    torso = [detection(2, frame, [102, 51, 159, 150], 0.72) for frame in range(2, 6)]
    neighbor = [detection(3, frame, [150, 49, 205, 162], 0.72) for frame in range(2, 6)]
    fragments = split_tracklets(person + torso + neighbor, 750)
    assert nested_duplicates(fragments, {}) == set(torso)


def test_one_sided_appearance_change_needs_another_person_nearby():
    def detection(ident, time, brightness, embedding, x):
        return Observation.parse(
            {
                "camera_id": "camera_3",
                "local_id": ident,
                "frame_index": time // 100,
                "timestamp_ms": time,
                "bbox_xyxy": [x, 40, x + 40, 160],
                "torso_brightness": brightness,
                "embedding": embedding,
            }
        )

    changed = [
        detection(1, time, brightness, embedding, 100)
        for time, brightness, embedding in (
            (0, 0.10, [1, 0]),
            (500, 0.12, [1, 0]),
            (1000, 0.64, [0, 1]),
            (1500, 0.66, [0, 1]),
        )
    ]
    neighbor = detection(2, 900, 0.65, [0, 1], 118)
    assert not appearance_switches(split_tracklets(changed, 750), crowded_switches=True)
    events = appearance_switches(split_tracklets(changed + [neighbor], 750), crowded_switches=True)
    assert len(events) == 1
    assert (events[0].old_sample, events[0].new_sample) == (500, 1000)
