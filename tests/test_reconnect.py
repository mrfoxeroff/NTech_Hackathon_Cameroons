from dataclasses import replace

from mtmc.associate import DEFAULTS, Observation, split_tracklets
from mtmc.reconnect import reconnect, trajectory_link

CONFIG = DEFAULTS | {
    "reappearance_memory_ms": 60000,
    "visibility_gap_ms": 350,
    "overlap_pairs": [["camera_2", "camera_3"]],
}


def fragment(camera, ident, start, count, position, embedding=(1.0, 0.0)):
    return [
        Observation.parse(
            {
                "camera_id": camera,
                "local_id": ident,
                "frame_index": start // 100 + index,
                "timestamp_ms": start + 100 * index,
                "map_xy": position(start + 100 * index),
                "embedding": embedding,
            }
        )
        for index in range(count)
    ]


def test_reappearance_memory_expires_after_a_minute_and_requires_same_location():
    old = fragment("camera_3", 1, 0, 4, lambda _: (10, 10))
    near = fragment("camera_3", 2, 60300, 4, lambda _: (11, 10))
    expired = fragment("camera_3", 3, 60400, 4, lambda _: (11, 10))
    far = fragment("camera_3", 4, 60300, 4, lambda _: (23, 10))

    assert len(reconnect([old, near], CONFIG)[0]) == 1
    assert len(reconnect([old, expired], CONFIG)[0]) == 2
    assert len(reconnect([old, far], CONFIG)[0]) == 2
    assert len(reconnect([old[:-1] + [replace(old[-1], exit_after=True)], near], CONFIG)[0]) == 2


def test_two_nearby_people_with_equal_return_evidence_stay_separate():
    old = fragment("camera_3", 1, 0, 4, lambda _: (10, 10))
    left = fragment("camera_3", 2, 30000, 4, lambda _: (9, 10))
    right = fragment("camera_3", 3, 30000, 4, lambda _: (11, 10))
    assert len(reconnect([old, left, right], CONFIG)[0]) == 3


def test_synchronized_path_connects_views_without_inheriting_disconnected_tail():
    first = fragment("camera_3", 1, 0, 15, lambda t: (t / 50, 30))
    wrong_tail = fragment("camera_3", 2, 3500, 11, lambda t: (220 - t / 50, 30))
    other_view = fragment("camera_2", 3, 200, 46, lambda t: (t / 50, 30), (0.5, 0.866))

    groups, links, pending = reconnect([first + wrong_tail, other_view], CONFIG)
    assert set(groups[0]) == set(first + other_view)
    assert groups[1] == wrong_tail
    assert [link["mode"] for link in links] == ["trajectory"]
    assert not pending


def test_a_sparse_close_pass_with_weak_spatial_continuity_does_not_recover_id():
    old = fragment("camera_3", 1, 0, 1, lambda _: (10, 10))
    different = fragment("camera_3", 2, 500, 1, lambda _: (21, 10), (0.74, 0.67))
    assert len(reconnect([old, different], CONFIG)[0]) == 2


def test_short_tail_without_cross_view_support_is_excluded_from_identity():
    target = fragment("camera_3", 1, 0, 21, lambda t: (t / 50 if t < 1600 else 32 - (t - 1600) / 15, 10))
    reference = fragment("camera_2", 2, 0, 22, lambda t: (t / 50, 10))
    for index, brightness in ((8, 0.85), (13, 0.86), (18, 0.1)):
        target[index] = replace(target[index], brightness=brightness)
    reference = [replace(o, brightness=0.76) for o in reference]
    left, right = (split_tracklets(rows, 350)[0] for rows in (target, reference))

    _, _, _, pending = trajectory_link(left, right, 40)
    assert pending == set(target[16:])
