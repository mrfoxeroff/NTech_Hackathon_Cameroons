from dataclasses import replace

from mtmc.associate import Observation
from mtmc.doorways import restore_door_returns

CONFIG = {
    "reappearance_memory_ms": 60000,
    "visibility_gap_ms": 350,
    "door_return_zones": {
        "left": {"camera_3": [0, 0, 12, 20], "camera_5": [0, 0, 12, 20]},
        "right": {"camera_3": [13, 0, 24, 20]},
    },
}


def fragment(ident, begin, camera="camera_3", x=10, embedding=(1, 0)):
    return [
        Observation.parse(
            {
                "camera_id": camera,
                "local_id": ident,
                "frame_index": timestamp // 100,
                "timestamp_ms": timestamp,
                "bbox_xyxy": [100 + x, 30, 150 + x, 160],
                "map_xy": [x, 10],
                "embedding": embedding,
            }
        )
        for timestamp in range(begin, begin + 400, 100)
    ]


def test_recovery_across_cameras_uses_the_last_global_sighting_and_renews_memory():
    departure = fragment(1, 0) + fragment(1, 100, "camera_5")
    entry = fragment(2, 60400, "camera_3")
    second_entry = fragment(3, 120700, "camera_5")
    groups, links = restore_door_returns([departure, entry, second_entry], CONFIG)
    assert groups == [departure + entry + second_entry]
    assert [link["gap_ms"] for link in links] == [60000, 60000]


def test_different_door_expired_id_and_different_appearance_are_rejected():
    departure = fragment(1, 0)
    for arrival in (
        fragment(2, 60400),
        fragment(2, 45000, x=14),
        fragment(2, 45000, embedding=(0, 1)),
    ):
        groups, links = restore_door_returns([departure, arrival], CONFIG)
        assert groups == [departure, arrival]
        assert not links


def test_simultaneous_equally_plausible_arrivals_do_not_steal_old_id():
    departure = fragment(1, 0)
    a, b = fragment(2, 45000), fragment(3, 45200, x=11)
    groups, links = restore_door_returns([departure, a, b], CONFIG)
    assert groups == [departure, a, b]
    assert not links


def test_recovery_with_no_office_zones_or_calibration():
    config = CONFIG | {"door_return_zones": {}}
    departure = [replace(obs, xy=None) for obs in fragment(1, 0)]
    arrival = [replace(obs, xy=None) for obs in fragment(2, 60300)]
    groups, links = restore_door_returns([departure, arrival], config)
    assert groups == [departure + arrival]
    assert links[0]["gap_ms"] == 60000
    other_view = [replace(obs, camera="camera_5") for obs in arrival]
    assert restore_door_returns([departure, other_view], config)[1] == []


def test_recovery_between_verified_overlapping_views_without_office_zones():
    departure = fragment(1, 0)
    arrival = fragment(2, 60000, "camera_5", x=15)
    config = CONFIG | {"door_return_zones": {}, "overlap_pairs": [["camera_3", "camera_5"]]}
    assert restore_door_returns([departure, arrival], config)[0] == [departure + arrival]
    assert restore_door_returns([departure, arrival], config | {"overlap_pairs": []})[1] == []
