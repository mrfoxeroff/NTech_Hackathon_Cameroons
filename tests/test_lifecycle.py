from mtmc.associate import DEFAULTS, Observation
from mtmc.lifecycle import review_lifecycle


def fragment(ident, height, length):
    return [
        Observation.parse(
            {
                "camera_id": "camera_3",
                "local_id": ident,
                "frame_index": ident * 1000 + index,
                "timestamp_ms": ident * 100000 + index * 120,
                "bbox_xyxy": [100, 100, 100 + height / 2, 100 + height],
                "confidence": 0.9,
                "map_xy": [302, 209],
            }
        )
        for index in range(length)
    ]


def test_confirmed_ids_require_lifetime_and_consistent_contour_support():
    config = DEFAULTS | {
        "reappearance_memory_ms": 60000,
        "visibility_gap_ms": 350,
        "lifecycle": {
            "min_ms": 1750,
            "far_ms": 2850,
            "far_height_px": 170,
            "near_support": 5,
            "far_support": 9,
        },
    }
    short = fragment(1, 400, 6)
    short_far = fragment(2, 100, 19)
    partial_far = fragment(3, 100, 30)
    good = fragment(4, 100, 30)
    details = {
        o: {
            "footpoint_status": "observed",
            "border_flags": {"top": o.local_id == 3},
            "silhouette_polygons": [[[100, 100], [150, 100], [150, 195], [100, 195]]],
        }
        for o in short + short_far + partial_far + good
    }
    confirmed, pending, report = review_lifecycle([short, short_far, partial_far, good], details, config)
    assert confirmed == [good]
    assert pending == short + short_far + partial_far
    assert [item["reason"] for item in report["tentative"]] == [
        "short_unconfirmed",
        "short_distant",
        "weak_distant_contour",
    ]
