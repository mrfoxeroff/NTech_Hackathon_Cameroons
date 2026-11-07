from mtmc.associate import DEFAULTS, Observation
from mtmc.corridor import reconcile_corridor


def observations(camera, ident, x, start, end, embedding):
    return [
        Observation.parse(
            {
                "camera_id": camera,
                "local_id": ident,
                "frame_index": time // 100,
                "timestamp_ms": time,
                "map_xy": [x, 220],
                "bbox_xyxy": [1000, 40, 1040, 180],
                "embedding": embedding,
            }
        )
        for time in range(start, end, 100)
    ]


def test_distant_identity_is_recovered_with_live_anchor_and_matching_appearance():
    config = DEFAULTS | {
        "distant_corridor": {
            "cameras": ["camera_3", "camera_5"],
            "map_rect": [280, 170, 330, 300],
            "image_bottom_px": 345,
        }
    }
    first = observations("camera_3", 1, 300, 0, 2400, [1, 0])
    first += observations("camera_5", 2, 292, 0, 2400, [1, 0])
    second = observations("camera_3", 3, 316, 0, 2400, [0, 1])
    second += observations("camera_5", 4, 308, 0, 1100, [0, 1])
    continuation = observations("camera_5", 5, 308, 1200, 2400, [0, 1])

    groups, pending, links = reconcile_corridor([first, second, continuation], config)
    assert groups == [first, second + continuation]
    assert not pending and links[0]["aligned_observations"] == len(continuation)

    wrong_appearance = observations("camera_5", 6, 308, 1200, 2400, [1, 0])
    groups, pending, links = reconcile_corridor([first, second, wrong_appearance], config)
    assert groups == [first, second, wrong_appearance]
    assert not pending and not links
