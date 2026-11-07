from tools.world_state import build_world_state


def source(camera, x, time, status="observed"):
    return {
        "camera_id": camera,
        "frame_index": int(time * 25),
        "timestamp_sec": time,
        "source_segment": 0,
        "confidence": 0.8,
        "point": {"x": x, "y": 50, "status": status},
    }


def test_world_state_combines_agreeing_views_and_leaves_ambiguous_positions_empty():
    views = {
        (7, 10): {"camera_1": source("camera_1", 20, 1.22), "camera_2": source("camera_2", 25, 1.29)},
        (7, 11): {"camera_1": source("camera_1", 25, 1.35), "camera_2": source("camera_2", 60, 1.38)},
        (7, 20): {"camera_1": source("camera_1", 25, 2.42), "camera_2": source("camera_2", 60, 2.45)},
    }
    tracks, index = build_world_state(views)
    assert tracks[7][10]["x"] == 22.5
    assert tracks[7][11]["x"] == 25
    assert 20 not in tracks[7] and 12 not in index["buckets"]
    assert index["buckets"][20][0]["position"] is None
    assert len(index["buckets"][20][0]["sources"]) == 2
