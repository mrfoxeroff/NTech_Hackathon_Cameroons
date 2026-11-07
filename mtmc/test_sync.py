"""Clock reconstruction must not turn PTS jumps into physical ten-second gaps."""

import pytest

from build_clock_alignment import build_camera, interpolate
from local_tracking.config import CameraConfig
from render_synced import nearest


def fixture(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"clock test fixture")
    rows = [{"frame_index": i, "source_pts": round((i / 10 + (10 if i >= 4 else 0)) * 1000),
             "video_time_sec": i / 10 + (10 if i >= 4 else 0), "osd_sec": i // 3} for i in range(9)]
    audit = {"source": str(source), "decoded_frames": len(rows), "errors": [],
             "first_media_time_sec": 0, "transitions": [{"frame_index": i, "timestamp_sec": i // 3}
                                                         for i in (3, 6)],
             "media_gaps": [{"frame_index": 4, "gap_sec": 10.1}]}
    return audit, rows


def test_discontinuity_maps_adjacent_frames_to_adjacent_clock_times(tmp_path):
    audit, rows = fixture(tmp_path)
    mapping = build_camera(audit, rows)
    camera = CameraConfig("test", tmp_path / "source.bin", clock_anchors=mapping["clock_anchors"])
    assert camera.common_time(10.4) - camera.common_time(.3) == pytest.approx(1 / 3)
    for row in rows:
        assert camera.common_time(row["video_time_sec"]) == pytest.approx(row["frame_index"] / 3)


def test_piecewise_map_can_be_inverted(tmp_path):
    audit, rows = fixture(tmp_path)
    mapping = build_camera(audit, rows)
    xs = [a["video_time_sec"] for a in mapping["clock_anchors"]]
    ys = [a["timestamp_sec"] for a in mapping["clock_anchors"]]
    for value in (0, .15, .3, 10.4, 10.75):
        assert interpolate(interpolate(value, xs, ys), ys, xs) == pytest.approx(value)


def test_opening_second_longer_than_next_starts_on_its_displayed_clock(tmp_path):
    audit, rows = fixture(tmp_path)
    rows.pop()
    audit["decoded_frames"] = len(rows)
    seconds = [1, 1, 1, 1, 2, 2, 3, 3]
    for row, second in zip(rows, seconds):
        row["osd_sec"] = second
    audit["transitions"] = [{"frame_index": i, "timestamp_sec": seconds[i]} for i in (4, 6)]

    mapping = build_camera(audit, rows)
    assert mapping["common_start_sec"] == 1
    camera = CameraConfig("test", tmp_path / "source.bin", clock_anchors=mapping["clock_anchors"])
    assert [int(camera.common_time(row["video_time_sec"])) for row in rows] == seconds


def test_uncertain_clock_not_silently_accepted(tmp_path):
    audit, rows = fixture(tmp_path)
    audit["errors"] = [{"frame_index": 4}]
    with pytest.raises(ValueError, match="uncertain"):
        build_camera(audit, rows)


def test_missing_real_clock_seconds_need_review(tmp_path):
    audit, rows = fixture(tmp_path)
    audit["transitions"][1]["timestamp_sec"] = 4
    with pytest.raises(ValueError, match="missing"):
        build_camera(audit, rows)


def test_missing_source_frame_needs_review(tmp_path):
    audit, rows = fixture(tmp_path)
    rows[4]["frame_index"] = 40
    with pytest.raises(ValueError, match="every source frame"):
        build_camera(audit, rows)


def test_resampling_uses_nearest_real_frame():
    frames = iter([(0, "a"), (.1, "b"), (.2, "c")])
    assert list(nearest(frames, [.01, .09, .19])) == [(0, "a"), (.1, "b"), (.2, "c")]
