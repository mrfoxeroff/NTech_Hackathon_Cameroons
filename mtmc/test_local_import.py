"""Contract, uncertainty, artifact and failure tests for the integrated local stage."""

import json
import subprocess
import sys
from types import SimpleNamespace
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from associate import DEFAULTS, Observation, associate, export_tracks, summarize
from import_local_tracking import artifact_path, crop_weight, import_outputs
from local_tracking.artifacts import write_json


def make_input(tmp_path, camera="cam_1", recording="session", untracked=False, empty=False):
    root = tmp_path / camera
    root.mkdir()
    crop = np.full((60, 30, 3), 150, np.uint8)
    cv2.imwrite(str(root / "crop.jpg"), crop)
    cv2.imwrite(str(root / "mask.png"), np.full((60, 30), 255, np.uint8))
    manifest = {"schema_version": "1.0", "status": "complete", "recording_id": recording,
                "camera_id": camera, "timeline_origin": "test", "time_offset_sec": 0,
                "clock_anchors": [], "source": {"path": "unused.mkv", "width": 100, "height": 100,
                                                 "time_base": 0.001}}
    frames = []
    for index in range(3):
        obs = {"observation_id": f"{recording}:{camera}:{index}:0", "local_track_id": None if untracked else 1,
               "bbox_xyxy": [10, 5, 40, 65], "detection_confidence": 0.9,
               "silhouette_polygons": [[[10, 5], [40, 5], [40, 65], [10, 65]]],
               "mask_ref": "mask.png", "mask_origin_px": [10, 5], "mask_size": [30, 60],
               "crop_ref": "crop.jpg", "crop_quality": {"width": 30, "height": 60,
                    "laplacian_variance": 200, "truncated": False, "selection_score": 30},
               "footpoint_px": [25, 65], "footpoint_method": "bbox_bottom_center",
               "touches_frame_bottom": False, "border_flags": {"bottom": False}}
        frames.append({"recording_id": recording, "camera_id": camera, "frame_index": index * 2,
                       "video_time_sec": index * 0.1, "timestamp_sec": index * 0.1,
                       "timestamp_source": "pts", "media_time_sec": 7 + index * 0.1,
                       "source_gap_before_sec": 0, "frame_width": 100, "frame_height": 100,
                       "observations": [] if empty and index == 1 else [obs]})
    count = sum(len(f["observations"]) for f in frames)
    tracks = [] if untracked else [{"local_track_id": 1, "observation_count": count, "selected_crops": []}]
    write_json(root / "manifest.json", manifest)
    write_json(root / "metrics.json", {"processed_frames": 3, "observations": count,
               "empty_frames": int(empty), "local_track_count": len(tracks)})
    write_json(root / "local_tracks.json", {"tracks": tracks})
    write_frames(root, frames)
    return root


def write_frames(root, frames):
    (root / "frames.jsonl").write_text("".join(json.dumps(f) + "\n" for f in frames), encoding="utf-8")


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def encoder(crop):
    assert crop.shape == (60, 30, 3)
    return [1.0, 0.0]


def test_bridge_two_cameras_joins_and_preserves_source_keys(tmp_path):
    roots = [make_input(tmp_path, camera) for camera in ("cam_1", "cam_2")]
    output = tmp_path / "combined"
    result = import_outputs(roots, output, encoder, {"overlap_pairs": [["cam_1", "cam_2"]]}, run_association=True)
    assert result["status"] == "complete" and result["total_tracks"] == 1
    rows = read_rows(output / "observations.jsonl")
    assert len(rows) == 6
    assert [r["frame_index"] for r in rows[:3]] == [0, 2, 4]
    assert [r["source_pts"] for r in rows[:3]] == [7000, 7100, 7200]
    assert rows[0]["silhouette_polygons"] and rows[0]["mask_ref"]
    assert 0 < rows[0]["appearance_weight"] < 1
    assert rows[0]["torso_brightness"] == pytest.approx(150 / 255)
    assert len(read_rows(output / "assignments.jsonl")) == 6


def test_no_null_global_track_and_empty_frames_are_preserved(tmp_path):
    root = make_input(tmp_path, untracked=True, empty=True)
    output = tmp_path / "combined"
    result = import_outputs([root], output, encoder, run_association=True)
    assert result["total_tracks"] == 0
    assert len(read_rows(output / "unassigned.jsonl")) == 2
    assert len(read_rows(output / "frames.jsonl")) == 3
    assert not read_rows(output / "observations.jsonl")


def test_piecewise_common_time_does_not_replace_original_pts(tmp_path):
    root = make_input(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["clock_anchors"] = [{"video_time_sec": 0, "timestamp_sec": 10},
                                  {"video_time_sec": 1, "timestamp_sec": 12}]
    write_json(root / "manifest.json", manifest)
    frames = read_rows(root / "frames.jsonl")
    for frame in frames:
        frame["timestamp_sec"] = 10 + 2 * frame["video_time_sec"]
    write_frames(root, frames)
    output = tmp_path / "combined"
    import_outputs([root], output, encoder)
    rows = read_rows(output / "observations.jsonl")
    assert [r["timestamp_ms"] for r in rows] == [10000, 10200, 10400]
    assert [r["source_pts"] for r in rows] == [7000, 7100, 7200]


def test_gap_on_empty_frame_reaches_next_observation(tmp_path):
    root = make_input(tmp_path, empty=True)
    frames = read_rows(root / "frames.jsonl")
    frames[1]["source_gap_before_sec"] = 10.0
    write_frames(root, frames)
    output = tmp_path / "combined"
    result = import_outputs([root], output, encoder)
    rows = read_rows(output / "observations.jsonl")
    assert [r["source_segment"] for r in rows] == [0, 1]
    assert not any(r.get("confirmed_global_exit") for r in rows)
    assert result["cameras"]["cam_1"]["stats"]["source_gaps"] == 1


@pytest.mark.parametrize("change", ["failed", "recording", "origin", "duplicate"])
def test_reject_incompatible_or_partial_runs_before_writing(tmp_path, change):
    first, second = make_input(tmp_path), make_input(tmp_path, "cam_2")
    path = second / "manifest.json"
    manifest = json.loads(path.read_text())
    if change == "failed":
        manifest["status"] = "failed"
    elif change == "origin":
        manifest["timeline_origin"] = "other_epoch"
    elif change == "recording":
        manifest["recording_id"] = "other_session"
    write_json(path, manifest)
    output = tmp_path / "combined"
    with pytest.raises(ValueError):
        import_outputs([first, first if change == "duplicate" else second], output, encoder)
    assert not output.exists()


def test_does_not_overwrite_previous_result(tmp_path):
    root = make_input(tmp_path)
    output = tmp_path / "combined"
    import_outputs([root], output, encoder)
    before = (output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        import_outputs([root], output, encoder)
    assert (output / "manifest.json").read_bytes() == before


def test_encoder_failure_is_not_reported_as_complete(tmp_path):
    root = make_input(tmp_path)
    output = tmp_path / "combined"
    def broken(crop):
        raise RuntimeError("test failure")
    with pytest.raises(RuntimeError, match="test failure"):
        import_outputs([root], output, broken)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["error"]["type"] == "RuntimeError"


def test_quality_filter_keeps_observations_without_reid(tmp_path):
    root = make_input(tmp_path)
    output = tmp_path / "combined"
    import_outputs([root], output, encoder, min_quality=1)
    rows = read_rows(output / "observations.jsonl")
    assert len(rows) == 3 and not any("embedding" in r for r in rows)


def test_unreadable_mask_rejected_before_import(tmp_path):
    root = make_input(tmp_path)
    (root / "mask.png").write_bytes(b"broken")
    with pytest.raises(ValueError, match="Unreadable"):
        import_outputs([root], tmp_path / "combined", encoder)


def test_no_path_escape(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        artifact_path(tmp_path, "../outside.jpg")


def test_truncated_crop_gets_lower_weight(tmp_path):
    root = make_input(tmp_path)
    obs = read_rows(root / "frames.jsonl")[0]["observations"][0]
    clean = crop_weight(obs)
    obs["crop_quality"]["truncated"] = True
    assert crop_weight(obs) == clean / 2


def test_zero_features_fail(tmp_path):
    root = make_input(tmp_path)
    with pytest.raises(ValueError, match="zero vector"):
        import_outputs([root], tmp_path / "combined", lambda crop: [0.0, 0.0])


def sample(camera, ident, t, segment=0, vector=(1, 0), weight=1):
    return Observation.parse({"camera_id": camera, "local_id": ident, "frame_index": t,
        "timestamp_ms": t, "embedding": vector, "appearance_weight": weight,
        "source_segment": segment, "bbox_xyxy": [t / 10, 0, t / 10 + 50, 200]})


def test_bad_crop_cannot_dominate_gallery():
    rows = [sample("a", 1, 0), sample("a", 1, 100, vector=(0, 1), weight=0.05)]
    v = summarize(rows, "reid")
    assert v[0] > 0.99
    assert summarize([replace(o, appearance_weight=0) for o in rows], "reid") is None


def test_source_gap_cannot_be_locally_stitched():
    rows = [sample("a", 1, t) for t in range(0, 501, 100)]
    rows += [sample("a", 1, t, segment=1) for t in range(600, 1101, 100)]
    assert len(associate(rows, {"stitch_local_fragments": True})) == 2


def test_other_camera_may_bridge_gap_without_hiding_visibility_gap(tmp_path):
    rows = [sample("a", 1, t, segment=int(t >= 600)) for t in range(0, 1101, 100)]
    rows += [sample("b", 1, t) for t in range(0, 1101, 100)]
    groups = associate(rows, {"overlap_pairs": [["a", "b"]], "stitch_local_fragments": True})
    assert len(groups) == 1
    export_tracks(groups, tmp_path / "result", 350)
    track = json.loads((tmp_path / "result/tracks.json").read_text())["tracks"][0]
    assert track["cameras"]["a"] == [[0, 500], [600, 1100]]


def test_no_map_coordinates_invented_from_bbox_footpoint(tmp_path):
    root = make_input(tmp_path)
    output = tmp_path / "combined"
    import_outputs([root], output, encoder)
    assert all("map_xy" not in r for r in read_rows(output / "observations.jsonl"))


@pytest.mark.parametrize("failure", [False, True])
def test_model_path_with_apostrophe_is_local_and_restores_cwd(tmp_path, monkeypatch, failure):
    import ultralytics
    from local_tracking.pipeline import load_model

    directory = tmp_path / "AI's_directory"
    directory.mkdir()
    weights = directory / "existing-seg.pt"
    weights.write_bytes(b"not actual weights; constructor mocked")
    previous = Path.cwd()
    def constructor(name):
        assert name == weights.name and Path(name).is_file()
        if failure:
            raise RuntimeError("constructor failed")
        return SimpleNamespace(task="segment", names={0: "person"})
    monkeypatch.setattr(ultralytics, "YOLO", constructor)
    settings = SimpleNamespace(model=weights)
    if failure:
        with pytest.raises(RuntimeError, match="constructor failed"):
            load_model(settings)
    else:
        assert load_model(settings)[1] == 0
    assert Path.cwd() == previous


def test_nonmonotonic_clock_anchors_rejected(tmp_path):
    root = make_input(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["clock_anchors"] = [{"video_time_sec": 0, "timestamp_sec": 1},
                                  {"video_time_sec": 1, "timestamp_sec": 0}]
    write_json(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="increase"):
        import_outputs([root], tmp_path / "combined", encoder)


def test_association_cli_rejects_failed_import(tmp_path):
    root = tmp_path / "failed"
    root.mkdir()
    write_json(root / "manifest.json", {"stage": "local_import", "status": "failed"})
    (root / "observations.jsonl").write_text("")
    config = tmp_path / "config.json"
    write_json(config, {})
    result = subprocess.run([sys.executable, str(Path(__file__).with_name("associate.py")),
        str(root / "observations.jsonl"), "--config", str(config), "--output", str(tmp_path / "result")],
        capture_output=True, text=True)
    assert result.returncode == 2 and "incomplete or failed" in result.stderr
    assert not (tmp_path / "result").exists()
