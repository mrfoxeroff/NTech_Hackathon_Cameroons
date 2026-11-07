"""Export a completed MTMC run as a read-only web session with synced MP4s."""

import argparse
import json
import math
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

import av
import cv2
import numpy as np
from PIL import Image

from local_tracking.artifacts import write_json
from local_tracking.config import safe_id
from local_tracking.pipeline import file_hash
from local_tracking.video import iter_video
from mtmc.associate import DEFAULTS, export_tracks
from mtmc.identity_consistency import associate_consistently
from tools.world_state import STEP_MS, build_world_state

COLORS = ("#38bdf8", "#fb7185", "#4ade80", "#facc15", "#c084fc", "#fb923c")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(path):
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def observation_key(row):
    return row["camera_id"], row["local_id"], row["frame_index"]


def load_calibration(directory, camera, source, plan_size):
    number = camera.removeprefix("camera_").removeprefix("cam_")
    data = read_json(directory / f"cam_{number}_calibration.json")
    if (
        data["camera_size"] != [source["width"], source["height"]]
        or data["plan_size"] != list(plan_size)
        or data.get("lens_correction_applied")
    ):
        raise ValueError(f"Calibration size mismatch for {camera}")
    matrix = np.asarray(data["homography"], dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) != 3:
        raise ValueError(f"Invalid homography for {camera}")
    return matrix


def project(row, matrix, size):
    point = row.get("footpoint_px")
    status = row.get("footpoint_status")
    if status == "unknown" or point is None or row.get("touches_frame_bottom"):
        return None
    x, y, scale = matrix @ [*point, 1]
    if not np.isfinite([x, y, scale]).all() or abs(scale) < 1e-9:
        return None
    x, y = x / scale, y / scale
    if 0 <= x < size[0] and 0 <= y < size[1]:
        return {"x": round(x, 2), "y": round(y, 2), "status": status or "observed"}
    return None


def color_for(ident):
    return COLORS[(ident - 1) % len(COLORS)]


def build_tracks(track_data, points, gap_sec):
    result = {}
    for track in track_data["tracks"]:
        ident = track["global_id"]
        segments = sorted(
            (
                {"cam": camera, "start": a / 1000, "end": b / 1000}
                for camera, intervals in track["cameras"].items()
                for a, b in intervals
            ),
            key=lambda item: (item["start"], item["cam"]),
        )
        path = sorted(points.get(ident, {}).values(), key=lambda point: point["t"])
        for previous, current in zip(path, path[1:]):
            current["gap_before"] = current["t"] - previous["t"] > gap_sec or (
                current["cam"] == previous["cam"] and current["segment"] != previous["segment"]
            )
        for point in path:
            point.pop("segment")
        result[str(ident)] = {
            "id": ident,
            "color": color_for(ident),
            "start_time": track["start_ms"] / 1000,
            "end_time": track["end_ms"] / 1000,
            "camera_segments": segments,
            "cameras_visited": list(dict.fromkeys(segment["cam"] for segment in segments)),
            "map_path": path,
        }
    if len(result) != track_data["total_tracks"]:
        raise ValueError("Inconsistent global track count")
    return result


def render_camera(camera, local_frames, manifest, annotations, path):
    source = manifest["source"]
    video = Path(source["path"])
    start = local_frames[0][2]
    width = min(1280, source["width"]) // 2 * 2
    height = round(source["height"] * width / source["width"] / 2) * 2
    scale = np.array([width / source["width"], height / source["height"]] * 2)
    fps = max(1, round(source["fps"] / manifest["frame_stride"]))
    time_base = Fraction(1, 1000)
    coverage = []
    last_pts = -1
    frames = iter_video(
        video,
        start_sec=local_frames[0][1],
        end_sec=local_frames[-1][1] + 0.001,
        stride=manifest["frame_stride"],
    )
    with av.open(str(path), "w", options={"movflags": "+faststart"}) as output:
        stream = output.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.time_base = stream.codec_context.time_base = time_base
        stream.options = {"preset": "fast", "crf": "23", "tune": "zerolatency", "threads": "3"}
        for frame, (frame_index, _, t) in zip(frames, local_frames, strict=True):
            if frame.index != frame_index:
                raise ValueError(f"Source frame index mismatch for {camera}")
            if not coverage or t - coverage[-1][1] > 0.35:
                coverage.append([t, t])
            coverage[-1][1] = t
            image = cv2.resize(frame.image, (width, height))
            for ident, box in annotations.get(frame.index, []):
                left, top, right, bottom = np.rint(np.asarray(box) * scale).astype(int)
                rgb = bytes.fromhex(color_for(ident)[1:])
                color = tuple(reversed(rgb))
                cv2.rectangle(image, (left, top), (right, bottom), color, 2)
                cv2.putText(image, f"ID {ident}", (left, max(16, top - 5)), 0, 0.65, color, 2)
            encoded = av.VideoFrame.from_ndarray(image, format="bgr24")
            last_pts = max(last_pts + 1, round((t - start) * 1000))
            encoded.pts, encoded.time_base = last_pts, time_base
            for packet in stream.encode(encoded):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return {"url": f"/media/{path.name}", "start": start, "end": coverage[-1][1], "coverage": coverage}


def transcode_camera(original, stream, source, path, offset):
    """Encode browser-compatible H.264 while retaining every source PTS."""
    stream.thread_type = "AUTO"
    stream.codec_context.thread_count = 4
    encoders = (
        ("h264_nvenc", {"preset": "p3", "rc": "constqp", "qp": "23"}),
        ("libx264", {"preset": "veryfast", "crf": "23", "tune": "zerolatency", "threads": "4"}),
    )
    for codec, options in encoders:
        if codec == "h264_nvenc" and codec not in av.codecs_available:
            continue
        with av.open(str(path), "w", options={"movflags": "+faststart"}) as output:
            target = output.add_stream(codec, rate=stream.average_rate or Fraction(str(source["fps"])))
            target.width, target.height, target.pix_fmt = source["width"], source["height"], "yuv420p"
            target.time_base = target.codec_context.time_base = stream.time_base
            target.codec_context.max_b_frames = 0
            target.options = options
            try:
                target.codec_context.open()
            except av.FFmpegError:
                if codec == "h264_nvenc":
                    continue
                raise
            for frame in original.decode(stream):
                if frame.pts is None:
                    return False
                encoded = frame.reformat(format="yuv420p")
                encoded.pts = frame.pts - offset
                encoded.time_base = frame.time_base
                for packet in target.encode(encoded):
                    output.mux(packet)
            for packet in target.encode():
                output.mux(packet)
        return True


def package_camera(camera, frames, manifest, annotations, path):
    source = manifest["source"]
    video = Path(source["path"])
    if file_hash(video) != manifest["source_sha256"]:
        raise ValueError(f"Source video changed for {camera}")
    with av.open(str(video)) as original:
        stream = original.streams.video[0]
        if source.get("first_frame_time_sec") is None:
            return render_camera(camera, frames, manifest, annotations, path)
        offset = round(source["first_frame_time_sec"] / stream.time_base)
        if stream.codec_context.name != "h264" or stream.codec_context.has_b_frames:
            if not transcode_camera(original, stream, source, path, offset):
                return render_camera(camera, frames, manifest, annotations, path)
        else:
            with av.open(str(path), "w", options={"movflags": "+faststart"}) as output:
                target = output.add_stream_from_template(stream)
                for packet in original.demux(stream):
                    if packet.size == 0:
                        continue
                    if packet.dts is None or packet.pts is None:
                        raise ValueError(f"Cannot remux missing source timestamps for {camera}")
                    packet.pts -= offset
                    packet.dts -= offset
                    packet.stream = target
                    output.mux(packet)

    overlay = path.with_name(camera + "_overlay.json")
    write_json(
        overlay,
        {
            "width": source["width"],
            "height": source["height"],
            "frames": [
                [media, [[ident, *box] for ident, box in annotations.get(index, [])]]
                for index, media, _ in frames
            ],
        },
    )
    coverage = []
    for _, _, time in frames:
        if not coverage or time - coverage[-1][1] > 0.35:
            coverage.append([time, time])
        coverage[-1][1] = time
    return {
        "url": f"/media/{path.name}",
        "overlay_url": f"/media/{overlay.name}",
        "start": coverage[0][0],
        "end": coverage[-1][1],
        "coverage": coverage,
        "clock_anchors": manifest["clock_anchors"],
        "time_offset_sec": manifest["time_offset_sec"],
    }


def session_sources(global_root, plan_size, calibrations):
    manifest = read_json(global_root / "manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("Global run must be complete")
    cameras = {}
    for camera, info in manifest["cameras"].items():
        safe_id(camera)
        root = Path(info["input"])
        local = read_json(root / "manifest.json")
        if (
            local["status"] != "complete"
            or local["camera_id"] != camera
            or local["recording_id"] != manifest["recording_id"]
            or local["timeline_origin"] != manifest["timeline_origin"]
            or file_hash(root / "frames.jsonl") != info["frames_sha256"]
        ):
            raise ValueError(f"Local artifacts no longer match global run: {camera}")
        cameras[camera] = (root, local, load_calibration(calibrations, camera, local["source"], plan_size))
    if not cameras:
        raise ValueError("Global run contains no cameras")
    return manifest, cameras


def repair_associations(global_root, plan, calibrations):
    """Assign global IDs to a completed observation import."""
    with Image.open(plan) as image:
        size = image.size
    manifest, cameras = session_sources(global_root, size, calibrations)
    if (global_root / "assignments.jsonl").exists():
        raise FileExistsError("Use a new import directory for repaired associations")
    observations = list(rows(global_root / "observations.jsonl"))
    groups, pending, report = associate_consistently(
        observations, manifest["settings"], {camera: data[2] for camera, data in cameras.items()}, size
    )
    gap = (DEFAULTS | manifest["settings"])["visibility_gap_ms"]
    with TemporaryDirectory(prefix=".association-", dir=global_root) as temporary:
        staging = Path(temporary)
        export_tracks(groups, staging, gap)
        with (staging / "assignments.jsonl").open("a", encoding="utf-8") as stream:
            for obs in pending:
                stream.write(
                    json.dumps(
                        {
                            "camera_id": obs.camera,
                            "local_id": obs.local_id,
                            "frame_index": obs.frame,
                            "timestamp_ms": obs.time,
                            "global_id": None,
                            "status": "identity_uncertain",
                        }
                    )
                    + "\n"
                )
        for name in ("tracks.json", "tracks.csv", "assignments.jsonl"):
            (staging / name).replace(global_root / name)
    manifest.update(total_tracks=len(groups), identity_consistency=report)
    write_json(global_root / "manifest.json", manifest)
    return report


def export_session(global_root, output, plan, calibrations):
    if output.exists():
        raise FileExistsError(output)
    with Image.open(plan) as image:
        plan_size = image.size
    manifest, cameras = session_sources(global_root, plan_size, calibrations)
    frame_catalog = {}
    for camera, (root, _, _) in cameras.items():
        frame_catalog[camera] = [
            (row["frame_index"], row["video_time_sec"], row["timestamp_sec"])
            for row in rows(root / "frames.jsonl")
        ]
        if not frame_catalog[camera]:
            raise ValueError(f"Missing source frames for {camera}")
    frame_times = {
        (camera, index): round(time * 1000)
        for camera, frames in frame_catalog.items()
        for index, _, time in frames
    }
    assignments = {}
    for item in rows(global_root / "assignments.jsonl"):
        key = observation_key(item)
        if key in assignments:
            raise ValueError(f"Duplicate assignment: {key}")
        assignments[key] = (item["global_id"], item["timestamp_ms"])
    track_data = read_json(global_root / "tracks.json")
    valid_ids = {track["global_id"] for track in track_data["tracks"]}
    annotations = defaultdict(lambda: defaultdict(list))
    world_observations = defaultdict(dict)
    for row in rows(global_root / "observations.jsonl"):
        key = observation_key(row)
        ident, timestamp = assignments.pop(key)
        if (
            frame_times.get((row["camera_id"], row["frame_index"])) != timestamp
            or row["timestamp_ms"] != timestamp
        ):
            raise ValueError(f"Assignment is out of sync with its source frame: {key}")
        if ident is None:
            continue
        if ident not in valid_ids:
            raise ValueError(f"Assignment refers to missing track {ident}")
        camera = row["camera_id"]
        box = row.get("bbox_stabilized_xyxy") or row["bbox_xyxy"]
        annotations[camera][row["frame_index"]].append((ident, box))
        point = project(row, cameras[camera][2], plan_size)
        bucket = math.floor(row["timestamp_sec"] * 1000 + 1e-7) // STEP_MS
        views = world_observations[ident, bucket]
        old = views.get(camera)
        rank = (bool(point and point["status"] == "observed"), bool(point), row["confidence"])
        if old is None or rank > old[0]:
            views[camera] = (
                rank,
                {
                    "camera_id": camera,
                    "frame_index": row["frame_index"],
                    "timestamp_sec": row["timestamp_sec"],
                    "source_segment": row["source_segment"],
                    "confidence": row["confidence"],
                    "point": point,
                },
            )
    if assignments:
        raise ValueError("Global assignments contain observations missing from the import")
    gap_sec = manifest["settings"].get("visibility_gap_ms", 350) / 1000
    samples, world_state = build_world_state(
        {
            key: {camera: source for camera, (_, source) in views.items()}
            for key, views in world_observations.items()
        }
    )
    tracks = build_tracks(track_data, samples, gap_sec)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".web-session-", dir=output.parent) as temporary:
        bundle = Path(temporary) / "session"
        bundle.mkdir()
        with Image.open(plan) as image:
            image.convert("RGB").save(bundle / "map.png")
        videos = {
            camera: package_camera(
                camera, frame_catalog[camera], local, annotations[camera], bundle / f"{camera}.mp4"
            )
            for camera, (_, local, _) in cameras.items()
        }
        start = min(video["start"] for video in videos.values())
        end = max(video["end"] for video in videos.values())
        payload = {
            "schema_version": "1.0",
            "recording_id": manifest["recording_id"],
            "timeline_origin": manifest["timeline_origin"],
            "start_time": start,
            "end_time": end,
            "duration_sec": end - start,
            "total_unique_tracks": len(tracks),
            "cameras": videos,
            "map": {"url": "/media/map.png", "width": plan_size[0], "height": plan_size[1]},
            "tracks_summary": tracks,
            "state_index": {"url": "/api/state", "step_ms": STEP_MS},
            "visibility_gap_sec": gap_sec,
        }
        write_json(bundle / "world_state.json", world_state)
        write_json(bundle / "session.json", payload)
        bundle.rename(output)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("global_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--plan", type=Path, default=Path("office.jpg"))
    parser.add_argument("--calibrations", type=Path, default=Path("."))
    args = parser.parse_args()
    try:
        export_session(args.global_root, args.output, args.plan, args.calibrations)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"error: {error}\n")
