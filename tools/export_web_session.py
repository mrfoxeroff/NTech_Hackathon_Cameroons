"""Export a completed MTMC run as a read-only web session with synced MP4s."""

import argparse
import json
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
    if data["camera_size"] != [source["width"], source["height"]] or data["plan_size"] != list(plan_size):
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


def render_camera(camera, local_root, manifest, annotations, path):
    source = manifest["source"]
    video = Path(source["path"])
    if file_hash(video) != manifest["source_sha256"]:
        raise ValueError(f"Source video changed for {camera}")
    local_frames = [
        (row["frame_index"], row["video_time_sec"], row["timestamp_sec"])
        for row in rows(local_root / "frames.jsonl")
    ]
    if not local_frames:
        raise ValueError(f"Missing source frames for {camera}")
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


def export_session(global_root, output, plan, calibrations):
    manifest = read_json(global_root / "manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("Global run must be complete")
    if output.exists():
        raise FileExistsError(output)
    with Image.open(plan) as image:
        plan_size = image.size
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
    assignments = {}
    for item in rows(global_root / "assignments.jsonl"):
        key = observation_key(item)
        if key in assignments:
            raise ValueError(f"Duplicate assignment: {key}")
        assignments[key] = item["global_id"]
    track_data = read_json(global_root / "tracks.json")
    valid_ids = {track["global_id"] for track in track_data["tracks"]}
    annotations = defaultdict(lambda: defaultdict(list))
    samples = defaultdict(dict)
    ranks = {}
    for row in rows(global_root / "observations.jsonl"):
        key = observation_key(row)
        ident = assignments.pop(key)
        if ident is None:
            continue
        if ident not in valid_ids:
            raise ValueError(f"Assignment refers to missing track {ident}")
        camera = row["camera_id"]
        box = row.get("bbox_stabilized_xyxy") or row["bbox_xyxy"]
        annotations[camera][row["frame_index"]].append((ident, box))
        point = project(row, cameras[camera][2], plan_size)
        if point:
            # Prefer an observed footpoint, then the more confident view in each 120 ms bin.
            bucket = row["timestamp_ms"] // 120
            rank = (point["status"] == "observed", row["confidence"])
            if (ident, bucket) not in ranks or rank > ranks[ident, bucket]:
                ranks[ident, bucket] = rank
                point.update(t=row["timestamp_sec"], cam=camera, segment=row["source_segment"])
                samples[ident][bucket] = point
    if assignments:
        raise ValueError("Global assignments contain observations missing from the import")
    gap_sec = manifest["settings"].get("visibility_gap_ms", 350) / 1000
    tracks = build_tracks(track_data, samples, gap_sec)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".web-session-", dir=output.parent) as temporary:
        bundle = Path(temporary) / "session"
        bundle.mkdir()
        with Image.open(plan) as image:
            image.convert("RGB").save(bundle / "map.png")
        videos = {
            camera: render_camera(camera, root, local, annotations[camera], bundle / f"{camera}.mp4")
            for camera, (root, local, _) in cameras.items()
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
            "visibility_gap_sec": gap_sec,
        }
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
