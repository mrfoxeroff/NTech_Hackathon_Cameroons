"""Sample source-bound frames on the displayed clock, preserving PTS discontinuities."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import av
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
sys.path.insert(0, str(WORKSPACE / "mtmc"))
from audit_sync import read_clock, templates
from motion_config import load_config


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_window(path, camera, window, gallery, folder, fps):
    frames, diagnostics = [], {"ocr_failures": 0, "seconds_kept": 0, "seconds_rejected": 0, "pts_jumps": []}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.thread_count = 3
        origin = float(stream.start_time * stream.time_base)
        seek = max(0.0, window["start"] - 45.0)
        container.seek(int((origin + seek) / stream.time_base), stream=stream, backward=True)
        clock = None
        buffer = []
        full_start = False
        bad = 0
        previous_pts = None
        for index, frame in enumerate(container.decode(stream)):
            raw = float(frame.pts * frame.time_base)
            if previous_pts is not None and raw - previous_pts > 0.5:
                diagnostics["pts_jumps"].append([previous_pts - origin, raw - origin])
            previous_pts = raw
            plane = frame.planes[0]
            gray = (
                np.frombuffer(plane, np.uint8).reshape(plane.height, plane.line_size)
                if frame.format.name in ("yuv420p", "yuvj420p")
                else frame.to_ndarray(format="gray")
            )
            try:
                current, score, margin = read_clock(gray, gallery)
                if clock is not None and (current < clock or current - clock > 3):
                    raise ValueError("Nonsequential clock reading")
            except ValueError:
                diagnostics["ocr_failures"] += 1
                bad += 1
                if clock is not None:
                    buffer.append((index, frame))
                continue
            if current != clock:
                if clock is not None and window["start"] <= clock < window["end"]:
                    if full_start and current - clock == 1 and len(buffer) >= fps and bad <= 1:
                        total = index - buffer[0][0]
                        for k in range(fps):
                            selected = min(len(buffer) - 1, round(k * len(buffer) / fps))
                            number, chosen = buffer[selected]
                            image_path = folder / f"{camera}-{window['id']}-{clock:04d}-{k}.jpg"
                            cv2.imwrite(
                                str(image_path),
                                chosen.to_ndarray(format="bgr24"),
                                [cv2.IMWRITE_JPEG_QUALITY, 90],
                            )
                            frames.append(
                                {
                                    "id": image_path.stem,
                                    "camera": camera,
                                    "window": window["id"],
                                    "split": window["split"],
                                    "image": str(image_path.resolve()),
                                    "timestamp_sec": clock + (number - buffer[0][0]) / total,
                                    "osd_second": clock,
                                    "source_pts": chosen.pts,
                                    "time_base": float(chosen.time_base),
                                    "video_time_sec": float(chosen.pts * chosen.time_base) - origin,
                                    "size": [chosen.width, chosen.height],
                                    "clock_interpolation_uncertainty_sec": max(0.12, (bad + 1) / total),
                                }
                            )
                        diagnostics["seconds_kept"] += 1
                    else:
                        diagnostics["seconds_rejected"] += 1
                if current >= window["end"] + 1:
                    break
                full_start = clock is not None and current - clock == 1
                buffer, bad, clock = [], 0, current
            buffer.append((index, frame))
    return frames, diagnostics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--video-root", type=Path)
    args = p.parse_args()
    config = load_config(args.config, args.video_root, check_files=True)
    args.output.mkdir(parents=True, exist_ok=False)
    gallery = templates(Path(config["clock_templates"]))
    cv2.setNumThreads(2)
    manifest = {
        "status": "running",
        "manual_plan_landmarks_used": False,
        "global_track_IDs_used": False,
        "timeline_origin": config["timeline_origin"],
        "clock_method": "existing_fixed_OSD_templates_second_transitions_native_frame_order",
        "clock_limitation": "Interpolated displayed clocks; physical inter-camera exposure synchronization is not certified.",
        "config": config,
        "cameras": {},
        "frames": [],
    }
    try:
        for camera, source in config["cameras"].items():
            path = Path(source["video"])
            manifest["cameras"][camera] = {"source_sha256": file_hash(path), "windows": {}}
            for window in config["windows"]:
                folder = args.output / camera / window["id"]
                folder.mkdir(parents=True)
                frames, diagnostics = sample_window(
                    path, camera, window, gallery, folder, config["sample_fps"]
                )
                manifest["frames"].extend(frames)
                manifest["cameras"][camera]["windows"][window["id"]] = diagnostics
                print(camera, window["id"], len(frames), diagnostics, flush=True)
                (args.output / "frames.json").write_text(
                    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
                )
        manifest["status"] = "complete"
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        (args.output / "frames.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
