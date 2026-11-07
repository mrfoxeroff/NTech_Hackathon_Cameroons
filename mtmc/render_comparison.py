"""Render the reviewed ReID benchmark windows as a synchronized MP4 mosaic."""

import argparse
import json
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from associate import Observation, associate


WIDTH, HEIGHT = 1440, 640
PANEL_WIDTH, PANEL_HEIGHT = 696, 522
PALETTE = [(52, 206, 255), (191, 230, 67), (105, 129, 255), (255, 188, 104),
           (255, 136, 205), (124, 255, 165), (236, 193, 174), (117, 222, 255)]


def read_case(directory, case, settings):
    windows = {(camera, track_id): times or case["window_ms"]
               for camera, track_id, _, *times in case["labels"]}
    rows = []
    with (directory / case["observations"]).open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            limits = windows.get((row["camera_id"], row["local_id"]))
            if limits and limits[0] <= row["timestamp_ms"] < limits[1]:
                rows.append(row)
    groups = associate([Observation.parse(row) for row in rows], settings)
    assignments = {(obs.camera, obs.local_id, obs.frame): global_id
                   for global_id, group in enumerate(groups, 1) for obs in group}
    detections = defaultdict(list)
    for row in rows:
        key = (row["camera_id"], row["local_id"], row["frame_index"])
        detections[row["camera_id"], row["source_pts"]].append((assignments[key], row))
    return detections, len(groups)


def sample_frames(path, start, end, offset):
    with av.open(str(path)) as video:
        stream = video.streams.video[0]
        stream.thread_type = "AUTO"
        origin = stream.start_time * stream.time_base
        seek = stream.start_time + int(max(0, start - offset) / (1000 * stream.time_base))
        video.seek(seek, stream=stream, backward=True)
        for index, frame in enumerate(video.decode(stream)):
            time = round((frame.pts * frame.time_base - origin) * 1000) + offset
            if time >= end:
                break
            if time < start or index % 2:
                continue
            image = frame.reformat(width=PANEL_WIDTH, height=PANEL_HEIGHT, format="bgr24").to_ndarray()
            yield time, frame.pts, image, frame.width, frame.height


def nearest_frames(frames, timeline):
    previous = None
    upcoming = next(frames, None)
    for time in timeline:
        while upcoming and upcoming[0] < time:
            previous, upcoming = upcoming, next(frames, None)
        candidates = [frame for frame in (previous, upcoming) if frame is not None]
        yield min(candidates, key=lambda frame: abs(frame[0] - time)) if candidates else None


def annotate(frame, detections):
    _, _, image, width, height = frame
    if not detections:
        return image
    output = image.copy()
    shade = image.copy()
    scaled = []
    for ident, row in detections:
        color = PALETTE[(ident - 1) % len(PALETTE)]
        contour = np.rint(np.asarray(row["contour"]) * [PANEL_WIDTH / width, PANEL_HEIGHT / height]).astype(np.int32)
        cv2.fillPoly(shade, [contour], color)
        scaled.append((ident, color, contour))
    cv2.addWeighted(shade, 0.25, output, 0.75, 0, output)
    for ident, color, contour in scaled:
        cv2.polylines(output, [contour], True, color, 2, cv2.LINE_AA)
        x = min(PANEL_WIDTH - 59, max(0, int(contour[:, 0].min())))
        y = max(1, int(contour[:, 1].min()) - 24)
        cv2.rectangle(output, (x, y), (x + 57, y + 22), color, -1)
        cv2.putText(output, f"ID {ident}", (x + 5, y + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (15, 25, 35), 1, cv2.LINE_AA)
    return output


def banner(index, count, tracks, improved=False, timing_note="Проверенные фрагменты"):
    image = Image.new("RGB", (WIDTH, HEIGHT), (16, 23, 34))
    draw = ImageDraw.Draw(image)
    regular = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 17)
    bold = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 22)
    title = "Улучшенная стыковка" if improved else "Межкамерный трекинг"
    draw.text((16, 9), f"{title}  /  Intel 0288  /  Эпизод {index} из {count}", font=bold, fill="white")
    draw.text((16, 49), "КАМЕРА 1", font=regular, fill=(178, 199, 223))
    draw.text((728, 49), "КАМЕРА 2", font=regular, fill=(178, 199, 223))
    draw.text((16, 608), f"Треков в эпизоде: {tracks}    |    ID относятся к текущему эпизоду", font=regular, fill=(178, 199, 223))
    draw.text((1090, 608), timing_note, font=regular, fill=(178, 199, 223))
    return np.asarray(image)[:, :, ::-1].copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("camera_graph.pilot.json"))
    parser.add_argument("--stitching-report", type=Path, help="Verify against evaluate_stitching.py report")
    parser.add_argument("--timing-note", default="Проверенные фрагменты", help="Visible timing limitation, if known")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    manifest = json.loads(args.cases.read_text(encoding="utf-8"))
    report = json.loads((args.stitching_report or args.cases.parent / "reid-comparison.json").read_text(encoding="utf-8"))
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    if args.stitching_report and settings != report["settings"]["improved"]:
        parser.error("Config differs from the improved settings in the supplied report")
    cv2.setNumThreads(1)
    with av.open(str(args.output), "w", options={"movflags": "+faststart"}) as video:
        stream = video.add_stream("libx264", rate=Fraction(25, 2))
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        stream.options = {"preset": "fast", "crf": "21", "threads": "4"}
        for index, case in enumerate(manifest["cases"], 1):
            start, end = case["window_ms"]
            detections, count = read_case(args.cases.parent, case, settings)
            if args.stitching_report:
                expected_count = report["cases"][case["name"]]["variants"]["improved"]["total_tracks"]
            else:
                expected = report["cases"][case["name"]]["models"]["intel_0288_openvino"]["sweep"]["pilot_brightness_gate"]
                expected_count = next(trial["total_tracks"] for trial in expected if trial["threshold"] == settings["reid_min_similarity"])
            if count != expected_count:
                raise ValueError(f"Recomputed track count differs from report: {case['name']}")
            base = banner(index, len(manifest["cases"]), count, improved=args.stitching_report is not None,
                          timing_note=args.timing_note)
            timeline = range(start, end, 80)
            cameras = [nearest_frames(sample_frames(Path(manifest["cameras"][f"cam_{i}"]), start, end,
                                                     0 if i == 1 else manifest["camera_2_offset_ms"]), timeline)
                       for i in (1, 2)]
            for time, left, right in zip(timeline, *cameras, strict=True):
                canvas = base.copy()
                for cam, frame, x in (("cam_1", left, 16), ("cam_2", right, 728)):
                    if frame is not None and abs(frame[0] - time) <= 100:
                        canvas[75:75+PANEL_HEIGHT, x:x+PANEL_WIDTH] = annotate(frame, detections.get((cam, frame[1]), ()))
                stamp = f"{time // 60000:02}:{time % 60000 / 1000:05.2f}"
                cv2.putText(canvas, stamp, (1280, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (225, 235, 245), 2, cv2.LINE_AA)
                for packet in stream.encode(av.VideoFrame.from_ndarray(canvas, format="bgr24")):
                    video.mux(packet)
            print(f"{case['name']}: {len(timeline)} frames, {count} tracks", flush=True)
        for packet in stream.encode():
            video.mux(packet)


if __name__ == "__main__":
    main()
