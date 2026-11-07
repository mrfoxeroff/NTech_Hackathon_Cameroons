"""Render a full two-camera OSD-aligned recording or newly processed demo clips."""

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from build_clock_alignment import interpolate
from local_tracking.pipeline import file_hash
from render_comparison import PALETTE


WIDTH, HEIGHT, PW, PH = 1440, 660, 696, 522


def source_frames(info, start, end, wanted=None):
    anchors = info["clock_anchors"]
    media = [a["video_time_sec"] for a in anchors]
    common = [a["timestamp_sec"] for a in anchors]
    local_start = interpolate(start - 0.25, common, media)
    with av.open(info["source"]) as video:
        stream = video.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.thread_count = 4
        video.seek(int(max(0, local_start + info["first_media_time_sec"]) / stream.time_base), stream=stream, backward=True)
        for frame in video.decode(video=0):
            local = float(frame.pts * frame.time_base) - info["first_media_time_sec"]
            timestamp = interpolate(local, media, common)
            if timestamp > end + 0.25:
                break
            if timestamp < start - 0.25 or (wanted is not None and frame.pts not in wanted):
                continue
            yield timestamp, frame.pts, frame


def nearest(frames, targets):
    previous, upcoming = None, next(frames, None)
    for time in targets:
        while upcoming is not None and upcoming[0] < time:
            previous, upcoming = upcoming, next(frames, None)
        choices = [f for f in (previous, upcoming) if f is not None]
        yield min(choices, key=lambda f: abs(f[0] - time)) if choices else None


def annotate(image, rows, width, height):
    output, overlay = image.copy(), image.copy()
    contours = []
    for ident, row in rows:
        color = PALETTE[(ident - 1) % len(PALETTE)] if ident is not None else (150, 150, 150)
        polygons = row.get("silhouette_polygons") or [row.get("contour", [])]
        polygons = [np.rint(np.asarray(p) * [PW / width, PH / height]).astype(np.int32) for p in polygons if len(p) >= 3]
        if polygons:
            cv2.fillPoly(overlay, polygons, color)
        contours.append((ident, row, color, polygons))
    cv2.addWeighted(overlay, 0.25, output, 0.75, 0, output)
    for ident, row, color, polygons in contours:
        if polygons:
            cv2.polylines(output, polygons, True, color, 2, cv2.LINE_AA)
        box = row["bbox_xyxy"]
        x, y = max(0, min(PW - 75, round(box[0] * PW / width))), max(0, round(box[1] * PH / height) - 24)
        cv2.rectangle(output, (x, y), (x + 73, y + 23), color, -1)
        cv2.putText(output, f"ID {ident}" if ident is not None else "ID ?", (x + 5, y + 17), cv2.FONT_HERSHEY_SIMPLEX, .55, (20, 25, 30), 1, cv2.LINE_AA)
    return output


def frame_catalog(case):
    root = Path(case["global_output"])
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Demo contains an incomplete import")
    assignments = {(r["camera_id"], r["local_id"], r["frame_index"]): r["global_id"]
                   for line in (root / "assignments.jsonl").read_text().splitlines() if (r := json.loads(line))}
    detections, wanted = defaultdict(list), defaultdict(set)
    for camera, info in manifest["cameras"].items():
        local_root = Path(info["input"])
        source = json.loads((local_root / "manifest.json").read_text())["source"]
        for line in (local_root / "frames.jsonl").read_text().splitlines():
            frame = json.loads(line)
            wanted[camera].add(round(frame["media_time_sec"] / source["time_base"]))
    for line in (root / "observations.jsonl").read_text().splitlines():
        row = json.loads(line)
        key = (row["camera_id"], row["local_id"], row["frame_index"])
        detections[row["camera_id"], row["source_pts"]].append((assignments[key], row))
    return detections, wanted


def banner(case, index, count, tracked):
    canvas = Image.new("RGB", (WIDTH, HEIGHT), (16, 23, 34))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 18)
    bold = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 23)
    title = f"Сквозной трекинг / Эпизод {index} из {count}" if tracked else "Две камеры / Полная запись"
    draw.text((16, 10), title, font=bold, fill="white")
    draw.text((16, 48), "КАМЕРА 1", font=font, fill=(180, 204, 227))
    draw.text((728, 48), "КАМЕРА 2", font=font, fill=(180, 204, 227))
    note = f"Треков алгоритма: {case['total_tracks']} | ID заново в каждом эпизоде" if tracked else "Полная синхронная запись без автоматической разметки людей"
    if tracked and case.get("pending_observations"):
        note += " | Серый ID ? — соответствие не подтверждено"
    draw.text((16, 609), note, font=font, fill=(180, 204, 227))
    draw.text((16, 635), "Выравнивание по экранным часам; кадровая точность не гарантируется", font=font, fill=(180, 204, 227))
    return np.asarray(canvas)[:, :, ::-1].copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--demo", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    alignment = json.loads(args.alignment.read_text())
    for info in alignment["cameras"].values():
        if file_hash(info["source"]) != info["source_sha256"]:
            raise ValueError("Clock mapping does not match source file")
    if args.demo:
        demo = json.loads(args.demo.read_text())
        if demo["status"] != "complete":
            raise ValueError("Demo processing has not completed")
        cases = demo["cases"]
    else:
        start = max(info["common_start_sec"] for info in alignment["cameras"].values())
        end = min(info["common_end_sec"] for info in alignment["cameras"].values())
        cases = [{"name": "full", "window_ms": [round(start * 1000), round(end * 1000)]}]
    epoch = datetime.fromisoformat(alignment["timeline_origin"])
    osd = {}
    for camera in alignment["cameras"]:
        path = args.alignment.parent / "final-audit" / f"{camera}-clock-frames.jsonl"
        osd[camera] = {r["source_pts"]: r["osd_sec"] for line in path.read_text().splitlines() if (r := json.loads(line))}
    stats = {"frames": 0, "blank_panels": 0, "paired_frames": 0, "same_displayed_second": 0,
             "max_model_pair_difference_sec": 0.0, "cases": []}
    cv2.setNumThreads(1)
    with av.open(str(args.output), "w", options={"movflags": "+faststart"}) as output:
        stream = output.add_stream("libx264", rate=Fraction(25, 2))
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        stream.options = {"preset": "fast", "crf": "23", "threads": "3"}
        for index, case in enumerate(cases, 1):
            start, end = (v / 1000 for v in case["window_ms"])
            timeline = [start + k * .08 for k in range(int(np.ceil((end - start) / .08)))]
            detections, wanted = frame_catalog(case) if args.demo else ({}, None)
            sources = [nearest(source_frames(info, start, end, wanted.get(camera, set()) if wanted is not None else None), timeline)
                       for camera, info in alignment["cameras"].items()]
            base, cache = banner(case, index, len(cases), bool(args.demo)), {}
            for timestamp, left, right in zip(timeline, *sources, strict=True):
                canvas = base.copy()
                valid = []
                for camera, selected, x in (("cam_1", left, 16), ("cam_2", right, 728)):
                    if selected is None or abs(selected[0] - timestamp) > .18:
                        stats["blank_panels"] += 1
                        cv2.putText(canvas, "NO SOURCE FRAME", (x + 170, 320), cv2.FONT_HERSHEY_SIMPLEX, .9, (170, 190, 215), 2)
                        continue
                    time, pts, frame = selected
                    if cache.get(camera, (None,))[0] != pts:
                        image = frame.reformat(width=PW, height=PH, format="bgr24").to_ndarray()
                        rows = detections.get((camera, pts), [])
                        if rows:
                            image = annotate(image, rows, frame.width, frame.height)
                        cache[camera] = (pts, image)
                    canvas[75:75+PH, x:x+PW] = cache[camera][1]
                    valid.append((time, osd[camera][pts]))
                if len(valid) == 2:
                    stats["paired_frames"] += 1
                    stats["same_displayed_second"] += valid[0][1] == valid[1][1]
                    stats["max_model_pair_difference_sec"] = max(stats["max_model_pair_difference_sec"], abs(valid[0][0] - valid[1][0]))
                stamp = (epoch + timedelta(seconds=timestamp)).strftime("%H:%M:%S.%f")[:-4]
                cv2.putText(canvas, f"OSD {stamp}", (1080, 35), cv2.FONT_HERSHEY_SIMPLEX, .75, (230, 240, 250), 2, cv2.LINE_AA)
                for packet in stream.encode(av.VideoFrame.from_ndarray(canvas, format="bgr24")):
                    output.mux(packet)
                stats["frames"] += 1
                if stats["frames"] % 1500 == 0:
                    print(f"Encoded {stats['frames']} frames", flush=True)
            stats["cases"].append({"name": case["name"], "frames": len(timeline), "window_ms": case["window_ms"]})
            print(f"{case['name']}: {len(timeline)} frames", flush=True)
        for packet in stream.encode():
            output.mux(packet)
    stats["displayed_second_agreement"] = stats["same_displayed_second"] / max(1, stats["paired_frames"])
    stats["note"] = "Pair difference measures the fitted OSD model, not certified physical capture error."
    args.output.with_suffix(".validation.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
