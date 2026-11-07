"""Align uploaded recordings to their burned-in calendar clocks."""

import re
import shutil
import subprocess
from datetime import datetime, timedelta
from statistics import median

import av
import cv2
import numpy as np

from local_tracking.pipeline import file_hash
from local_tracking.video import _fps, _video_stream
from mtmc.build_clock_alignment import interpolate

EPOCH = datetime(1970, 1, 1)
DEFAULT_ROI = (0, 0, 0.42, 0.07)
STAMP = re.compile(r"(?<!\d)(\d{4})[-/.](\d{2})[-/.](\d{2})\D{0,8}(\d{2})[:-](\d{2})[:-](\d{2})(?!\d)")


def parse_stamp(text):
    match = STAMP.search(re.sub(r"\s*([:/.-])\s*", r"\1", text))
    if not match:
        return None
    try:
        return int((datetime(*map(int, match.groups())) - EPOCH).total_seconds())
    except ValueError:
        return None


def recognize(samples, executable):
    height = 90
    sheet = np.full((height * len(samples), 1120), 255, np.uint8)
    for index, (_, image) in enumerate(samples):
        factor = min(1100 / image.shape[1], 72 / image.shape[0])
        tile = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
        if np.mean(tile) < 128:
            tile = 255 - tile
        top = index * height + (height - tile.shape[0]) // 2
        sheet[top : top + tile.shape[0], 10 : 10 + tile.shape[1]] = tile
    png = cv2.imencode(".png", sheet)[1].tobytes()
    result = subprocess.run(
        [
            executable,
            "stdin",
            "stdout",
            "--psm",
            "6",
            "-c",
            "tessedit_char_whitelist=0123456789:-./",
            "tsv",
        ],
        input=png,
        capture_output=True,
        timeout=90,
        check=False,
    )
    if result.returncode:
        raise ValueError("Tesseract не смог обработать кадры. Проверьте наличие языкового пакета eng.")
    words = [[] for _ in samples]
    for line in result.stdout.decode("utf-8", "replace").splitlines()[1:]:
        fields = line.split("\t")
        if len(fields) == 12 and fields[0] == "5" and fields[11].strip():
            slot = (int(fields[7]) + int(fields[9]) // 2) // height
            if 0 <= slot < len(words):
                words[slot].append((int(fields[6]), fields[11]))
    return [
        (index, parse_stamp(" ".join(text for _, text in sorted(row))))
        for (index, _), row in zip(samples, words, strict=True)
    ]


def sampled_anchors(times, readings, fps):
    valid = [(i, t) for i, t in readings if t is not None]
    if len(valid) < 6 or len(valid) < 0.55 * len(readings):
        raise ValueError("Не удалось устойчиво прочитать дату и время OSD.")
    baseline = median(t - i / fps for i, t in valid)
    valid = [(i, t) for i, t in valid if abs(t - i / fps - baseline) <= 2.0]
    if len(valid) < 0.55 * len(readings):
        raise ValueError("Показания OSD противоречат порядку кадров.")
    transitions = [
        ((a + b) / 2, end)
        for (a, start), (b, end) in zip(valid, valid[1:])
        if end == start + 1 and (b - a) / fps <= 0.65
    ]
    if (
        len(transitions) < 2
        or any(a[1] >= b[1] for a, b in zip(transitions, transitions[1:]))
        or valid[0][0] / fps > 2
        or (len(times) - 1 - valid[-1][0]) / fps > 2
        or max(b[0] - a[0] for a, b in zip(valid, valid[1:])) / fps > 4
    ):
        raise ValueError("Не хватает надёжных смен секунд OSD по длине записи.")
    indexes, seconds = map(list, zip(*transitions))
    if valid[0][0] == 0 and interpolate(0, indexes, seconds) < valid[0][1]:
        indexes.insert(0, 0)
        seconds.insert(0, valid[0][1])
    knots = {0, len(times) - 1, *(round(index) for index in indexes)}
    knots.update(j for i in range(1, len(times)) if times[i] - times[i - 1] > 0.5 for j in (i - 1, i))
    return [(times[i], interpolate(i, indexes, seconds)) for i in sorted(knots)]


def scan_clock(path, roi):
    executable = shutil.which("tesseract")
    if executable is None:
        raise ValueError("Для новых записей установите Tesseract и добавьте его в PATH.")
    alternatives = (roi, (0, 0, 0.25, 0.036), (0.74, 0.015, 1, 0.055)) if roi == DEFAULT_ROI else (roi,)
    times, samples, readings = [], [], []

    def flush():
        nonlocal alternatives
        results = []
        for box in alternatives:
            tiles = []
            for index, image, h in samples:
                w = image.shape[1]
                x1, y1, x2, y2 = (round(a * b) for a, b in zip(box, (w, h, w, h), strict=True))
                tiles.append((index, image[y1:y2, x1:x2]))
            results.append((box, recognize(tiles, executable)))
        box, best = max(results, key=lambda result: sum(t is not None for _, t in result[1]))
        readings.extend(best)
        alternatives = (box,)
        samples.clear()

    with av.open(str(path)) as video:
        stream = _video_stream(video)
        fps = _fps(stream)
        step = max(1, round(fps / 4))
        for i, frame in enumerate(video.decode(stream)):
            if frame.pts is None:
                raise ValueError("Кадр не содержит исходной метки PTS.")
            timestamp = float(frame.pts * frame.time_base)
            if i == 0:
                origin = timestamp
            time = timestamp - origin
            if times and time <= times[-1]:
                raise ValueError("Исходные метки PTS не возрастают.")
            times.append(time)
            if i % step == 0:
                image = frame.to_ndarray(format="gray")
                # Only the upper rows are needed for the default overlays.
                bottom = round(max(box[3] for box in alternatives) * frame.height)
                samples.append((i, image[:bottom].copy(), frame.height))
            if len(samples) == 24:
                flush()
    if samples:
        flush()
    return sampled_anchors(times, readings, fps)


def align_videos(videos, roi, known_maps):
    lookup = {
        camera["source_sha256"]: (datetime.fromisoformat(bundle["timeline_origin"]), camera["clock_anchors"])
        for bundle in known_maps
        for camera in bundle["cameras"].values()
    }
    absolute = {}
    for name, path in videos.items():
        verified = lookup.get(file_hash(path))
        if verified is not None:
            epoch, anchors = verified
            absolute[name] = [
                (a["video_time_sec"], (epoch - EPOCH).total_seconds() + a["timestamp_sec"]) for a in anchors
            ]
        else:
            try:
                absolute[name] = scan_clock(path, roi)
            except (ValueError, OSError, subprocess.SubprocessError) as error:
                raise ValueError(f"{name}: {error}") from error
    start = int(min(values[0][1] for values in absolute.values()) // 1)
    return (EPOCH + timedelta(seconds=start)).isoformat(), {
        name: [{"video_time_sec": media, "timestamp_sec": stamp - start} for media, stamp in values]
        for name, values in absolute.items()
    }
