"""Read the fixed office OSD clock using manually verified digit templates.

This decoder is scene-specific. It rejects uncertain digits and exports every
measurement so clock alignment can be independently audited.
"""

import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np


CELLS = [(450, 485), (485, 519), (548, 581), (581, 615)]
MANUAL = {0: "3840", 175: "4138", 225: "4229", 236: "4241", 273: "4318",
          535: "4736", 900: "5338", 1190: "5834"}
MANUAL_EXTRA = {"error-cam1-178.2.png": "4141", "error-cam1-482.png": "4640",
                "error-cam1-483.png": "4641", "error-cam1-484.5.png": "4643",
                "error-cam2-9.5.png": "3850", "error-cam2-666.4.png": "4950",
                "error-cam2-80.5.png": "4002"}
EPOCH_MMSS = 38 * 60 + 39


def glyph(gray, bounds):
    left, right = bounds
    binary = (gray[8:64, left:right] > 180).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    choices = [cv2.boundingRect(c) for c in contours]
    choices = [r for r in choices if r[3] >= 25 and r[2] >= 5]
    if not choices:
        raise ValueError("No clock digit found")
    x, y, w, h = max(choices, key=lambda r: r[2] * r[3])
    return cv2.resize(binary[y:y+h, x:x+w], (24, 40), interpolation=cv2.INTER_NEAREST).astype(bool)


def templates(directory):
    result = {str(i): [] for i in range(10)}
    samples = {f"cam2_{time}.png": label for time, label in MANUAL.items()} | MANUAL_EXTRA
    for filename, label in samples.items():
        path = directory / filename
        luma = path.with_suffix(".luma.png")
        gray = cv2.imread(str(luma if luma.is_file() else path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        for cell, digit in zip(CELLS, label):
            result[digit].append(glyph(gray, cell))
    return result


def read_clock(gray, gallery):
    digits, confidence, margins = [], [], []
    for cell in CELLS:
        sample = glyph(gray, cell)
        scores = sorted([(max(np.count_nonzero(sample & ref) / np.count_nonzero(sample | ref)
                              for ref in refs), digit) for digit, refs in gallery.items()], reverse=True)
        confidence.append(scores[0][0])
        margins.append(scores[0][0] - scores[1][0])
        digits.append(scores[0][1])
    minute, second = int("".join(digits[:2])), int("".join(digits[2:]))
    if minute >= 60 or second >= 60 or min(confidence) < 0.65 or min(margins) < 0.05:
        raise ValueError(f"Uncertain OSD {digits}: {min(confidence):.3f}, margin {min(margins):.3f}")
    return minute * 60 + second - EPOCH_MMSS, min(confidence), min(margins)


def audit(path, camera, directory, gallery):
    measurements, errors, gaps, transitions = [], [], [], []
    previous_time, previous_clock, origin = None, None, None
    with av.open(str(path)) as video:
        stream = video.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.thread_count = 4
        for index, frame in enumerate(video.decode(video=0)):
            if frame.pts is None:
                raise ValueError("Missing PTS")
            raw = float(frame.pts * frame.time_base)
            if origin is None:
                origin = raw
            time = raw - origin
            if previous_time is not None and time <= previous_time:
                raise ValueError("Nonmonotonic media clock")
            if previous_time is not None and time - previous_time > 0.5:
                gaps.append({"frame_index": index, "before_sec": previous_time, "after_sec": time,
                             "gap_sec": time - previous_time})
            if frame.format.name in ("yuv420p", "yuvj420p"):
                plane = frame.planes[0]
                gray = np.frombuffer(plane, np.uint8).reshape(plane.height, plane.line_size)
            else:
                gray = frame.to_ndarray(format="gray")
            try:
                clock, score, margin = read_clock(gray, gallery)
                if previous_clock is not None and clock < previous_clock:
                    raise ValueError(f"OSD moved backwards: {clock} < {previous_clock}")
                row = {"frame_index": index, "source_pts": frame.pts, "video_time_sec": time,
                       "osd_sec": clock, "score": round(score, 4), "margin": round(margin, 4)}
                measurements.append(row)
                if previous_clock is not None and clock != previous_clock:
                    transitions.append({"video_time_sec": time, "timestamp_sec": clock,
                                        "frame_index": index, "source_pts": frame.pts,
                                        "previous_video_time_sec": previous_time})
                previous_clock = clock
            except ValueError as error:
                errors.append({"frame_index": index, "video_time_sec": time, "error": str(error)})
            previous_time = time
            if index % 5000 == 0:
                print(f"{camera}: {index} frames, media={time:.2f}, OCR failures={len(errors)}", flush=True)
    with (directory / f"{camera}-clock-frames.jsonl").open("x", encoding="utf-8") as file:
        for row in measurements:
            file.write(json.dumps(row, separators=(",", ":")) + "\n")
    result = {"camera_id": camera, "source": str(path.resolve()), "first_media_time_sec": origin,
              "decoded_frames": index + 1, "recognized_frames": len(measurements),
              "errors": errors, "media_gaps": gaps, "transitions": transitions}
    (directory / f"{camera}-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    gallery = templates(args.templates)
    for camera in (1, 2):
        result = audit(Path(f"camera_new_{camera}.mkv"), f"cam_{camera}", args.output, gallery)
        print(f"cam_{camera}: {result['recognized_frames']}/{result['decoded_frames']}, gaps={result['media_gaps']}", flush=True)


if __name__ == "__main__":
    main()
