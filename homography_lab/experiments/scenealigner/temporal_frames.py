"""Select predeclared temporal samples and two foreground stress cases; no labels."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

UNIFORM_FRACTIONS = [0.05, 0.1625, 0.275, 0.3875, 0.5, 0.6125, 0.75, 0.925]


def foreground_score(frame, background, mask):
    difference = frame.astype(float) - background.astype(float)
    difference -= np.median(difference[mask], axis=0)
    foreground = (np.max(abs(difference), axis=-1) > 35).astype(np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return float(foreground[mask].mean())


def choose_stress(probes, uniform_seconds, count=2):
    selected = []
    for probe in sorted(probes, key=lambda item: (-item["foreground_fraction"], item["actual_seconds"])):
        time = probe["actual_seconds"]
        if min(abs(time - t) for t in uniform_seconds) < 1:
            continue
        if any(abs(time - other["actual_seconds"]) < 60 for other in selected):
            continue
        selected.append(probe)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError("Insufficient separated stress samples")
    return selected


def main():
    import av

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--floor-mask", type=Path, required=True)
    parser.add_argument("--camera", default="cam_2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "probes").mkdir()
    background = cv2.resize(cv2.imread(str(args.background)), (640, 480))
    mask = cv2.resize(cv2.imread(str(args.floor_mask), 0), (640, 480), interpolation=cv2.INTER_NEAREST) > 128
    if mask.mean() < 0.01:
        raise ValueError("No usable floor region")
    with av.open(str(args.video)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.thread_count = 2
        first = next(container.decode(video=0))
        origin = float(first.pts * first.time_base)
        duration = (
            float(stream.duration * stream.time_base)
            if stream.duration
            else float(container.duration / av.time_base)
        )

        def extract(time):
            container.seek(int((time + origin) / stream.time_base), stream=stream, backward=True)
            for frame in container.decode(video=0):
                if frame.pts is not None and float(frame.pts * frame.time_base) - origin >= time:
                    return frame.to_ndarray(format="bgr24"), float(frame.pts * frame.time_base) - origin
            raise ValueError(f"Cannot decode time {time}")

        uniform_seconds = [fraction * duration for fraction in UNIFORM_FRACTIONS]
        probes = []
        for index, time in enumerate(np.arange(10, duration - 5, 20)):
            image, actual = extract(float(time))
            score = foreground_score(cv2.resize(image, (640, 480)), background, mask)
            path = args.output / "probes" / f"probe_{index:03d}.jpg"
            cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            probes.append(
                {"actual_seconds": actual, "foreground_fraction": score, "path": str(path.resolve())}
            )
            if index % 15 == 0:
                print(f"Foreground scan {actual:.0f}/{duration:.0f}s", flush=True)
        stress = choose_stress(probes, uniform_seconds)
        cases = []
        for index, (time, kind) in enumerate(
            [(t, "uniform") for t in uniform_seconds]
            + [(p["actual_seconds"], "foreground_stress") for p in stress],
            1,
        ):
            image, actual = extract(time)
            folder = args.output / f"case_{index:02d}"
            (folder / "views").mkdir(parents=True)
            path = folder / "views" / (args.camera + ".jpg")
            cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cases.append(
                {
                    "id": folder.name,
                    "kind": kind,
                    "requested_seconds": time,
                    "actual_seconds": actual,
                    "foreground_fraction": foreground_score(cv2.resize(image, (640, 480)), background, mask),
                    "image": str(path.resolve()),
                    "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    manifest = {
        "camera": args.camera,
        "video": str(args.video.resolve()),
        "duration_seconds": duration,
        "video_sha256": hashlib.sha256(args.video.read_bytes()).hexdigest(),
        "manual_landmarks_used_for_sampling": False,
        "uniform_fractions": UNIFORM_FRACTIONS,
        "stress_rule": "Top foreground fraction on a fixed floor mask; >=1s from uniform cases, >=60s apart",
        "cases": cases,
        "probes": probes,
    }
    (args.output / "samples.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Selected", len(cases), "new timestamps", flush=True)


if __name__ == "__main__":
    main()
