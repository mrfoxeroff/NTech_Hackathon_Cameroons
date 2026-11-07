"""Re-read uncertain OSD frames with additional visually checked luma templates."""

import argparse
import json
from pathlib import Path

import av
import numpy as np

from audit_sync import read_clock, templates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    gallery = templates(args.templates)
    for camera in ("cam_1", "cam_2"):
        audit = json.loads((args.input / f"{camera}-audit.json").read_text())
        rows = [json.loads(line) for line in (args.input / f"{camera}-clock-frames.jsonl").read_text().splitlines()]
        groups = []
        for error in audit["errors"]:
            if not groups or error["video_time_sec"] - groups[-1][-1]["video_time_sec"] > 0.5:
                groups.append([])
            groups[-1].append(error)
        remaining = []
        with av.open(audit["source"]) as video:
            stream = video.streams.video[0]
            stream.thread_type = "AUTO"
            stream.codec_context.thread_count = 4
            for group in groups:
                requested = {round((e["video_time_sec"] + audit["first_media_time_sec"]) / float(stream.time_base)): e
                             for e in group}
                video.seek(min(requested), stream=stream, backward=True)
                for frame in video.decode(video=0):
                    if frame.pts > max(requested):
                        break
                    if frame.pts not in requested:
                        continue
                    error = requested[frame.pts]
                    plane = frame.planes[0]
                    gray = np.frombuffer(plane, np.uint8).reshape(plane.height, plane.line_size)
                    try:
                        clock, score, margin = read_clock(gray, gallery)
                        rows.append({"frame_index": error["frame_index"], "source_pts": frame.pts,
                                     "video_time_sec": error["video_time_sec"], "osd_sec": clock,
                                     "score": round(score, 4), "margin": round(margin, 4)})
                    except ValueError as exc:
                        remaining.append(error | {"error": str(exc)})
        rows.sort(key=lambda row: row["frame_index"])
        transitions = []
        for left, right in zip(rows, rows[1:]):
            if right["osd_sec"] < left["osd_sec"]:
                raise ValueError("OSD moved backwards after refinement")
            if right["osd_sec"] != left["osd_sec"]:
                transitions.append({"video_time_sec": right["video_time_sec"], "timestamp_sec": right["osd_sec"],
                                    "frame_index": right["frame_index"], "source_pts": right["source_pts"],
                                    "previous_video_time_sec": left["video_time_sec"]})
        if len(rows) + len(remaining) != audit["decoded_frames"]:
            raise ValueError("Some requested frames could not be recovered")
        audit.update(errors=remaining, recognized_frames=len(rows), transitions=transitions)
        (args.output / f"{camera}-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        with (args.output / f"{camera}-clock-frames.jsonl").open("x") as file:
            for row in rows:
                file.write(json.dumps(row, separators=(",", ":")) + "\n")
        print(f"{camera}: {len(rows)} recognized, {len(remaining)} uncertain", flush=True)


if __name__ == "__main__":
    main()
