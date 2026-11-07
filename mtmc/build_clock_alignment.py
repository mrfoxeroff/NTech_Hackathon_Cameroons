"""Build a source-bound clock map with explicit knots at PTS discontinuities."""

import argparse
import json
from bisect import bisect_right
from pathlib import Path

from local_tracking.pipeline import file_hash


def interpolate(value, xs, ys):
    if len(xs) < 2 or len(xs) != len(ys):
        raise ValueError("Interpolation needs at least two matching knots")
    i = max(0, min(bisect_right(xs, value) - 1, len(xs) - 2))
    return ys[i] + (value - xs[i]) * (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])


def build_camera(audit, rows):
    if audit["errors"] or len(rows) != audit["decoded_frames"]:
        raise ValueError("Resolve uncertain OSD readings before constructing the final map")
    if [row["frame_index"] for row in rows] != list(range(len(rows))):
        raise ValueError("Clock audit must cover every source frame exactly once")
    transitions = audit["transitions"]
    indexes = [t["frame_index"] for t in transitions]
    seconds = [t["timestamp_sec"] for t in transitions]
    if any(b - a != 1 for a, b in zip(seconds, seconds[1:])):
        raise ValueError("OSD seconds are missing; inspect possible content loss")
    if interpolate(0, indexes, seconds) < rows[0]["osd_sec"]:
        # The opening OSD second can last more frames than the next one.
        # The first frame still has to belong to its displayed second.
        indexes.insert(0, 0)
        seconds.insert(0, rows[0]["osd_sec"])
    # Interpolate within one displayed second by source-frame order. Interpolating
    # across a 10-second PTS jump would incorrectly allocate almost a whole OSD
    # second to adjacent frames. Explicit before/after knots prevent that.
    corrected = [interpolate(r["frame_index"], indexes, seconds) for r in rows]
    if any(not r["osd_sec"] - 1e-7 <= t < r["osd_sec"] + 1 + 1e-7 for r, t in zip(rows, corrected)):
        raise ValueError("Reconstructed time contradicts the displayed clock")
    chosen = {0, len(rows) - 1, *indexes}
    for gap in audit["media_gaps"]:
        chosen.update((gap["frame_index"] - 1, gap["frame_index"]))
    anchors = [{"video_time_sec": rows[i]["video_time_sec"], "timestamp_sec": corrected[i]}
               for i in sorted(chosen)]
    xs = [a["video_time_sec"] for a in anchors]
    ys = [a["timestamp_sec"] for a in anchors]
    if any(b <= a for a, b in zip(xs, xs[1:])) or any(b <= a for a, b in zip(ys, ys[1:])):
        raise ValueError("Clock map must be strictly increasing")
    disagreement = max(abs(interpolate(r["video_time_sec"], xs, ys) - t) for r, t in zip(rows, corrected))
    return {"source": audit["source"], "source_sha256": file_hash(audit["source"]),
            "decoded_frames": len(rows), "recognized_frames": len(rows),
            "first_media_time_sec": audit["first_media_time_sec"],
            "common_start_sec": corrected[0], "common_end_sec": corrected[-1],
            "clock_anchors": anchors, "pts_discontinuities": audit["media_gaps"],
            "max_map_vs_frame_clock_error_sec": disagreement}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"timeline_origin": "2026-10-08T17:38:39", "timezone": "unknown",
              "method": "OSD second transitions with source-frame interpolation and explicit PTS-jump knots",
              "accuracy_note": "Aligns displayed clocks; subsecond physical capture synchronization is not certified",
              "cameras": {}}
    for camera in ("cam_1", "cam_2"):
        audit = json.loads((args.audit / f"{camera}-audit.json").read_text())
        rows = [json.loads(line) for line in (args.audit / f"{camera}-clock-frames.jsonl").read_text().splitlines()]
        result["cameras"][camera] = build_camera(audit, rows)
        info = result["cameras"][camera]
        print(camera, len(info["clock_anchors"]), "knots; interpolation residual", info["max_map_vs_frame_clock_error_sec"])
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
