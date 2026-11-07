"""Re-run local tracking and global association on OSD-aligned demonstration windows."""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from build_clock_alignment import interpolate
from import_local_tracking import IntelEncoder, import_outputs
from local_tracking.config import load_settings
from local_tracking.pipeline import load_model, process_camera, runtime_info, file_hash


WINDOWS = [("four_people", 175, 205), ("dark_outfits", 265, 285),
           ("blue_jacket", 534, 550), ("dark_jacket", 895, 912)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-only", action="store_true", help="Skip identity refinement; retain the previous baseline for comparison")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    settings = load_settings(root / "local_stage.example.yaml")
    settings.imgsz, settings.stride, settings.save_preview = 960, 3, False
    settings.timeline_origin = "2026-10-08T17:38:39"
    settings.output_root = args.output.resolve() / "local"
    alignment = json.loads(args.alignment.read_text())
    for camera in settings.cameras:
        info = alignment["cameras"][camera.camera_id]
        if file_hash(camera.video) != info["source_sha256"]:
            raise ValueError("Clock mapping belongs to a different video")
        camera.time_offset_sec = 0
        camera.clock_anchors = info["clock_anchors"]
    settings.validate()
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = runtime_info(settings.device)
    model, person = load_model(settings)
    encoder = IntelEncoder(root / "weights/person-reidentification-retail-0288.xml")
    config = json.loads((root / "camera_graph.improved.json").read_text())
    manifest = {"alignment": str(args.alignment.resolve()), "status": "running", "cases": []}
    manifest_path = args.output / "demo.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    try:
        for name, start, end in WINDOWS:
            selected = replace(settings, recording_id=name)
            inputs = []
            for camera in selected.cameras:
                xs = [a["timestamp_sec"] for a in camera.clock_anchors]
                ys = [a["video_time_sec"] for a in camera.clock_anchors]
                local_start = interpolate(start, xs, ys)
                local_end = interpolate(end, xs, ys)
                print(f"{name} {camera.camera_id}: media {local_start:.3f}..{local_end:.3f}, common {start}..{end}", flush=True)
                process_camera(selected, camera, model, person, runtime, local_start, local_end)
                inputs.append(selected.output_root / name / camera.camera_id)
            destination = args.output / "global" / name
            report = import_outputs(inputs, destination, encoder, config, run_association=True)
            manifest["cases"].append({"name": name, "window_ms": [start * 1000, end * 1000],
                                      "global_output": str(destination.resolve()),
                                      "total_tracks": report["total_tracks"]})
            manifest_path.write_text(json.dumps(manifest, indent=2))
            print(f"{name}: {report['total_tracks']} algorithm tracks", flush=True)
        manifest["status"] = "complete"
        if not args.baseline_only:
            baseline = args.output / "demo.local.json"
            baseline.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            enriched = args.output / "osnet"
            subprocess.run([sys.executable, str(root / "enrich_osnet.py"), str(args.output / "global"), str(enriched)],
                           check=True, creationflags=flags)
            refined = args.output / "refined"
            subprocess.run([sys.executable, str(root / "refine_cross_camera.py"), "--demo", str(baseline),
                            "--enriched", str(enriched), "--config", str(root / "camera_graph.stable.json"),
                            "--output", str(refined)], check=True, creationflags=flags)
            manifest = json.loads((refined / "demo.json").read_text(encoding="utf-8"))
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
