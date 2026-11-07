"""Process cameras 1 and 2 on the verified office clock and build a web session."""

import argparse
import json
import logging
import math
from pathlib import Path

from local_tracking.config import load_settings
from local_tracking.pipeline import file_hash, load_model, process_camera, runtime_info
from mtmc.build_clock_alignment import interpolate
from mtmc.import_local_tracking import IntelEncoder, import_outputs
from tools.export_web_session import export_session, repair_associations

ROOT = Path(__file__).resolve().parents[1]


def run(output, start, end, stride, reid):
    if output.exists():
        raise FileExistsError(output)
    alignment = json.loads((ROOT / "mtmc/sync-check/alignment.json").read_text(encoding="utf-8"))
    settings = load_settings(ROOT / "configs/cameras.yaml", camera_ids={"camera_1", "camera_2"})
    if settings.timeline_origin != alignment["timeline_origin"]:
        raise ValueError("Expected the verified office clock epoch")
    windows = []
    for camera in settings.cameras:
        info = alignment["cameras"][camera.camera_id.replace("camera_", "cam_")]
        if file_hash(camera.video) != info["source_sha256"]:
            raise ValueError(f"Clock calibration belongs to a different video: {camera.video}")
        camera.clock_anchors = info["clock_anchors"]
        camera.time_offset_sec = 0
        common = [anchor["timestamp_sec"] for anchor in camera.clock_anchors]
        media = [anchor["video_time_sec"] for anchor in camera.clock_anchors]
        if start >= common[-1] or (end is not None and end <= common[0]):
            raise ValueError(f"Selected window has no source frames: {camera.camera_id}")
        windows.append(
            (
                max(0.0, interpolate(start, common, media)),
                interpolate(end, common, media) if end is not None else None,
            )
        )

    settings.stride = stride
    settings.save_preview = False
    settings.output_root = output / "local"
    settings.validate()
    encoder = IntelEncoder(reid)
    runtime = runtime_info(settings.device)
    model, person = load_model(settings)
    output.mkdir(parents=True, exist_ok=False)
    inputs = []
    for camera, (local_start, local_end) in zip(settings.cameras, windows, strict=True):
        print(f"{camera.camera_id}: processing from source {local_start:.3f} s", flush=True)
        process_camera(settings, camera, model, person, runtime, local_start, local_end)
        inputs.append(settings.output_root / settings.recording_id / camera.camera_id)

    config = json.loads((ROOT / "mtmc/camera_graph.stable.json").read_text(encoding="utf-8"))
    config["overlap_pairs"] = [["camera_1", "camera_2"]]
    # The current local model already applies the camera-specific scene filters.
    config.pop("excluded_image_rects", None)
    global_root = output / "global"
    import_outputs(inputs, global_root, encoder, config)
    repair_associations(global_root, ROOT / "office.jpg", ROOT)
    session = output / "session"
    payload = export_session(global_root, session, ROOT / "office.jpg", ROOT)
    print(f"Ready: {session} ({payload['total_unique_tracks']} global tracks)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs_full")
    parser.add_argument("--start", type=float, default=0, help="Start on the common clock, seconds")
    parser.add_argument("--end", type=float, help="End on the common clock, seconds")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument(
        "--reid", type=Path, default=ROOT / "mtmc/weights/person-reidentification-retail-0288.xml"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if (
        not math.isfinite(args.start)
        or args.start < 0
        or (args.end is not None and (not math.isfinite(args.end) or args.start >= args.end))
    ):
        parser.error("Expected finite 0 <= start < end")
    try:
        run(args.output.resolve(), args.start, args.end, args.stride, args.reid.resolve())
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(2, f"error: {error}\n")
