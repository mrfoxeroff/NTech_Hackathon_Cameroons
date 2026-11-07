"""Process uploaded recordings with source-verified plan anchors."""

import argparse
import json
from datetime import timedelta
from pathlib import Path

from local_tracking.artifacts import write_json
from local_tracking.config import CameraConfig, Settings
from local_tracking.pipeline import load_model, process_camera, runtime_info
from mtmc.import_local_tracking import IntelEncoder, import_outputs
from tools.export_web_session import export_session, read_json, repair_associations, rows
from tools.upload_clock import DEFAULT_ROI, align_videos
from tools.upload_geometry import (
    extend_calibrations,
    localize,
    motion_matches,
    motion_overlaps,
    office_overlaps,
)

ROOT = Path(__file__).resolve().parents[1]


def camera_progress(stage, count, video_time, duration):
    timestamp = str(timedelta(seconds=int(video_time)))
    if duration is not None:
        timestamp += " / " + str(timedelta(seconds=int(duration)))
    return f"{stage}\nОбработано кадров: {count} · время видео {timestamp}"


def run_project(plan, sources, output, *, roi=DEFAULT_ROI, lab_report=None, progress=print):
    if output.exists():
        raise FileExistsError(output)
    if not sources:
        raise ValueError("Нужно хотя бы одно видео.")
    videos = {f"camera_{i}": path.resolve() for i, path in enumerate(sources, 1)}
    known = [
        read_json(ROOT / path)
        for path in ("mtmc/sync-check/alignment.json", "configs/clock-cameras-3-5.json")
    ]
    progress("Синхронизация по времени на видео")
    epoch, anchors = align_videos(videos, roi, known)
    progress("Автоматическая привязка камер к плану")
    calibrated, size = localize(plan, videos, lab_report, known_maps=known)
    settings = Settings(
        "uploaded_scene",
        output / "local",
        ROOT / "weights/person-office-v2.pt",
        [CameraConfig(name, video, clock_anchors=anchors[name]) for name, video in videos.items()],
        timeline_origin=epoch,
        imgsz=800,
        confidence=0.25,
        stride=3,
        save_preview=False,
        tracker={"track_high_thresh": 0.5, "track_low_thresh": 0.5, "new_track_thresh": 0.55},
    )
    settings.validate()
    encoder = IntelEncoder(ROOT / "mtmc/weights/person-reidentification-retail-0288.xml")
    model, person = load_model(settings)
    runtime = runtime_info(settings.device)
    device = runtime["gpu_name"] or "CPU"
    output.mkdir(parents=True, exist_ok=False)
    inputs = []
    for index, camera in enumerate(settings.cameras, 1):
        stage = f"Детекция и локальный трекинг · {camera.camera_id} ({index}/{len(videos)}) · {device}"
        progress(stage)
        process_camera(
            settings,
            camera,
            model,
            person,
            runtime,
            progress=lambda count, video_time, duration: progress(
                camera_progress(stage, count, video_time, duration)
            ),
        )
        inputs.append(settings.output_root / settings.recording_id / camera.camera_id)
    progress("Сопоставление ID между камерами")
    global_root = output / "global"
    config = read_json(ROOT / "mtmc/camera_graph.stable.json")
    office_graph = read_json(ROOT / "configs/web-camera-graph.json")
    config.pop("excluded_image_rects", None)
    config.update(
        overlap_pairs=[],
        reappearance_memory_ms=60000,
        lifecycle=office_graph["lifecycle"],
    )
    import_outputs(inputs, global_root, encoder, config)
    matches = motion_matches(rows(global_root / "observations.jsonl")) if calibrated else {}
    video_sizes = {}
    for name, path in zip(videos, inputs, strict=True):
        source = read_json(path / "manifest.json")["source"]
        video_sizes[name] = (source["width"], source["height"])
    calibrated = extend_calibrations(matches, calibrated, video_sizes, size)
    motion_pairs = motion_overlaps(matches, calibrated, size)
    office_pairs = office_overlaps(calibrated, office_graph["overlap_pairs"])
    overlaps = [list(pair) for pair in sorted({tuple(pair) for pair in motion_pairs + office_pairs})]
    manifest = read_json(global_root / "manifest.json")
    manifest["settings"].update(
        overlap_pairs=overlaps, overlap_requires_map=overlaps, geometry_repair_pairs=[]
    )
    write_json(global_root / "manifest.json", manifest)
    repair_associations(global_root, plan, calibrated)
    localization = {
        "calibrated_cameras": {name: value["method"] for name, value in calibrated.items()},
        "unlocalized_cameras": [name for name in videos if name not in calibrated],
        "motion_validated_overlaps": motion_pairs,
        "office_verified_overlaps": office_pairs,
    }
    write_json(output / "localization.json", localization | {"calibrations": calibrated})
    progress("Сборка синхронного плеера")
    return export_session(global_root, output / "session", plan, calibrated, localization=localization)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--video", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lab-report", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            run_project(args.plan, args.video, args.output, lab_report=args.lab_report)["localization"]
        )
    )
