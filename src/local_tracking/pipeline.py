"""Offline processing with immutable per-camera output directories."""

import hashlib
import json
import logging
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from .artifacts import ObservationWriter, PreviewWriter, TrackSummary, write_json
from .config import CameraConfig, Settings, tracker_parameters
from .scene_filter import SceneFilter
from .stabilization import GeometryStabilizer
from .tracker import LocalTracker
from .video import iter_video, probe_video

LOG = logging.getLogger(__name__)


def jsonable(value):
    return json.loads(json.dumps(value, default=str))


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def runtime_info(device="auto"):
    import torch

    available = torch.cuda.is_available()
    if device == "auto":
        device = "0" if available else "cpu"
    if device != "cpu" and not available:
        raise RuntimeError("CUDA requested but unavailable; run doctor or explicitly choose --device cpu")
    gpu_index = 0 if device == "cpu" else int(device.removeprefix("cuda:"))
    if device != "cpu":
        # Actual CUDA work catches incompatible GPU architectures/wheels early.
        (torch.ones(1, device=f"cuda:{gpu_index}") + 1).cpu()
    return {
        "device": device,
        "cuda_available": available,
        "torch_cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(gpu_index) if device != "cpu" else None,
        "gpu_memory_bytes": torch.cuda.get_device_properties(gpu_index).total_memory
        if device != "cpu"
        else None,
        "package_versions": {
            name: version(name)
            for name in (
                "torch",
                "torchvision",
                "ultralytics",
                "av",
                "numpy",
                "opencv-python",
                "lap",
            )
        },
    }


def load_model(settings):
    from ultralytics import YOLO

    if not settings.model.is_file():
        raise FileNotFoundError(
            f"Weights not found: {settings.model}. Download once with: "
            "python -m local_tracking download-model --config configs/cameras.yaml"
        )
    model = YOLO(str(settings.model))
    if model.task != "segment":
        raise ValueError("A segmentation model is required to export actual silhouettes")
    person_ids = [int(k) for k, name in model.names.items() if name.lower() == "person"]
    if len(person_ids) != 1:
        raise ValueError("Model must contain exactly one class named person")
    return model, person_ids[0]


def process_camera(
    settings: Settings,
    camera: CameraConfig,
    model,
    person_class: int,
    runtime: dict,
    start_sec=0.0,
    end_sec=None,
    max_frames=None,
):
    import torch

    info = probe_video(camera.video)
    output = settings.output_root / settings.recording_id / camera.camera_id
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": "1.0",
        "status": "running",
        "recording_id": settings.recording_id,
        "camera_id": camera.camera_id,
        "timeline_origin": settings.timeline_origin,
        "created_at": datetime.now(UTC).isoformat(),
        "source": asdict(info),
        "timestamp_origin": "first_decoded_frame",
        "time_offset_sec": camera.time_offset_sec,
        "clock_anchors": camera.clock_anchors,
        "frame_stride": settings.stride,
        "selection": {"start_sec": start_sec, "end_sec": end_sec, "max_frames": max_frames},
        "configuration": jsonable(asdict(settings)),
        "runtime": runtime,
        "tracker_parameters": tracker_parameters(settings),
        "artifacts": {
            "frames": "frames.jsonl",
            "tracks": "local_tracks.json",
            "metrics": "metrics.json",
            "preview": "preview.mp4" if settings.save_preview else None,
        },
    }
    write_json(output / "manifest.json", manifest)
    preview = None
    started = time.perf_counter()
    metrics = {
        "processed_frames": 0,
        "observations": 0,
        "untracked_observations": 0,
        "empty_frames": 0,
        "timestamp_fallback_frames": 0,
        "source_gap_count": 0,
        "scene_rejected_detections": 0,
    }
    summary = TrackSummary(settings.selected_crops_per_track)
    crop_interval = (
        min(settings.crop_interval_sec, 0.25) if camera.appearance_tracking else settings.crop_interval_sec
    )
    writer = ObservationWriter(output, crop_interval, settings.save_masks)
    original_error = None
    previous_video_time = None
    try:
        manifest["source_sha256"] = file_hash(camera.video)
        manifest["weights_sha256"] = file_hash(settings.model)
        tracker = LocalTracker(settings, info.fps, appearance=camera.appearance_tracking)
        scene_filter = SceneFilter(camera.scene_regions, info.width, info.height)
        stabilizer = GeometryStabilizer(
            info.width, info.height, camera.occluder_polygons, reset_gap_sec=settings.lost_track_sec
        )
        manifest["tracker_buffer_steps"] = tracker.buffer_steps
        if runtime["device"] != "cpu":
            torch.cuda.reset_peak_memory_stats(int(runtime["device"].removeprefix("cuda:")))
        if settings.save_preview:
            preview = PreviewWriter(
                output / "preview.mp4",
                info.width,
                info.height,
                info.fps / settings.stride,
                settings.preview_max_width,
            )

        def predict(image, confidence):
            return model.predict(
                image,
                classes=[person_class],
                conf=confidence,
                iou=settings.iou,
                imgsz=settings.imgsz,
                device=runtime["device"],
                quantize=16 if settings.fp16 and runtime["device"] != "cpu" else 32,
                retina_masks=True,
                verbose=False,
            )[0]

        frames = iter_video(camera.video, camera.time_offset_sec, start_sec, end_sec, settings.stride)
        with (output / "frames.jsonl").open("w", encoding="utf-8") as stream:
            for step, frame in enumerate(frames):
                frame = replace(frame, timestamp_sec=camera.common_time(frame.video_time_sec))
                delta = 0.0 if previous_video_time is None else frame.video_time_sec - previous_video_time
                source_gap = delta if delta > max(0.5, 3 * settings.stride / info.fps) else 0.0
                previous_video_time = frame.video_time_sec
                if frame.image.shape[:2] != (info.height, info.width):
                    raise ValueError("Source resolution changed inside recording")

                result = predict(frame.image, settings.confidence)
                if camera.detail_roi is not None:
                    from .detail import refine_distant

                    result = refine_distant(result, frame.image, camera.detail_roi, predict)
                result, rejected = scene_filter.apply(result, frame.image)
                metrics["scene_rejected_detections"] += len(rejected)
                features = None
                if camera.appearance_tracking:
                    from .appearance import clothing_colors

                    features = clothing_colors(result, frame.image)
                assignments = tracker.update(
                    result.boxes.cpu().numpy(), frame.image, frame.timestamp_sec, features
                )
                observations = writer.build(
                    result, assignments, frame, settings.recording_id, camera.camera_id
                )
                stabilizer.update(observations, frame.timestamp_sec)
                record = {
                    "recording_id": settings.recording_id,
                    "camera_id": camera.camera_id,
                    "frame_index": frame.index,
                    "video_time_sec": frame.video_time_sec,
                    "timestamp_sec": frame.timestamp_sec,
                    "timestamp_source": frame.timestamp_source,
                    "media_time_sec": frame.media_time_sec,
                    "source_gap_before_sec": source_gap,
                    "frame_width": info.width,
                    "frame_height": info.height,
                    "observations": observations,
                    "scene_rejections": rejected,
                }
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                for obs in observations:
                    summary.add(obs, frame.timestamp_sec, step, source_gap > 0)
                if preview is not None:
                    preview.add(frame, observations, camera.camera_id)
                metrics["processed_frames"] += 1
                metrics["observations"] += len(observations)
                metrics["untracked_observations"] += sum(o["local_track_id"] is None for o in observations)
                metrics["empty_frames"] += not observations
                metrics["timestamp_fallback_frames"] += frame.timestamp_source != "pts"
                metrics["source_gap_count"] += source_gap > 0
                if step == 0:
                    metrics["first_video_time_sec"] = frame.video_time_sec
                metrics["last_video_time_sec"] = frame.video_time_sec
                if step % 100 == 0:
                    stream.flush()
                    LOG.info(
                        "%s: %d frames, t=%.2fs, %d local tracks",
                        camera.camera_id,
                        metrics["processed_frames"],
                        frame.video_time_sec,
                        len(summary.tracks),
                    )
                if max_frames is not None and metrics["processed_frames"] >= max_frames:
                    break
        if not metrics["processed_frames"]:
            raise ValueError("Selected interval contains no frames; check start/end/stride")
        manifest["status"] = "complete"
    except BaseException as exc:
        original_error = exc
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        finalization_errors = []

        def finalize(action):
            try:
                action()
            except BaseException as exc:
                finalization_errors.append(exc)
                LOG.error("%s finalization failed: %s: %s", camera.camera_id, type(exc).__name__, exc)

        if preview is not None:
            finalize(preview.close)
            metrics["preview_frames"] = preview.count
        metrics["elapsed_sec"] = time.perf_counter() - started
        metrics["processing_fps"] = metrics["processed_frames"] / max(metrics["elapsed_sec"], 1e-9)
        metrics["local_track_count"] = len(summary.tracks)
        if runtime["device"] != "cpu":

            def collect_cuda_metrics():
                metrics["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(
                    int(runtime["device"].removeprefix("cuda:"))
                )

            finalize(collect_cuda_metrics)
        finalize(
            lambda: write_json(
                output / "local_tracks.json",
                {
                    "schema_version": "1.0",
                    "recording_id": settings.recording_id,
                    "camera_id": camera.camera_id,
                    "tracks": summary.export(),
                },
            )
        )
        finalize(lambda: write_json(output / "metrics.json", metrics))
        if finalization_errors:
            manifest["status"] = "failed"
            manifest["finalization_errors"] = [
                {"type": type(exc).__name__, "message": str(exc)} for exc in finalization_errors
            ]
            if original_error is None:
                first_error = finalization_errors[0]
                manifest["error"] = {"type": type(first_error).__name__, "message": str(first_error)}
        manifest["finished_at"] = datetime.now(UTC).isoformat()
        finalize(lambda: write_json(output / "manifest.json", manifest))
        # Preserve the processing exception and its traceback. If processing
        # succeeded, finalization failure still makes the command fail.
        if original_error is None and finalization_errors:
            raise finalization_errors[0]
    LOG.info("%s complete: %s", camera.camera_id, json.dumps(metrics))
    return metrics
