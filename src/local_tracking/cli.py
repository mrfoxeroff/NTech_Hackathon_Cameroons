"""Command line entry point; inspection does not import neural-network packages."""

import argparse
import json
import logging
import math
import sys
from dataclasses import asdict
from pathlib import Path

from .config import load_settings
from .video import probe_video


def parser():
    result = argparse.ArgumentParser(description="Offline per-camera person segmentation and ByteTrack")
    sub = result.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Check installed versions and execute a CUDA operation")
    audit = sub.add_parser("validate", help="Audit a completed camera output and referenced masks/crops")
    audit.add_argument("output", type=Path)
    for name in ("inspect", "process", "download-model"):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, default=Path("configs/cameras.yaml"))
        if name == "inspect":
            p.add_argument("--output", type=Path)
        if name == "process":
            p.add_argument("--camera", action="append", help="Camera ID; repeat for multiple cameras")
            p.add_argument("--start", type=float, default=0.0)
            p.add_argument("--end", type=float)
            p.add_argument("--max-frames", type=int)
            p.add_argument("--stride", type=int)
            p.add_argument("--device")
            p.add_argument("--output-root", type=Path)
            p.add_argument("--no-preview", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        if args.command == "validate":
            from .validate import validate_output

            print(json.dumps(validate_output(args.output), indent=2))
            return 0
        if args.command == "doctor":
            from .pipeline import runtime_info

            print(json.dumps(runtime_info(), indent=2))
            return 0
        settings = load_settings(args.config)
        if args.command == "inspect":
            payload = {c.camera_id: asdict(probe_video(c.video)) for c in settings.cameras}
            text = json.dumps(payload, ensure_ascii=False, indent=2)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(text, encoding="utf-8")
            print(text)
            return 0
        if args.command == "download-model":
            from ultralytics import YOLO

            settings.model.parent.mkdir(parents=True, exist_ok=True)
            if settings.model.exists():
                print(f"Already exists: {settings.model}")
            else:
                if not settings.model.name.startswith("yolo"):
                    raise FileNotFoundError(
                        "Custom trained weights must be copied locally. See docs/mirror_finetuning_report.md; "
                        "for official weights use configs/cameras.pretrained.yaml"
                    )
                YOLO(str(settings.model))
                if not settings.model.is_file():
                    raise RuntimeError(f"Model download did not produce {settings.model}")
                print(f"Downloaded: {settings.model}")
            return 0
        if not math.isfinite(args.start) or args.start < 0:
            raise ValueError("--start must be finite and non-negative")
        if args.end is not None and (not math.isfinite(args.end) or args.end <= args.start):
            raise ValueError("--end must be finite and greater than --start")
        if args.max_frames is not None and args.max_frames <= 0:
            raise ValueError("--max-frames must be positive")
        if args.stride is not None:
            settings.stride = args.stride
        if args.device is not None:
            settings.device = args.device
        if args.output_root is not None:
            settings.output_root = args.output_root.resolve()
        if args.no_preview:
            settings.save_preview = False
        settings.validate()
        wanted = set(args.camera or [c.camera_id for c in settings.cameras])
        unknown = wanted - {c.camera_id for c in settings.cameras}
        if unknown:
            raise ValueError(f"Unknown camera IDs: {sorted(unknown)}")
        cameras = [c for c in settings.cameras if c.camera_id in wanted]
        # Validate all targets before doing expensive work; never erase a prior run.
        for camera in cameras:
            target = settings.output_root / settings.recording_id / camera.camera_id
            if target.exists():
                raise FileExistsError(f"Output already exists: {target}. Choose a new --output-root")
        from .pipeline import load_model, process_camera, runtime_info

        runtime = runtime_info(settings.device)
        model, person_class = load_model(settings)
        for camera in cameras:
            process_camera(
                settings, camera, model, person_class, runtime, args.start, args.end, args.max_frames
            )
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        logging.error("%s: %s", type(exc).__name__, exc)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == "__main__":
    sys.exit(main())
