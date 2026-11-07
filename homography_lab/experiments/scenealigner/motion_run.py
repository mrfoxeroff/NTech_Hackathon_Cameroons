"""Portable offline motion calibration with preflight and immutable outputs."""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

from motion_config import HERE, LAB, OFFICE, REPOSITORY, dump_json, load_config


def inspect_inputs(args, config):
    required = [
        args.pose_weights,
        args.lenses,
        args.plan_walls,
        args.bank,
        args.source_manifest,
        args.weights_root / "osnet_x0_25_msmt17.pth",
        args.weights_root / "person-reidentification-retail-0288.xml",
        args.weights_root / "person-reidentification-retail-0288.bin",
    ]
    required += [Path(v[key]) for v in config["cameras"].values() for key in ["video", "floor_mask"]]
    missing = [str(p) for p in required if not p.is_file()]
    templates = Path(config["clock_templates"])
    if not templates.is_dir() or not list(templates.glob("*.png")):
        missing.append(str(templates))
    snapshot = args.hf_cache / (
        "models--nvidia--segformer-b0-finetuned-ade-512-512/snapshots/"
        "489d5cd81a0b59fab9b7ea758d3548ebe99677da"
    )
    for name in ["config.json", "preprocessor_config.json", "model.safetensors"]:
        if not (snapshot / name).is_file():
            missing.append(str(snapshot / name))
    modules = [
        name
        for name in ["numpy", "cv2", "scipy", "av", "torch", "torchvision", "transformers", "ultralytics"]
        if importlib.util.find_spec(name) is None
    ]
    check = (
        subprocess.run(
            [
                str(args.cpu_python),
                "-c",
                'import importlib.util; raise SystemExit(0 if importlib.util.find_spec("openvino") else 1)',
            ],
            capture_output=True,
            check=False,
        )
        if args.cpu_python.is_file()
        else None
    )
    if check is None or check.returncode:
        modules.append("openvino in --cpu-python " + str(args.cpu_python))
    return {
        "ready": not missing and not modules,
        "missing_files": missing,
        "missing_modules": modules,
        "config_paths_relative_to_config_file": True,
        "no_output_created": True,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--video-root", type=Path)
    p.add_argument("--cpu-python", type=Path, default=Path(sys.executable))
    p.add_argument("--weights-root", type=Path, default=REPOSITORY / "mtmc/weights")
    p.add_argument("--pose-weights", type=Path, default=HERE / "cache/yolo11n-pose.pt")
    p.add_argument("--hf-cache", type=Path, default=LAB / "neural_weights/huggingface")
    p.add_argument("--lenses", type=Path, default=OFFICE / "lenses.json")
    p.add_argument("--plan-walls", type=Path, default=OFFICE / "plan-walls.png")
    p.add_argument("--bank", type=Path, default=OFFICE / "automatic-candidates.json")
    p.add_argument("--source-manifest", type=Path, default=OFFICE / "source-manifest.json")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--check-inputs", action="store_true")
    args = p.parse_args()
    config = load_config(args.config, args.video_root)
    for name in [
        "cpu_python",
        "weights_root",
        "pose_weights",
        "hf_cache",
        "lenses",
        "plan_walls",
        "bank",
        "source_manifest",
    ]:
        setattr(args, name, getattr(args, name).resolve())
    preflight = inspect_inputs(args, config)
    if args.check_inputs:
        print(json.dumps(preflight, indent=2, ensure_ascii=False))
        return 0 if preflight["ready"] else 1
    if not preflight["ready"]:
        p.error("Inputs are not ready. Run --check-inputs to see missing files and packages.")
    if args.output is None:
        p.error("--output is required for processing")
    from motion_candidates import validate_bank, verify_sources
    from motion_frames import file_hash

    validate_bank(json.loads(args.bank.read_text(encoding="utf-8")))
    verify_sources(
        {"cameras": {c: {"source_sha256": file_hash(v["video"])} for c, v in config["cameras"].items()}},
        json.loads(args.source_manifest.read_text(encoding="utf-8")),
    )
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    dump_json(root / "resolved-config.json", config)
    manifest = {
        "status": "running",
        "manual_plan_landmarks_used": False,
        "global_track_IDs_used": False,
        "config": config,
        "stages": [],
        "automatic_deployment_ready": False,
        "pose_weight_sha256": file_hash(args.pose_weights),
        "environment": {
            name: version(name)
            for name in ["torch", "torchvision", "numpy", "opencv-python", "ultralytics", "av", "scipy"]
        },
    }

    def run(name, arguments, python=sys.executable):
        command = [str(python), str(HERE / name), *map(str, arguments)]
        manifest["current_stage"] = name
        dump_json(root / "protocol.json", manifest)
        start = time.perf_counter()
        environment = os.environ | {"PYTHONIOENCODING": "utf-8"}
        with (root / (Path(name).stem + ".log")).open("w", encoding="utf-8") as log:
            subprocess.run(
                command,
                check=True,
                cwd=REPOSITORY,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        manifest["stages"].append(
            {
                "script": name,
                "status": "complete",
                "arguments": list(map(str, arguments)),
                "elapsed_sec": time.perf_counter() - start,
            }
        )
        dump_json(root / "protocol.json", manifest)

    try:
        frames = root / "frames"
        observations = root / "observations"
        semantics = root / "semantics"
        foreground = root / "motion-filtered"
        geometry = root / "geometry-associated-final"
        candidates = root / "candidates-final.json"
        joint = root / "joint-final"
        run("motion_frames.py", ["--config", root / "resolved-config.json", "--output", frames])
        run(
            "motion_observations.py",
            [
                "--frames",
                frames / "frames.json",
                "--pose-weights",
                args.pose_weights,
                "--weights-root",
                args.weights_root,
                "--device",
                args.device,
                "--output",
                observations,
            ],
        )
        run(
            "motion_intel_features.py",
            [
                "--observations",
                observations / "observations.jsonl",
                "--weights-root",
                args.weights_root,
                "--output",
                root / "observations-dual.jsonl",
            ],
            args.cpu_python,
        )
        run(
            "motion_floor.py",
            [
                "--frames",
                frames / "frames.json",
                "--observations",
                root / "observations-dual.jsonl",
                "--lenses",
                args.lenses,
                "--hf-cache",
                args.hf_cache,
                "--device",
                args.device,
                "--output",
                semantics,
            ],
        )
        run(
            "motion_foreground.py",
            [
                "--frames",
                frames / "frames.json",
                "--observations",
                semantics / "observations.jsonl",
                "--output",
                foreground,
            ],
        )
        run(
            "motion_geometry_matching.py",
            [
                "--observations",
                foreground / "observations.jsonl",
                "--lenses",
                args.lenses,
                "--output",
                geometry,
            ],
        )
        run(
            "motion_body_geometry.py",
            [
                "--observations",
                foreground / "observations.jsonl",
                "--lenses",
                args.lenses,
                "--output",
                root / "body-diagnostic-final",
            ],
        )
        common = ["--bank", args.bank, "--source-manifest", args.source_manifest]
        run("motion_candidates.py", [*common, "--output", root / "baseline-candidates.json"])
        run(
            "motion_candidates.py",
            [
                *common,
                "--motion-pairs",
                geometry / "pairs.json",
                "--lenses",
                args.lenses,
                "--output",
                candidates,
            ],
        )
        run(
            "motion_joint.py",
            [
                "--candidates",
                candidates,
                "--pairs",
                geometry / "pairs.json",
                "--plan-walls",
                args.plan_walls,
                "--output",
                joint,
            ],
        )
        run("motion_export.py", ["--joint", joint / "joint.json", "--output", root / "accepted"])
        manifest.update(status="complete", current_stage=None)
    except BaseException as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        dump_json(root / "protocol.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
