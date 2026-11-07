"""Explicitly download the two public motion-lab models into ignored local caches."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

LAB = Path(__file__).resolve().parent
POSE_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt"
POSE_SHA256 = "869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0"
SEGFORMER_REVISION = "489d5cd81a0b59fab9b7ea758d3548ebe99677da"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--download", action="store_true", help="Explicit permission to download public weights")
    args = p.parse_args()
    if not args.download:
        p.error("Use --download to fetch models; the offline demo does not need this step")
    pose = LAB / "experiments/scenealigner/cache/yolo11n-pose.pt"
    pose.parent.mkdir(parents=True, exist_ok=True)
    if not pose.exists():
        temporary = pose.with_suffix(".download")
        urllib.request.urlretrieve(POSE_URL, temporary)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != POSE_SHA256:
            raise ValueError("Pose checkpoint SHA-256 mismatch; incomplete file is not installed")
        temporary.replace(pose)
    if hashlib.sha256(pose.read_bytes()).hexdigest() != POSE_SHA256:
        raise ValueError("Existing pose checkpoint does not match the pinned model")
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        revision=SEGFORMER_REVISION,
        cache_dir=LAB / "neural_weights/huggingface",
        allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"],
    )
    print(
        json.dumps(
            {
                "pose": str(pose),
                "pose_sha256": POSE_SHA256,
                "segformer_snapshot": snapshot,
                "external_reid_weights_still_required": [
                    "osnet_x0_25_msmt17.pth",
                    "person-reidentification-retail-0288.xml",
                    "person-reidentification-retail-0288.bin",
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
