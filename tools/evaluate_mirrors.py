"""Compare raw models and the complete scene-filter policy at one confidence."""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.cwd() / ".runtime/ultralytics"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
import cv2
import numpy as np
from evaluate_people import boxes_for, iou
from ultralytics import YOLO

from local_tracking.config import load_settings
from local_tracking.scene_filter import SceneFilter


def count(result, truth, mirror_zones):
    matched = set()
    mirror_fp = 0
    for index in np.argsort(-result.boxes.conf.cpu().numpy()):
        pred = result.boxes.xyxyn[index].cpu().numpy()
        score, target = max(
            ((iou(pred, gt), n) for n, gt in enumerate(truth) if n not in matched), default=(0, -1)
        )
        if score >= 0.5:
            matched.add(target)
        elif result.masks is not None:
            mask = result.masks.data[index].cpu().numpy() > 0.5
            # Diagnostic category only; it does not change matching or filtering.
            mirror_fp += int(any((mask & zone).sum() / max(1, mask.sum()) >= 0.3 for zone in mirror_zones))
    return dict(
        tp=len(matched),
        fp=len(result.boxes) - len(matched),
        fn=len(truth) - len(matched),
        mirror_fp=mirror_fp,
    )


def evaluate(weights, root, conf, splits=("val", "test"), imgsz=None):
    settings = load_settings(Path("configs/cameras.yaml"))
    imgsz = imgsz or settings.imgsz
    cameras = {c.camera_id: c for c in settings.cameras}
    entries = json.loads((root / "split_manifest.json").read_text())
    model = YOLO(str(weights))
    filters, rows = {}, []
    for entry in entries:
        if entry["split"] not in splits:
            continue
        path = root / "images" / entry["split"] / entry["image"]
        im = cv2.imread(str(path))
        height, width = im.shape[:2]
        camera = entry["camera_id"]
        if camera not in filters:
            filters[camera] = SceneFilter(cameras[camera].scene_regions, width, height)
        filt = filters[camera]
        zones = [z for r, z, _ in filt.regions if r["mode"] == "reflection"]
        truth = boxes_for(root / "labels" / entry["split"] / (path.stem + ".txt"))
        result = model.predict(
            im,
            classes=[0],
            conf=conf,
            iou=0.7,
            imgsz=imgsz,
            device=0,
            quantize=16,
            retina_masks=True,
            verbose=False,
        )[0]
        raw = count(result, truth, zones)
        filtered, rejected = filt.apply(result, im)
        rows.append(
            dict(
                image=path.name,
                split=entry["split"],
                camera_id=camera,
                truth=len(truth),
                raw=raw,
                filtered=count(filtered, truth, zones),
                rejected=rejected,
            )
        )
    totals = {}
    for split in splits:
        totals[split] = {}
        for mode in ("raw", "filtered"):
            counts = {
                k: sum(r[mode][k] for r in rows if r["split"] == split)
                for k in ("tp", "fp", "fn", "mirror_fp")
            }
            counts["precision"] = counts["tp"] / max(1, counts["tp"] + counts["fp"])
            counts["recall"] = counts["tp"] / max(1, counts["tp"] + counts["fn"])
            totals[split][mode] = counts
    return dict(
        weights=str(weights),
        weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256(Path("configs/cameras.yaml").read_bytes()).hexdigest(),
        confidence=conf,
        imgsz=imgsz,
        bbox_match_iou=0.5,
        summary=totals,
        images=rows,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", type=Path, nargs="+")
    parser.add_argument("--dataset", type=Path, default=Path("data/person_mirrors_v3"))
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--split", nargs="+", choices=["val", "test"], default=["val", "test"])
    parser.add_argument("--imgsz", type=int)
    args = parser.parse_args()
    for weights in args.weights:
        metrics = evaluate(weights, args.dataset, args.confidence, args.split, args.imgsz)
        suffix = f"_{args.imgsz}" if args.imgsz else ""
        out = args.dataset / (weights.stem + suffix + "_mirror_evaluation.json")
        out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(weights, json.dumps(metrics["summary"], indent=2), flush=True)
