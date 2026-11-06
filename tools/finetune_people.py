"""Prepare reviewable teacher masks, build a time-separated dataset and fine-tune."""

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.cwd() / ".runtime/ultralytics"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
import cv2
import numpy as np
import yaml
from ultralytics import YOLO

ROOT = Path("data/person_finetune")


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "candidates.json").exists():
        raise FileExistsError("Candidate annotations already exist; do not invalidate reviewed detection IDs")
    model = YOLO("weights/yolo11x-seg.pt")
    records = []
    crops = []
    contexts = []
    for path in sorted(Path("data/image/samples").glob("camera_*/*.jpg")):
        im = cv2.imread(str(path))
        h, w = im.shape[:2]
        r = model.predict(im, classes=[0], conf=0.15, imgsz=960, device=0, quantize=16, verbose=False)[0]
        detections = []
        display = cv2.resize(im, (648, 486))
        for idx, box in enumerate(r.boxes):
            uid = len(crops)
            xy = box.xyxy[0].cpu().numpy().tolist()
            poly = r.masks.xyn[idx].tolist()
            detections.append(dict(id=uid, bbox=xy, confidence=float(box.conf[0]), polygon=poly))
            x1, y1, x2, y2 = map(int, xy)
            crop = im[max(0, y1 - 10) : min(h, y2 + 10), max(0, x1 - 10) : min(w, x2 + 10)]
            tile = np.full((220, 160, 3), 25, np.uint8)
            scale = min(160 / crop.shape[1], 185 / crop.shape[0])
            small = cv2.resize(crop, None, fx=scale, fy=scale)
            tile[: small.shape[0], : small.shape[1]] = small
            cv2.putText(tile, f"{uid} {float(box.conf[0]):.2f}", (3, 204), 0, 0.5, (0, 255, 255), 1)
            crops.append(tile)
            p1 = (round(x1 * 648 / w), round(y1 * 486 / h))
            p2 = (round(x2 * 648 / w), round(y2 * 486 / h))
            cv2.rectangle(display, p1, p2, (0, 255, 0), 1)
            cv2.putText(display, str(uid), p1, 0, 0.5, (0, 255, 255), 1)
        canvas = np.zeros((516, 648, 3), np.uint8)
        canvas[:486] = display
        cv2.putText(canvas, path.stem, (5, 508), 0, 0.65, (255, 255, 255), 1)
        contexts.append(canvas)
        records.append(dict(path=path.as_posix(), width=w, height=h, detections=detections))
        print(path.stem, len(detections), flush=True)
    (ROOT / "candidates.json").write_text(json.dumps(records), encoding="utf-8")
    for kind, tiles, cols, per_page in [("crops", crops, 8, 80), ("context", contexts, 3, 12)]:
        for start in range(0, len(tiles), per_page):
            page = tiles[start : start + per_page]
            while len(page) % cols:
                page.append(np.zeros_like(page[0]))
            sheet = np.vstack([np.hstack(page[i : i + cols]) for i in range(0, len(page), cols)])
            cv2.imwrite(str(ROOT / f"{kind}_{start // per_page:02d}.jpg"), sheet)


def build():
    review = json.loads(Path("configs/person_annotation_review.json").read_text())
    actual_hash = hashlib.sha256((ROOT / "candidates.json").read_bytes()).hexdigest()
    if actual_hash != review["source_candidates_sha256"]:
        raise ValueError("Candidate annotations changed; visual review must be repeated")
    if (ROOT / "images").exists():
        raise FileExistsError("Dataset already exists; use it as-is or prepare a separate version")
    rejected = set(review["rejected_detection_ids"])
    records = json.loads((ROOT / "candidates.json").read_text())
    manifest = []
    for rec in records:
        path = Path(rec["path"])
        m = re.search(r"_(\d+)m(\d+)s", path.stem)
        seconds = int(m[1]) * 60 + int(m[2])
        # Guard against near-adjacent frames straddling a split boundary.
        if seconds > 10 and (seconds % 120 < 10 or seconds % 120 > 110):
            continue
        # Entire two-minute windows across cameras stay in one split.
        block = seconds // 120
        split = "test" if block in (4, 9) else "val" if block in (2, 7) else "train"
        image_dir = ROOT / "images" / split
        label_dir = ROOT / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, image_dir / path.name)
        labels = []
        for d in rec["detections"]:
            if d["id"] not in rejected:
                polygon = d["polygon"]
                if len(polygon) >= 3:
                    labels.append("0 " + " ".join(f"{v:.7f}" for p in polygon for v in p))
        for poly in review.get("added_polygons", {}).get(path.stem, []):
            labels.append("0 " + " ".join(f"{v:.7f}" for p in poly for v in p))
        (label_dir / (path.stem + ".txt")).write_text("\n".join(labels), encoding="utf-8")
        manifest.append(dict(image=path.name, split=split, instances=len(labels), source=rec["path"]))
    (ROOT / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    (ROOT / "dataset.yaml").write_text(
        yaml.safe_dump(
            dict(
                path=ROOT.resolve().as_posix(),
                train="images/train",
                val="images/val",
                test="images/test",
                names={0: "person"},
            )
        )
    )


def train():
    model = YOLO("weights/yolo11s-seg.pt")
    model.train(
        data=str(ROOT / "dataset.yaml"),
        epochs=40,
        patience=10,
        imgsz=640,
        batch=2,
        device=0,
        workers=0,
        cache=False,
        amp=False,
        project="outputs_training",
        name="person_v1",
        seed=42,
        deterministic=True,
        optimizer="AdamW",
        lr0=0.0003,
        freeze=10,
        mosaic=0.3,
        close_mosaic=8,
        mixup=0.0,
        degrees=0.0,
        fliplr=0.0,
        plots=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["prepare", "build", "train"])
    args = p.parse_args()
    {"prepare": prepare, "build": build, "train": train}[args.action]()
