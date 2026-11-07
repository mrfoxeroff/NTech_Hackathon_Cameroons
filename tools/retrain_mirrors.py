"""Review, train and evaluate a versioned mirror hard-negative dataset.

Source frames are never overwritten. The reported camera-3 interval 120..240 s
is held out, including existing samples from that interval.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.cwd() / ".runtime/ultralytics"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
import av
import cv2
import numpy as np
import yaml
from ultralytics import YOLO

ROOT = Path("data/person_mirrors_v3")
REVIEW = Path("configs/mirror_annotation_review.json")
TIMES = {
    "train": [
        46,
        50,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        62,
        66,
        70,
        842,
        843,
        844,
        845,
        846,
        850,
        854,
        858,
        886,
        890,
        892,
        893,
        894,
        895,
        896,
        898,
        902,
    ],
    "val": [770, 772, 773, 774, 775, 778, 782, 786, 790, 794, 798, 802],
    "test": [149, 154, 159, 164, 169],
}


def prepare():
    if (ROOT / "candidates.json").exists():
        raise FileExistsError("Candidates already exist; reviewed IDs must remain stable")
    (ROOT / "source").mkdir(parents=True, exist_ok=True)
    model = YOLO("weights/yolo11x-seg.pt")
    records, tiles = [], []
    with av.open("data/video/camera_3.mkv") as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.thread_count = 4
        origin = float(next(container.decode(stream)).time)
        for split, times in TIMES.items():
            for seconds in times:
                container.seek(int((origin + seconds) / float(stream.time_base)), stream=stream)
                frame = next(f for f in container.decode(stream) if float(f.time) >= origin + seconds)
                im = frame.to_ndarray(format="bgr24")
                h, w = im.shape[:2]
                name = f"cam3_extra_{seconds:05d}"
                path = ROOT / "source" / (name + ".jpg")
                cv2.imwrite(str(path), im)
                r = model.predict(im, classes=[0], conf=0.2, imgsz=960, device=0, quantize=16, verbose=False)[
                    0
                ]
                detections = []
                tile = cv2.resize(im, (864, 648))
                for index, box in enumerate(r.boxes):
                    uid = f"{name}:{index}"
                    polygon = r.masks.xyn[index].tolist()
                    detections.append(dict(id=uid, polygon=polygon, confidence=float(box.conf[0])))
                    points = np.rint(np.array(polygon) * [864, 648]).astype(np.int32)
                    cv2.polylines(tile, [points], True, (0, 255, 0), 1)
                    x, y = points.min(0)
                    cv2.putText(tile, str(index), (int(x), int(y)), 0, 0.65, (0, 0, 255), 2)
                cv2.putText(tile, f"{name} {split}", (5, 635), 0, 0.75, (0, 255, 255), 2)
                tiles.append(tile)
                records.append(
                    dict(
                        path=path.as_posix(),
                        split=split,
                        camera_id="camera_3",
                        seconds=seconds,
                        actual_time_sec=float(frame.time) - origin,
                        width=w,
                        height=h,
                        detections=detections,
                    )
                )
                print(name, len(detections), flush=True)
    (ROOT / "candidates.json").write_text(json.dumps(records), encoding="utf-8")
    for start in range(0, len(tiles), 4):
        page = tiles[start : start + 4]
        while len(page) < 4:
            page.append(np.zeros_like(page[0]))
        cv2.imwrite(
            str(ROOT / f"review_{start // 4:02d}.jpg"), np.vstack([np.hstack(page[:2]), np.hstack(page[2:])])
        )


def build():
    if (ROOT / "images").exists():
        raise FileExistsError("Dataset already exists; use a new version")
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    for key, path in [
        ("source_candidates_sha256", ROOT / "candidates.json"),
        ("base_candidates_sha256", Path("data/person_finetune/candidates.json")),
    ]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != review[key]:
            raise ValueError(f"Reviewed source changed: {path}")
    base_review = json.loads(Path("configs/person_annotation_review.json").read_text())
    rejected = set(base_review["rejected_detection_ids"])
    base_records = {
        Path(r["path"]).name: r for r in json.loads(Path("data/person_finetune/candidates.json").read_text())
    }
    manifest = json.loads(Path("data/person_finetune/split_manifest.json").read_text())
    records = []
    for entry in manifest:
        rec = dict(base_records[entry["image"]])
        rec["camera_id"] = Path(rec["path"]).parent.name
        rec["split"] = entry["split"]
        stem = Path(rec["path"]).stem
        match = re.search(r"_(\d+)m(\d+)s", stem)
        seconds = int(match[1]) * 60 + int(match[2])
        if rec["camera_id"] == "camera_3" and 120 <= seconds < 240:
            rec["split"] = "test"
        elif rec["camera_id"] == "camera_3" and 840 <= seconds < 960:
            rec["split"] = "train"
        elif rec["camera_id"] == "camera_3" and 720 <= seconds < 840:
            rec["split"] = "val"
        rec["detections"] = [d for d in rec["detections"] if d["id"] not in rejected]
        for i, poly in enumerate(base_review.get("added_polygons", {}).get(stem, [])):
            rec["detections"].append(dict(id=f"added:{stem}:{i}", polygon=poly))
        records.append(rec)
    for rec in json.loads((ROOT / "candidates.json").read_text()):
        rec["split"] = next(split for split, seconds in TIMES.items() if rec["seconds"] in seconds)
        records.append(rec)
    output_manifest = []
    for rec in records:
        path, split = Path(rec["path"]), rec["split"]
        labels, removed = [], []
        # Explicitly reject reviewed mirror people. Never erase every mirror pixel:
        # a physical person can stand in front of the mirror and remains positive.
        for detection in rec["detections"]:
            uid = str(detection["id"])
            if uid in review["rejected_detection_ids"]:
                removed.append(uid)
                continue
            polygon = detection["polygon"]
            if len(polygon) >= 3:
                labels.append("0 " + " ".join(f"{v:.7f}" for p in polygon for v in p))
        for polygon in review.get("added_polygons", {}).get(path.stem, []):
            labels.append("0 " + " ".join(f"{v:.7f}" for p in polygon for v in p))
        for kind in ("images", "labels"):
            (ROOT / kind / split).mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, ROOT / "images" / split / path.name)
        (ROOT / "labels" / split / (path.stem + ".txt")).write_text("\n".join(labels))
        output_manifest.append(
            dict(
                image=path.name,
                split=split,
                instances=len(labels),
                rejected=removed,
                source=rec["path"],
                camera_id=rec["camera_id"],
            )
        )
    (ROOT / "split_manifest.json").write_text(json.dumps(output_manifest, indent=2))
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
    print(
        {
            s: {
                "images": sum(r["split"] == s for r in output_manifest),
                "instances": sum(r["instances"] for r in output_manifest if r["split"] == s),
            }
            for s in ("train", "val", "test")
        }
    )


def train(mode="fresh"):
    presets = {
        "fresh": dict(
            initial="weights/yolo11s-seg.pt",
            epochs=50,
            patience=15,
            imgsz=640,
            name="person_v2",
            lr0=0.0003,
            mosaic=0.3,
            warmup_epochs=3,
        ),
        "incremental": dict(
            initial="weights/person-office-v1.pt",
            epochs=20,
            patience=8,
            imgsz=640,
            name="person_v2_incremental",
            lr0=0.0001,
            mosaic=0.0,
            warmup_epochs=3,
            save_period=5,
        ),
        "refined": dict(
            initial="weights/person-office-v2-fresh.pt",
            epochs=10,
            patience=15,
            imgsz=800,
            name="person_v2_refined",
            lr0=0.00003,
            mosaic=0.0,
            warmup_epochs=0,
        ),
    }
    parameters = presets[mode].copy()
    model = YOLO(parameters.pop("initial"))
    model.train(
        data=str(ROOT / "dataset.yaml"),
        batch=2,
        device=0,
        workers=0,
        cache=False,
        amp=False,
        project=str(Path("runs/mirrors").resolve()),
        seed=42,
        deterministic=True,
        optimizer="AdamW",
        freeze=10,
        close_mosaic=8,
        mixup=0.0,
        degrees=0.0,
        fliplr=0.0,
        plots=True,
        **parameters,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "build", "train", "train-incremental", "train-refined"])
    parser.add_argument("--dataset-root", type=Path, default=ROOT)
    args = parser.parse_args()
    ROOT = args.dataset_root
    {
        "prepare": prepare,
        "build": build,
        "train": train,
        "train-incremental": lambda: train("incremental"),
        "train-refined": lambda: train("refined"),
    }[args.action]()
