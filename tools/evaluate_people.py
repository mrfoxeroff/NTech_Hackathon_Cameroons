"""Compare frozen baseline and candidate against reviewed, held-out instance boxes."""

import json
import os
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.cwd() / ".runtime/ultralytics"))
import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path("data/person_finetune")


def boxes_for(path):
    result = []
    for line in path.read_text().splitlines():
        points = np.array(list(map(float, line.split()[1:]))).reshape(-1, 2)
        result.append([*points.min(0), *points.max(0)])
    return result


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return intersection / max(
        1e-12, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    )


def evaluate(weights, split, conf):
    model = YOLO(str(weights))
    tp = fp = fn = negative_fp = 0
    rows = []
    for path in sorted((ROOT / "images" / split).glob("*.jpg")):
        im = cv2.imread(str(path))
        h, w = im.shape[:2]
        gt = boxes_for(ROOT / "labels" / split / (path.stem + ".txt"))
        r = model.predict(im, classes=[0], conf=conf, imgsz=640, device=0, quantize=16, verbose=False)[0]
        pred = r.boxes.xyxyn.cpu().numpy().tolist()
        matched = set()
        hits = 0
        for p in pred:
            choices = [(iou(p, g), i) for i, g in enumerate(gt) if i not in matched]
            score, index = max(choices, default=(0, -1))
            if score >= 0.5:
                matched.add(index)
                hits += 1
        tp += hits
        fp += len(pred) - hits
        fn += len(gt) - hits
        if not gt:
            negative_fp += len(pred)
        rows.append(
            dict(
                image=path.name,
                true=len(gt),
                predicted=len(pred),
                tp=hits,
                fp=len(pred) - hits,
                fn=len(gt) - hits,
            )
        )
    return dict(
        weights=str(weights),
        split=split,
        confidence=conf,
        tp=tp,
        fp=fp,
        fn=fn,
        precision=tp / max(1, tp + fp),
        recall=tp / max(1, tp + fn),
        negative_frame_fp=negative_fp,
        images=rows,
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("weights", type=Path)
    args = p.parse_args()
    result = []
    for split in ["val", "test"]:
        for conf in [0.1, 0.3]:
            metrics = evaluate(args.weights, split, conf)
            result.append(metrics)
            print({k: v for k, v in metrics.items() if k != "images"}, flush=True)
    out = ROOT / (args.weights.stem + "_evaluation.json")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
