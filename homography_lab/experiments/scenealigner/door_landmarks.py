"""Open-vocabulary door hypotheses, automatically tied to nearby visible floor."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from floor_ray_geometry import project_floor, transform_points
from PIL import Image


def door_floor_anchor(box, floor):
    x1, y1, x2, y2 = np.asarray(box, float)
    height, width = floor.shape
    if x2 - x1 < 0.035 * width or y2 - y1 < 0.12 * height:
        return None
    # Search near the predicted bottom, never along an arbitrary unseen vertical extension.
    xs = np.linspace(x1 + 0.2 * (x2 - x1), x2 - 0.2 * (x2 - x1), 11).round().astype(int).clip(0, width - 1)
    low = max(0, int(y2 - 0.14 * (y2 - y1)))
    high = min(height, int(y2 + 0.14 * (y2 - y1)) + 1)
    if high <= low:
        return None
    samples = []
    for x in xs:
        ys = np.flatnonzero(floor[low:high, x] > 0) + low
        if len(ys):
            y = ys[np.argmin(abs(ys - y2))]
            samples.append([float(x), float(y)])
    if len(samples) < 6:
        return None
    points = np.asarray(samples)
    return np.median(points, axis=0)


def main():
    import run_experiment as runtime
    import torch

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--registration", type=Path, required=True)
    p.add_argument("--model-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    features = json.loads((args.features / "features.json").read_text())
    geometry = dict(np.load(args.features / "geometry.npz"))
    registration = json.loads(args.registration.read_text())["cases"]
    manifest = json.loads(args.model_manifest.read_text())
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(manifest["local_snapshot"], local_files_only=True)
    model = (
        AutoModelForZeroShotObjectDetection.from_pretrained(
            manifest["local_snapshot"], local_files_only=True, use_safetensors=True
        )
        .eval()
        .cuda()
    )
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    report = {
        "manual_landmarks_used": False,
        "reference_JSONs_opened": False,
        "model": manifest,
        "prompt": "a door. an open doorway.",
        "threshold": 0.3,
        "text_threshold": 0.25,
        "cases": [],
        "training_landmarks": [],
    }
    for case in features["cases"]:
        image = Image.open(case["image"]).convert("RGB")
        inputs = processor(images=image, text=report["prompt"], return_tensors="pt").to("cuda")
        with torch.inference_mode():
            outputs = model(**inputs)
        detected = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=0.3, text_threshold=0.25, target_sizes=[image.size[::-1]]
        )[0]
        boxes, scores = detected["boxes"].cpu().numpy(), detected["scores"].cpu().numpy()
        keep = __import__("torchvision").ops.nms(detected["boxes"], detected["scores"], 0.45).cpu().numpy()
        floor = cv2.imread(str(args.features / case["id"] / "floor.png"), 0)
        annotated = np.array(image)[:, :, ::-1].copy()
        item = {"id": case["id"], "kind": case["kind"], "detections": []}
        for index in keep:
            box, score = boxes[index], float(scores[index])
            anchor = door_floor_anchor(box, floor)
            detection = {
                "box": box.tolist(),
                "score": score,
                "label": detected["text_labels"][index],
                "floor_anchor": None if anchor is None else anchor.tolist(),
            }
            item["detections"].append(detection)
            cv2.rectangle(annotated, tuple(box[:2].astype(int)), tuple(box[2:].astype(int)), (0, 200, 255), 5)
            if anchor is not None:
                cv2.circle(annotated, tuple(anchor.astype(int)), 12, (0, 0, 255), -1)
                if case["kind"] == "training":
                    inverse = cv2.invertAffineTransform(np.asarray(registration[case["id"]]["matrix"]))
                    common = inverse @ np.r_[anchor, 1.0]
                    endpoints = np.array([[box[0], anchor[1], 1.0], [box[2], anchor[1], 1.0]]) @ inverse.T
                    ground, valid = project_floor(endpoints, features["lens"], geometry["floor_homography"])
                    width = (
                        float(
                            np.linalg.norm(
                                transform_points(ground, geometry["normalization"])[1]
                                - transform_points(ground, geometry["normalization"])[0]
                            )
                        )
                        if valid.all()
                        else None
                    )
                    report["training_landmarks"].append(
                        {
                            "case_id": case["id"],
                            "point": common.tolist(),
                            "score": score,
                            "approximate_density_width": width,
                        }
                    )
        cv2.imwrite(str(args.output / (case["id"] + ".jpg")), annotated)
        report["cases"].append(item)
        print(case["id"], [(round(d["score"], 3), d["floor_anchor"]) for d in item["detections"]], flush=True)
    # One vote per training frame for each stable image-space door cluster.
    groups = []
    threshold = 0.04 * np.hypot(*features["lens"]["size"])
    for landmark in sorted(report["training_landmarks"], key=lambda v: -v["score"]):
        point = np.asarray(landmark["point"])
        group = next(
            (
                g
                for g in groups
                if np.linalg.norm(point - np.median([v["point"] for v in g], axis=0)) <= threshold
            ),
            None,
        )
        if group is None:
            groups.append([landmark])
        elif landmark["case_id"] not in {v["case_id"] for v in group}:
            group.append(landmark)
    stable = [np.median([v["point"] for v in g], axis=0) for g in groups if len(g) >= 3]
    report["stable_image_anchors"] = [p.tolist() for p in stable]
    report["support_frames"] = [len(g) for g in groups if len(g) >= 3]
    report["stable_density_widths"] = [
        float(
            np.median(
                [v["approximate_density_width"] for v in g if v["approximate_density_width"] is not None]
            )
        )
        if any(v["approximate_density_width"] is not None for v in g)
        else None
        for g in groups
        if len(g) >= 3
    ]
    report["width_limitation"] = "Approximation from detection box bottom extent; not a surveyed jamb width."
    if stable:
        ground, valid = project_floor(np.asarray(stable), features["lens"], geometry["floor_homography"])
        report["density_anchors"] = transform_points(ground[valid], geometry["normalization"]).tolist()
    else:
        report["density_anchors"] = []
    runtime.save_json(args.output / "doors.json", report)


if __name__ == "__main__":
    main()
