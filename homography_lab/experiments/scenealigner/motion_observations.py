"""Local pose/appearance observations, deliberately independent of global IDs and map hypotheses."""

import argparse
import importlib.util
import json
import os
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]


def visible_foot(box, keypoints, floor, others=()):
    box, keypoints = np.asarray(box, float), np.asarray(keypoints, float)
    height, width = floor.shape
    if keypoints.shape != (17, 3) or max(keypoints[15:17, 2]) < 0.7 and min(keypoints[15:17, 2]) < 0.55:
        return None, "uncertain_ankles"
    if min(box[:2]) < 2 or box[2] >= width - 2 or box[3] >= height - 2:
        return None, "truncated_body"
    feet = keypoints[15:17, :2][keypoints[15:17, 2] >= 0.55]
    if np.min(feet[:, 1]) < box[1] + 0.55 * (box[3] - box[1]):
        return None, "implausible_ankle_position"
    point = np.mean(feet, axis=0)
    x, y = np.rint(point).astype(int)
    if not (0 <= x < width and 0 <= y < height) or floor[y, x] == 0:
        return None, "not_on_visible_floor"
    for other in others:
        if other[0] <= x <= other[2] and other[1] <= y <= other[3] and other[3] > y + 5:
            return None, "occluded_by_another_detection"
    return point, None


def iou(a, b):
    lo, hi = np.maximum(a[:2], b[:2]), np.minimum(a[2:], b[2:])
    intersection = np.maximum(hi - lo, 0).prod()
    return float(intersection / max((a[2:] - a[:2]).prod() + (b[2:] - b[:2]).prod() - intersection, 1e-9))


class LocalTracks:
    def __init__(self):
        self.tracks, self.next_id = {}, 1

    def update(self, detections, timestamp, diagonal):
        self.tracks = {k: v for k, v in self.tracks.items() if timestamp - v["time"] <= 1.5}
        ids = list(self.tracks)
        cost = np.full((len(ids), len(detections)), 10.0)
        for i, key in enumerate(ids):
            old = self.tracks[key]
            for j, current in enumerate(detections):
                similarity = float(old["embedding"] @ current["embedding"])
                overlap = iou(old["box"], current["box"])
                displacement = (
                    np.linalg.norm(
                        np.mean(old["box"].reshape(2, 2), axis=0)
                        - np.mean(current["box"].reshape(2, 2), axis=0)
                    )
                    / diagonal
                )
                if similarity >= 0.6 and (overlap >= 0.1 or displacement <= 0.04):
                    cost[i, j] = 0.55 * (1 - overlap) + 0.45 * (1 - similarity)
        matched = {}
        if cost.size:
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] < 0.85:
                    matched[j] = ids[i]
        for j, detection in enumerate(detections):
            key = matched.get(j)
            if key is None:
                key, self.next_id = self.next_id, self.next_id + 1
            self.tracks[key] = {
                "time": timestamp,
                "box": detection["box"],
                "embedding": detection["embedding"],
            }
            detection["local_id"] = key


def load_reid(weights_root=None, device="cuda"):
    import torch

    weights = Path(weights_root) if weights_root else WORKSPACE / "mtmc/weights"
    architecture = HERE / "vendor_sources/torchreid_osnet.py"
    spec = importlib.util.spec_from_file_location("motion_osnet", architecture)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = torch.load(weights / "osnet_x0_25_msmt17.pth", map_location="cpu", weights_only=True)
    model = module.osnet_x0_25(num_classes=state["classifier.weight"].shape[0], pretrained=False)
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


def appearances(model, image, boxes):
    import torch

    device = next(model.parameters()).device
    inputs = []
    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)
        crop = image[max(0, y1) : y2, max(0, x1) : x2]
        crop = cv2.resize(crop, (128, 256))[:, :, ::-1].copy()
        inputs.append(torch.from_numpy(crop).permute(2, 0, 1).float() / 255)
    batch = torch.stack(inputs).to(device)
    batch = (batch - torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]) / torch.tensor(
        [0.229, 0.224, 0.225], device=device
    )[None, :, None, None]
    with torch.inference_mode():
        return torch.nn.functional.normalize(model(batch).float(), dim=-1).cpu().numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--pose-weights", type=Path, default=HERE / "cache/yolo11n-pose.pt")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--weights-root", type=Path, default=WORKSPACE / "mtmc/weights")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["YOLO_CONFIG_DIR"] = str(HERE / "cache/ultralytics-motion")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    import torch
    from ultralytics import YOLO, settings

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    settings.update({"sync": False})
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    previous = Path.cwd()
    try:
        os.chdir(args.pose_weights.resolve().parent)
        pose = YOLO(args.pose_weights.name)
    finally:
        os.chdir(previous)
    reid = load_reid(args.weights_root, device)
    manifest = json.loads(args.frames.read_text())
    if manifest["status"] != "complete":
        raise ValueError("Frame sampling incomplete")
    masks, trackers = {}, {}
    counts = {"frames": 0, "detections": 0, "usable_feet": 0, "rejections": {}}
    with (args.output / "observations.jsonl").open("x", encoding="utf-8") as out:
        for index, frame in enumerate(manifest["frames"]):
            camera, window = frame["camera"], frame["window"]
            image = cv2.imread(frame["image"])
            size = tuple(image.shape[1::-1])
            if camera not in masks:
                mask = cv2.imread(manifest["config"]["cameras"][camera]["floor_mask"], 0)
                mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
                dilation = max(3, round(0.007 * np.hypot(*size)))
                masks[camera] = cv2.dilate(mask, np.ones((dilation, dilation), np.uint8))
            result = pose.predict(
                image, imgsz=args.imgsz, device=0 if device == "cuda" else "cpu", conf=0.35, verbose=False
            )[0]
            counts["frames"] += 1
            detections = []
            boxes = result.boxes.xyxy.cpu().numpy()
            keypoints = result.keypoints.data.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            for j, (box, points, confidence) in enumerate(zip(boxes, keypoints, confidences)):
                counts["detections"] += 1
                foot, reason = visible_foot(box, points, masks[camera], np.delete(boxes, j, axis=0))
                if foot is None:
                    counts["rejections"][reason] = counts["rejections"].get(reason, 0) + 1
                detections.append(
                    {
                        "box": box,
                        "foot": foot,
                        "keypoints": points,
                        "confidence": float(confidence),
                        "foot_confidence": float(np.mean(points[15:17, 2][points[15:17, 2] >= 0.55]))
                        if foot is not None
                        else 0.0,
                        "foot_rejection": reason,
                    }
                )
            if detections:
                embeddings = appearances(reid, image, [d["box"] for d in detections])
                for detection, embedding in zip(detections, embeddings):
                    detection["embedding"] = embedding
                tracker = trackers.setdefault((camera, window), LocalTracks())
                tracker.update(detections, frame["timestamp_sec"], np.hypot(*size))
                for detection in detections:
                    row = {
                        key: value.tolist() if isinstance(value, np.ndarray) else value
                        for key, value in detection.items()
                    }
                    row.update(
                        frame_id=frame["id"],
                        camera=camera,
                        window=window,
                        split=frame["split"],
                        timestamp_sec=frame["timestamp_sec"],
                        image=frame["image"],
                        size=list(size),
                        source_pts=frame["source_pts"],
                        clock_interpolation_uncertainty_sec=frame["clock_interpolation_uncertainty_sec"],
                        track_id=f"{camera}:{window}:{detection['local_id']}",
                    )
                    out.write(json.dumps(row, separators=(",", ":")) + "\n")
                    counts["usable_feet"] += int(detection["foot"] is not None)
            if index % 100 == 0:
                print(index, camera, window, counts, flush=True)
    report = {
        "status": "complete",
        "counts": counts,
        "manual_plan_landmarks_used": False,
        "global_track_IDs_used": False,
        "pose_model": str(args.pose_weights.resolve()),
        "reid": "osnet_x0_25_msmt17_appearance_only",
        "foot_definition": "mean_confident_ankles_both_055_or_one_070_static_floor_gate_not_measured_sole_contact",
        "frames_manifest": str(args.frames.resolve()),
    }
    (args.output / "observations.json").write_text(json.dumps(report, indent=2) + "\n")
    print(counts, flush=True)


if __name__ == "__main__":
    main()
