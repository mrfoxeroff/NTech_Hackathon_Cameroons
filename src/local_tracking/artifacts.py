"""Lossless ROI masks, clean crops, local summaries and timestamped diagnostics."""

import json
import math
from pathlib import Path

import cv2
import numpy as np


def write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def save_image(path: Path, image, extension: str):
    # imencode + tofile also handles non-ASCII Windows paths.
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise RuntimeError(f"Cannot encode {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))


def observation_geometry(bbox, width, height):
    values = np.asarray(bbox, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("Invalid detection coordinates")
    x1, y1, x2, y2 = values.tolist()
    x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
    y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Empty detection box")
    border = {"left": x1 <= 1, "top": y1 <= 1, "right": x2 >= width - 1, "bottom": y2 >= height - 1}
    roi = [math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)]
    return [x1, y1, x2, y2], roi, [(x1 + x2) / 2, y2], border


class TrackSummary:
    def __init__(self, selected_count: int):
        self.tracks = {}
        self.selected_count = selected_count

    def add(self, observation, timestamp: float, step: int, source_gap: bool = False):
        local_id = observation["local_track_id"]
        if local_id is None:
            return
        if local_id not in self.tracks:
            self.tracks[local_id] = {
                "local_track_id": local_id,
                "first_seen_sec": timestamp,
                "last_seen_sec": timestamp,
                "observation_count": 0,
                "visibility_intervals": [],
                "selected_crops": [],
                "_last_step": -2,
            }
        item = self.tracks[local_id]
        if source_gap or item["_last_step"] != step - 1:
            item["visibility_intervals"].append({"start_sec": timestamp, "end_sec": timestamp})
        else:
            item["visibility_intervals"][-1]["end_sec"] = timestamp
        item["_last_step"] = step
        item["last_seen_sec"] = timestamp
        item["observation_count"] += 1
        if observation["crop_ref"]:
            item["selected_crops"].append(
                {
                    "crop_ref": observation["crop_ref"],
                    "observation_id": observation["observation_id"],
                    "timestamp_sec": timestamp,
                    "quality": observation["crop_quality"],
                }
            )
            item["selected_crops"].sort(key=lambda x: x["quality"]["selection_score"], reverse=True)
            del item["selected_crops"][self.selected_count :]

    def export(self):
        return [{k: v for k, v in t.items() if not k.startswith("_")} for _, t in sorted(self.tracks.items())]


class ObservationWriter:
    def __init__(self, output: Path, crop_interval_sec: float, save_masks: bool):
        self.output = output
        self.crop_interval_sec = crop_interval_sec
        self.save_masks = save_masks
        self.last_crop = {}

    def build(self, result, assignments, frame, recording_id, camera_id):
        image = frame.image
        height, width = image.shape[:2]
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        masks = result.masks.data.cpu().numpy() if result.masks is not None else None
        if masks is None and len(boxes):
            raise RuntimeError("Segmentation model returned boxes without masks")
        if masks is not None and len(masks) != len(boxes):
            raise RuntimeError("Masks and source detections are not aligned")
        observations = []
        for index, (bbox, confidence) in enumerate(zip(boxes, scores)):
            bbox, roi, foot, borders = observation_geometry(bbox, width, height)
            left, top, right, bottom = roi
            local_id = assignments.get(index)
            stem = f"{frame.index:08d}_{index:03d}"
            obs = {
                "observation_id": f"{recording_id}:{camera_id}:{frame.index}:{index}",
                "local_track_id": local_id,
                "bbox_xyxy": bbox,
                "detection_confidence": float(confidence),
                "mask_ref": None,
                "mask_origin_px": [left, top],
                "mask_size": [right - left, bottom - top],
                "silhouette_polygons": [],
                "footpoint_px": foot,
                "footpoint_method": "bbox_bottom_center",
                "touches_frame_bottom": borders["bottom"],
                "border_flags": borders,
                "crop_ref": None,
                "crop_quality": None,
            }
            # retina_masks=True guarantees full source resolution. Never resize letterboxed masks blindly.
            mask = masks[index]
            if mask.shape != (height, width):
                raise RuntimeError(f"Expected source-sized retina mask, got {mask.shape}")
            roi_mask = (mask[top:bottom, left:right] > 0.5).astype(np.uint8) * 255
            contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if len(contour) < 3:
                    continue
                simplified = cv2.approxPolyDP(contour, 1.0, True).reshape(-1, 2)
                if len(simplified) >= 3:
                    obs["silhouette_polygons"].append((simplified + [left, top]).tolist())
            if self.save_masks:
                obs["mask_ref"] = f"masks/{stem}.png"
                save_image(self.output / obs["mask_ref"], roi_mask, ".png")
            crop_key = local_id if local_id is not None else f"untracked:{stem}"
            if frame.timestamp_sec - self.last_crop.get(crop_key, -math.inf) >= self.crop_interval_sec - 1e-9:
                crop = image[top:bottom, left:right]
                sharpness = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
                # Ranking heuristic, not a calibrated probability of crop correctness.
                score = float(confidence) * math.sqrt(crop.shape[0] * crop.shape[1])
                score *= min(1.0, sharpness / 100) * (0.5 if any(borders.values()) else 1.0)
                obs["crop_quality"] = {
                    "width": crop.shape[1],
                    "height": crop.shape[0],
                    "laplacian_variance": sharpness,
                    "truncated": any(borders.values()),
                    "selection_score": score,
                }
                obs["crop_ref"] = f"crops/{stem}.jpg"
                save_image(self.output / obs["crop_ref"], crop, ".jpg")
                self.last_crop[crop_key] = frame.timestamp_sec
            observations.append(obs)
        return observations


class PreviewWriter:
    """CFR diagnostic video retimed to source timestamps; JSONL is authoritative."""

    def __init__(self, path, source_width, source_height, fps, max_width):
        scale = min(1.0, max_width / source_width)
        self.size = (max(2, int(source_width * scale) // 2 * 2), max(2, int(source_height * scale) // 2 * 2))
        self.fps = fps
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, self.size)
        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot open diagnostic video writer: {path}")
        self.first_time = None
        self.previous = None
        self.next_tick = 0
        self.count = 0

    def add(self, frame, observations, camera_id):
        image = cv2.resize(frame.image, self.size)
        scale = np.array([self.size[0] / frame.image.shape[1], self.size[1] / frame.image.shape[0]])
        for obs in observations:
            local_id = obs["local_track_id"]
            palette = [(0, 255, 80), (255, 180, 0), (255, 0, 255), (0, 100, 255), (255, 255, 0)]
            color = (0, 200, 255) if local_id is None else palette[(local_id - 1) % len(palette)]
            overlay = image.copy()
            polygons = [(np.asarray(p) * scale).astype(np.int32) for p in obs["silhouette_polygons"]]
            if polygons:
                cv2.fillPoly(overlay, polygons, color)
                image = cv2.addWeighted(overlay, 0.25, image, 0.75, 0)
            x1, y1, x2, y2 = (np.asarray(obs["bbox_xyxy"]) * np.tile(scale, 2)).astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"local={local_id if local_id is not None else '?'} {obs['detection_confidence']:.2f}"
            cv2.putText(image, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.circle(
                image, tuple((np.asarray(obs["footpoint_px"]) * scale).astype(int)), 4, (0, 0, 255), -1
            )
        cv2.rectangle(image, (0, self.size[1] - 40), (self.size[0], self.size[1]), (0, 0, 0), -1)
        cv2.putText(
            image,
            f"{camera_id} t={frame.timestamp_sec:.3f}s frame={frame.index}",
            (15, self.size[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        if self.first_time is None:
            self.first_time = frame.video_time_sec
        tick = max(0, round((frame.video_time_sec - self.first_time) * self.fps))
        if self.previous is not None:
            gap = tick - self.next_tick
            filler = self.previous
            if gap > max(3, self.fps * 0.5):
                filler = np.zeros_like(image)
                cv2.putText(
                    filler,
                    "NO SOURCE FRAMES (timestamp gap)",
                    (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
            while self.next_tick < tick:
                # The previous actual frame is valid for its own tick only;
                # don't erase it when the next source frame follows a long gap.
                self.writer.write(self.previous if gap == tick - self.next_tick else filler)
                self.next_tick += 1
                self.count += 1
        self.previous = image

    def close(self):
        if self.previous is not None:
            self.writer.write(self.previous)
            self.count += 1
        self.writer.release()
        self.previous = None
