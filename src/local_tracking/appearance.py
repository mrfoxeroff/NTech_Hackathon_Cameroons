"""Same-camera color evidence measured on exclusive silhouette pixels."""

import cv2
import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker, STrack, parse_bboxes


def clothing_colors(result, image):
    boxes = result.boxes.xyxy.cpu().numpy()
    features = np.full((len(boxes), 2, 3), np.nan, dtype=np.float32)
    if not len(boxes):
        return features
    masks = result.masks.data
    height, width = image.shape[:2]
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(width, int(np.ceil(x2))), min(height, int(np.ceil(y2)))
        w, h = x2 - x1, y2 - y1
        if w < 14 or h < 45 or y1 == 0 or y2 == height:
            continue
        region = masks[:, y1:y2, x1:x2] > 0.5
        exclusive = (region[i] & (region.sum(dim=0) == 1)).cpu().numpy()
        lab = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2LAB)
        for part, (start, end) in enumerate(((0.23, 0.49), (0.56, 0.84))):
            section = slice(round(start * h), round(end * h))
            valid = exclusive[section]
            if np.count_nonzero(valid) >= max(45, 0.22 * valid.size):
                features[i, part] = np.median(lab[section][valid], axis=0)
    return features


class ClothingTrack(STrack):
    def __init__(self, xywh, score, cls, colors):
        super().__init__(xywh, score, cls)
        self.colors = colors.copy()

    def remember(self, detection):
        observed = np.isfinite(detection.colors).all(axis=1)
        existing = np.isfinite(self.colors).all(axis=1)
        self.colors[observed & ~existing] = detection.colors[observed & ~existing]
        shared = observed & existing
        # A mixed or suddenly truncated crop should not overwrite the gallery.
        shared &= np.linalg.norm(self.colors - detection.colors, axis=1) < 44
        self.colors[shared] = 0.85 * self.colors[shared] + 0.15 * detection.colors[shared]

    def update(self, new_track, frame_id):
        super().update(new_track, frame_id)
        self.remember(new_track)

    def re_activate(self, new_track, frame_id, new_id=False):
        super().re_activate(new_track, frame_id, new_id)
        self.remember(new_track)


class AppearanceByteTracker(BYTETracker):
    def init_track(self, results, colors=None):
        if not len(results):
            return []
        if colors is None:
            colors = np.full((len(results), 2, 3), np.nan, dtype=np.float32)
        return [
            ClothingTrack(box, score, cls, color)
            for box, score, cls, color in zip(
                parse_bboxes(results), results.conf, results.cls, colors, strict=True
            )
        ]

    def get_dists(self, tracks, detections):
        costs = super().get_dists(tracks, detections)
        if not tracks or not detections:
            return costs
        old = np.stack([track.colors for track in tracks])
        new = np.stack([det.colors for det in detections])
        distances = np.linalg.norm(old[:, None] - new[None, :], axis=-1)
        known = np.isfinite(distances)
        worst = np.max(np.where(known, distances, 0), axis=-1)
        costs += np.minimum(worst / 185, 0.28)
        costs[worst > 62] = 1
        return costs
