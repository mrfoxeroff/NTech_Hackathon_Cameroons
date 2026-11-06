"""Conservative scene filters applied before local tracking; never black out input pixels."""

from pathlib import Path

import cv2
import numpy as np


class SceneFilter:
    def __init__(self, regions, width, height):
        self.regions = []
        for region in regions:
            mask = np.zeros((height, width), np.uint8)
            points = np.rint(np.asarray(region["polygon"]) * [width, height]).astype(np.int32)
            cv2.fillPoly(mask, [points], 1)
            reference = None
            if region["mode"] == "static_object":
                reference = cv2.imdecode(np.fromfile(Path(region["reference"]), np.uint8), cv2.IMREAD_COLOR)
                if reference is None or reference.shape[:2] != (height, width):
                    raise ValueError("Scene reference must match source image resolution")
            self.regions.append((region, mask.astype(bool), reference))

    def apply(self, result, image):
        if not self.regions or not len(result.boxes):
            return result, []
        if result.masks is None:
            raise ValueError("Scene filtering requires segmentation masks")
        masks = result.masks.cpu().numpy().data > 0.5
        keep = []
        rejected = []
        for index, candidate in enumerate(masks):
            reason = None
            for region, zone, reference in self.regions:
                area = int(candidate.sum())
                inside = candidate & zone
                if not area or inside.sum() / area < region.get("min_overlap", 0.85):
                    continue
                if region["mode"] == "screen":
                    reason = region["name"]
                    break
                # A person in front of the furniture changes its appearance. Compare
                # only the candidate's pixels, compensate uniform lighting shifts.
                delta = image[candidate].astype(np.float32) - reference[candidate].astype(np.float32)
                sampled_delta = image[::16, ::16].astype(np.float32) - reference[::16, ::16].astype(
                    np.float32
                )
                outside = ~candidate[::16, ::16]
                if outside.any():
                    delta -= np.median(sampled_delta[outside], axis=0)
                changed = np.max(np.abs(delta), axis=1) > region.get("pixel_difference", 30)
                if float(changed.mean()) < region.get("min_changed_fraction", 0.15):
                    reason = region["name"]
                    break
            if reason is None:
                keep.append(index)
            else:
                rejected.append(
                    {
                        "source_detection_index": index,
                        "region": reason,
                        "confidence": float(result.boxes.conf[index]),
                    }
                )
        # Slice whole Results so boxes, masks and detection indices stay aligned.
        return result[np.asarray(keep, dtype=np.int64)], rejected
