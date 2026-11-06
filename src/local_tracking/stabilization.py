"""Per-local-ID display geometry with bounded, explicitly marked occlusion estimates."""

import math

import cv2
import numpy as np


class GeometryStabilizer:
    def __init__(self, width, height, occluders=(), max_prediction_sec=1.0, reset_gap_sec=1.0):
        self.width, self.height = width, height
        self.occluders = [np.asarray(p, np.float32) * [width, height] for p in occluders]
        self.max_prediction_sec = max_prediction_sec
        self.reset_gap_sec = reset_gap_sec
        self.states = {}

    def _context(self, bbox, others):
        x1, y1, x2, y2 = bbox
        pad = max(4.0, 0.05 * (y2 - y1))
        rectangle = np.array(
            [[x1 - pad, y1 - pad], [x2 + pad, y1 - pad], [x2 + pad, y2 + pad], [x1 - pad, y2 + pad]],
            np.float32,
        )
        area = max(1.0, (x2 - x1) * (y2 - y1))
        for poly in self.occluders:
            intersection, _ = cv2.intersectConvexConvex(rectangle, cv2.convexHull(poly.astype(np.float32)))
            if intersection / area > 0.01:
                return True
        for other in others:
            ox1, oy1, ox2, oy2 = other
            overlap = max(0, min(x2, ox2) - max(x1, ox1)) * max(0, min(y2, oy2) - max(y1, oy1))
            if overlap / area > 0.1:
                return True
        return False

    def update(self, observations, timestamp_sec):
        # Expire even IDs which have disappeared; no history crosses a source gap.
        self.states = {
            k: v for k, v in self.states.items() if timestamp_sec - v["time"] <= self.reset_gap_sec
        }
        for obs in observations:
            raw = np.asarray(obs["bbox_xyxy"], dtype=float)
            x1, y1, x2, y2 = raw
            w, h = x2 - x1, y2 - y1
            anchor = np.array([(x1 + x2) / 2, y1])
            obs["raw_footpoint_px"] = list(obs["footpoint_px"])
            obs["bbox_stabilized_xyxy"] = raw.tolist()
            obs["bbox_stabilization_status"] = "observed"
            obs["footpoint_status"] = "observed"
            obs["geometry_flags"] = []
            local_id = obs["local_track_id"]
            other_boxes = [o["bbox_xyxy"] for o in observations if o is not obs]
            context = self._context(raw, other_boxes)
            previous = self.states.get(local_id) if local_id is not None else None
            clipped = obs.get("touches_frame_bottom", False)
            if previous is None:
                if clipped:
                    obs["footpoint_px"] = None
                    obs["footpoint_status"] = "unknown"
                    obs["geometry_flags"].append("frame_bottom_clipped")
                if local_id is not None:
                    self.states[local_id] = dict(
                        anchor=anchor,
                        size=np.array([w, h]),
                        velocity=np.zeros(2),
                        time=timestamp_sec,
                        reliable_time=None if clipped else timestamp_sec,
                        occluded=False,
                    )
                continue
            dt = timestamp_sec - previous["time"]
            if dt <= 0:
                raise ValueError("Geometry timestamps must strictly increase within a track")
            predicted = previous["anchor"] + previous["velocity"] * min(dt, 0.2)
            reference_w, reference_h = previous["size"]
            top_consistent = abs(y1 - predicted[1]) < 0.25 * reference_h
            shrunk = h < 0.82 * reference_h
            continuing = previous["occluded"] and h < 0.9 * reference_h
            bottom_consistent = abs(y2 - (predicted[1] + reference_h)) < 0.2 * reference_h
            upper_occluded = (
                not clipped
                and (shrunk or continuing)
                and context
                and bottom_consistent
                and y1 - predicted[1] > 0.18 * reference_h
            )
            occluded = clipped or ((top_consistent or upper_occluded) and (shrunk or continuing) and context)
            alpha = 1 - math.exp(-dt / 0.10)
            if occluded:
                # Track the visible upper body, retain size from reliable observations.
                measurement = np.array([anchor[0], y2 - reference_h]) if upper_occluded else anchor
                stable_anchor = predicted + alpha * (measurement - predicted)
                size = previous["size"].copy()
                obs["bbox_stabilization_status"] = "estimated"
                obs["geometry_flags"].append("occlusion_suspected")
                reliable_time = timestamp_sec if upper_occluded and not clipped else previous["reliable_time"]
                if upper_occluded:
                    obs["geometry_flags"].append("upper_body_occluded")
            else:
                stable_anchor = predicted + alpha * (anchor - predicted)
                size_alpha = 1 - math.exp(-dt / 0.18)
                size = previous["size"] + size_alpha * (np.array([w, h]) - previous["size"])
                # Visible feet are positive evidence for the larger body extent;
                # don't let slow size smoothing cut them off before an occlusion.
                size = np.maximum(size, [w, h])
                reliable_time = timestamp_sec
            velocity = (stable_anchor - previous["anchor"]) / dt
            limit = max(h, reference_h) * 3
            velocity = 0.75 * previous["velocity"] + 0.25 * np.clip(velocity, -limit, limit)
            cx, top = stable_anchor
            sw, sh = size
            cx = float(np.clip(cx, 0.5, self.width - 0.5))
            top = float(np.clip(top, 0, self.height - 1))
            stable = [
                max(0.0, cx - sw / 2),
                max(0.0, top),
                min(float(self.width), cx + sw / 2),
                min(float(self.height), max(top + 1, top + sh)),
            ]
            obs["bbox_stabilized_xyxy"] = stable
            age = None if reliable_time is None else timestamp_sec - reliable_time
            if occluded and (age is None or age > self.max_prediction_sec):
                obs["footpoint_px"] = None
                obs["footpoint_status"] = "unknown"
                obs["geometry_flags"].append("prediction_horizon_exceeded")
            else:
                obs["footpoint_px"] = [(stable[0] + stable[2]) / 2, stable[3]]
                obs["footpoint_status"] = "estimated" if occluded and not upper_occluded else "observed"
            obs["footpoint_method"] = (
                "temporal_visible_lower_body"
                if upper_occluded
                else "temporal_bbox_prediction"
                if occluded
                else "temporal_bbox_bottom_center"
            )
            obs["geometry_estimate_age_sec"] = age
            self.states[local_id] = dict(
                anchor=stable_anchor,
                size=size,
                velocity=velocity,
                time=timestamp_sec,
                reliable_time=reliable_time,
                occluded=occluded,
            )
        return observations
