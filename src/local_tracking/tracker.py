"""Adapt ByteTrack assignments to actual detector observations, never predictions."""

import math
from types import SimpleNamespace

from .config import Settings, tracker_parameters


class LocalTracker:
    def __init__(self, settings: Settings, fps: float, appearance=False):
        from ultralytics.trackers.byte_tracker import BYTETracker

        backend = BYTETracker
        if appearance:
            from .appearance import AppearanceByteTracker

            backend = AppearanceByteTracker

        effective_fps = fps / settings.stride
        params = tracker_parameters(settings)
        # Ultralytics 8.4.203 uses processed-frame counts directly (no FPS
        # constructor parameter), and owns a separate ID counter per tracker.
        self.buffer_steps = max(1, math.ceil(settings.lost_track_sec * effective_fps))
        self.backend = backend(SimpleNamespace(**params, track_buffer=self.buffer_steps))
        self.lost_sec = settings.lost_track_sec
        self.identities = {}
        self.next_id = 1

    def update(self, boxes, image, timestamp_sec: float, features=None) -> dict[int, int]:
        """Return source detection index -> local ID; unmatched detections keep null ID."""
        assignments = self.backend.update(boxes, None if features is not None else image, feats=features)
        ids = {}
        for row in assignments:
            # Ultralytics BYTETracker output: xyxy, track_id, score, cls, detection_index.
            raw_id, index = int(row[4]), int(row[-1])
            if not 0 <= index < len(boxes) or index in ids:
                raise RuntimeError("Tracker returned invalid/duplicate detection index")
            old = self.identities.get(raw_id)
            if old is None or timestamp_sec - old[1] > self.lost_sec + 1e-9:
                local_id = self.next_id
                self.next_id += 1
            else:
                local_id = old[0]
            self.identities[raw_id] = (local_id, timestamp_sec)
            ids[index] = local_id
        return ids
