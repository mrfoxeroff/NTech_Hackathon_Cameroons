import av
import numpy as np

from local_tracking.artifacts import PreviewWriter
from local_tracking.video import VideoFrame


def test_source_timestamp_gap_is_black_instead_of_stale_detection(tmp_path):
    path = tmp_path / "preview.mp4"
    writer = PreviewWriter(path, 640, 480, 10, 640)
    image = np.full((480, 640, 3), 220, np.uint8)
    writer.add(VideoFrame(0, 0.0, 0.0, "pts", image), [], "cam")
    writer.add(VideoFrame(1, 2.0, 2.0, "pts", image), [], "cam")
    writer.close()
    with av.open(str(path)) as container:
        decoded = [frame.to_ndarray(format="bgr24") for frame in container.decode(video=0)]
    assert len(decoded) == 21
    assert decoded[0].mean() > 150
    assert decoded[10].mean() < 20  # Missing source data, no held observation.
    assert decoded[-1].mean() > 150


def test_estimated_box_with_unknown_feet_renders_without_fabricated_point(tmp_path):
    path = tmp_path / "unknown.mp4"
    writer = PreviewWriter(path, 640, 480, 10, 640)
    image = np.zeros((480, 640, 3), np.uint8)
    observation = dict(
        local_track_id=1,
        bbox_xyxy=[10, 10, 50, 80],
        bbox_stabilized_xyxy=[10, 10, 50, 140],
        bbox_stabilization_status="estimated",
        footpoint_px=None,
        footpoint_status="unknown",
        silhouette_polygons=[],
        detection_confidence=0.9,
    )
    writer.add(VideoFrame(0, 0.0, 0.0, "pts", image), [observation], "cam")
    writer.close()
    with av.open(str(path)) as container:
        assert len(list(container.decode(video=0))) == 1
