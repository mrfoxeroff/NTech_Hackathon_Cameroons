"""Sequential, timestamp-preserving video decoding with PyAV.

Coordinates and images always refer to the original decoded resolution.  Times
are seconds relative to the first decoded frame, plus the configured offset.
No seeking is performed, so ``index`` remains the original decoded frame index.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    duration_sec: float | None
    time_base: float | None
    stream_start_sec: float | None
    first_frame_time_sec: float | None = None


@dataclass(frozen=True)
class VideoFrame:
    index: int
    video_time_sec: float
    timestamp_sec: float
    timestamp_source: str
    image: np.ndarray
    media_time_sec: float | None = None


def _video_stream(container):
    if not container.streams.video:
        raise ValueError("Input has no video stream")
    stream = container.streams.video[0]
    # Frame threading preserves presentation order while substantially reducing
    # decoding overhead on the supplied 2592x1944 recordings. Bound workers so
    # decoding does not consume every CPU core alongside inference.
    if hasattr(stream, "codec_context"):
        stream.thread_type = "AUTO"
        stream.codec_context.thread_count = 4
    return stream


def _fps(stream) -> float:
    for candidate in (stream.average_rate, stream.guessed_rate, stream.base_rate):
        try:
            rate = float(candidate)
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if math.isfinite(rate) and rate > 0:
            return rate
    raise ValueError("Video has no valid positive frame rate")


def probe_video(path: str | Path) -> VideoInfo:
    """Read metadata and decode one frame to reject empty/unusable recordings."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with av.open(str(path)) as container:
        stream = _video_stream(container)
        fps = _fps(stream)
        first = next(container.decode(stream), None)
        if first is None:
            raise ValueError(f"Video contains no decodable frames: {path}")
        if first.width <= 0 or first.height <= 0:
            raise ValueError(f"Video has invalid dimensions: {path}")
        time_base = float(stream.time_base) if stream.time_base is not None else None
        duration = None
        if stream.duration is not None and time_base is not None:
            duration = float(stream.duration) * time_base
        elif container.duration is not None:
            duration = float(container.duration) / av.time_base
        if duration is not None and (not math.isfinite(duration) or duration <= 0):
            duration = None
        start = (
            float(stream.start_time) * time_base
            if stream.start_time is not None and time_base is not None
            else None
        )
        first_base = first.time_base if first.time_base is not None else stream.time_base
        first_frame_time = (
            float(first.pts * first_base) if first.pts is not None and first_base is not None else None
        )
        return VideoInfo(
            path=str(path),
            width=first.width,
            height=first.height,
            fps=fps,
            duration_sec=duration,
            time_base=time_base,
            stream_start_sec=start,
            first_frame_time_sec=first_frame_time,
        )


def iter_video(
    path: str | Path,
    time_offset_sec: float = 0.0,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    stride: int = 1,
) -> Iterator[VideoFrame]:
    """Yield frames in ``[start_sec, end_sec)`` on the video's local time axis.

    ``stride`` selects original indices divisible by stride; it does not reset at
    ``start_sec``. Bounds apply before the common-timeline offset. Missing PTS
    use original index / nominal FPS and are explicitly labelled ``index_fps``.
    Duplicate or backwards decoded timestamps are rejected even on skipped
    frames: silently accepting them would corrupt downstream synchronisation.
    """
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("stride must be a positive integer")
    if not math.isfinite(start_sec) or start_sec < 0:
        raise ValueError("start_sec must be finite and non-negative")
    if not math.isfinite(time_offset_sec):
        raise ValueError("time_offset_sec must be finite")
    if end_sec is not None and (not math.isfinite(end_sec) or end_sec <= start_sec):
        raise ValueError("end_sec must be finite and greater than start_sec")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with av.open(str(path)) as container:
        stream = _video_stream(container)
        fps = _fps(stream)
        origin = None
        previous = None
        decoded_any = False
        for index, frame in enumerate(container.decode(stream)):
            decoded_any = True
            base = frame.time_base if frame.time_base is not None else stream.time_base
            if frame.pts is not None and base is not None:
                raw_time = frame.pts * base
                media_time = float(raw_time)
                if origin is None:
                    # If initial frames lack PTS, anchor the first known PTS to
                    # its nominal original-frame position rather than resetting
                    # the timeline in the middle of the recording.
                    origin = raw_time - Fraction(index, 1) / Fraction(str(fps))
                video_time = float(raw_time - origin)
                source = "pts"
            else:
                video_time = index / fps
                source = "index_fps"
                media_time = None
            if not math.isfinite(video_time):
                raise ValueError(f"Non-finite timestamp at source frame {index}: {path}")
            if previous is not None and video_time <= previous:
                raise ValueError(
                    f"Non-increasing timestamp at source frame {index}: {video_time} <= {previous} ({path})"
                )
            previous = video_time
            if end_sec is not None and video_time >= end_sec:
                break
            if video_time < start_sec or index % stride:
                continue
            yield VideoFrame(
                index=index,
                video_time_sec=video_time,
                timestamp_sec=video_time + time_offset_sec,
                timestamp_source=source,
                image=frame.to_ndarray(format="bgr24"),
                media_time_sec=media_time,
            )
        if not decoded_any:
            raise ValueError(f"Video contains no decodable frames: {path}")
