"""Export synchronized local observations from video for associate.py."""

import argparse
import json
import os
from pathlib import Path


def keyed(items, value_type):
    result = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not separator or not key or not value or key in result:
            raise ValueError(f"Expected a unique CAMERA=VALUE, got {item!r}")
        result[key] = value_type(value)
    return result


def torso_brightness(image, box, contour, cv2, np):
    x1, y1, x2, y2 = box
    height, width = y2 - y1, x2 - x1
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(contour, dtype=np.int32) - np.array([x1, y1], dtype=np.int32)], 255)
    top, bottom = int(height * 0.22), int(height * 0.52)
    pixels = image[y1 + top:y1 + bottom, x1:x2][mask[top:bottom] != 0]
    return round(float(np.median(pixels.max(axis=1))) / 255, 4) if len(pixels) else None


def extract(camera, path, offset, excluded, args, output, cv2, np, av, yolo, reid):
    detector = yolo(str(args.weights))
    last_embedding = {}
    count = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        origin = None
        if args.window_ms:
            if stream.start_time is None:
                raise ValueError(f"{path}: video has no start timestamp for window extraction")
            origin = stream.start_time * stream.time_base
            start_local = max(0, args.window_ms[0] - offset)
            container.seek(stream.start_time + int(start_local / (1000 * stream.time_base)),
                           stream=stream, backward=True)
        for frame_index, frame in enumerate(container.decode(stream)):
            if frame.pts is None:
                raise ValueError(f"{path}: frame {frame_index} has no presentation timestamp")
            pts = frame.pts * frame.time_base
            if origin is None:
                origin = pts
            local_ms = round((pts - origin) * 1000)
            if args.max_seconds is not None and local_ms >= args.max_seconds * 1000:
                break
            time_ms = local_ms + offset
            if args.window_ms:
                if time_ms >= args.window_ms[1]:
                    break
                if time_ms < args.window_ms[0]:
                    continue
            if time_ms < 0 or frame_index % args.stride:
                continue
            image = frame.to_ndarray(format="bgr24")
            result = detector.track(image, persist=True, tracker="bytetrack.yaml", classes=[0],
                                    imgsz=args.imgsz, device=args.device, verbose=False)[0]
            if result.boxes is None or result.boxes.id is None or result.masks is None:
                continue
            height, width = image.shape[:2]
            for local_id, box, confidence, polygon in zip(
                result.boxes.id.int().cpu().tolist(), result.boxes.xyxy.cpu().tolist(),
                result.boxes.conf.cpu().tolist(), result.masks.xy, strict=True,
            ):
                if len(polygon) < 3:
                    continue
                x1, y1, x2, y2 = map(round, box)
                x1, x2 = max(0, x1), min(width, x2)
                y1, y2 = max(0, y1), min(height, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                foot_x = (x1 + x2) / 2
                if any(left <= foot_x <= right and top <= y2 <= bottom
                       for left, top, right, bottom in excluded):
                    continue
                contour = np.rint(polygon).astype(np.int32)
                observation = {"camera_id": camera, "frame_index": frame_index,
                               "timestamp_ms": time_ms, "local_id": local_id,
                               "bbox_xyxy": [x1, y1, x2, y2], "confidence": round(confidence, 4),
                               "contour": contour.tolist()}
                if args.window_ms:
                    observation["source_pts"] = frame.pts
                if time_ms - last_embedding.get(local_id, -args.embedding_ms) >= args.embedding_ms:
                    if y2 - y1 >= 40:
                        crop = image[y1:y2, x1:x2]
                        # OMZ person-reidentification-retail-0288: BGR, NCHW, 256x128.
                        tensor = cv2.resize(crop, (128, 256)).transpose(2, 0, 1)[None].astype(np.float32)
                        embedding = reid([tensor])[reid.output(0)].reshape(-1)
                        norm = np.linalg.norm(embedding)
                        if np.isfinite(norm) and norm > 0:
                            observation["embedding"] = np.round(embedding / norm, 6).tolist()
                            observation["torso_brightness"] = torso_brightness(
                                image, observation["bbox_xyxy"], contour, cv2, np)
                            last_embedding[local_id] = time_ms
                output.write(json.dumps(observation, separators=(",", ":")) + "\n")
                count += 1
            if frame_index % (100 * args.stride) == 0:
                print(f"{camera}: frame {frame_index}, {count} observations", flush=True)
    print(f"{camera}: {count} observations from {path.name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", action="append", required=True, metavar="ID=VIDEO")
    parser.add_argument("--offset-ms", action="append", default=[], metavar="ID=MILLISECONDS")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--reid", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Optional camera configuration with excluded_image_rects")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--embedding-ms", type=int, default=350)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--window-ms", nargs=2, type=int, metavar=("START", "END"),
                        help="Decode only this window on the synchronized timeline")
    args = parser.parse_args()
    try:
        cameras = keyed(args.camera, Path)
        offsets = keyed(args.offset_ms, int)
        if offsets.keys() - cameras.keys():
            raise ValueError("Offsets refer to cameras that were not provided")
        if min(args.stride, args.embedding_ms, args.imgsz) <= 0:
            raise ValueError("stride, embedding-ms and imgsz must be positive")
        if args.max_seconds is not None and args.max_seconds <= 0:
            raise ValueError("max-seconds must be positive")
        if args.window_ms and (not 0 <= args.window_ms[0] < args.window_ms[1] or args.max_seconds is not None):
            raise ValueError("window-ms requires 0 <= START < END and cannot be used with max-seconds")
        for path in [*cameras.values(), args.weights, args.reid, args.reid.with_suffix(".bin")]:
            if not path.is_file():
                raise ValueError(f"File not found: {path}")
        settings = json.loads(args.config.read_text(encoding="utf-8-sig")) if args.config else {}
        excluded = settings.get("excluded_image_rects", {})
        for camera, regions in excluded.items():
            for left, top, right, bottom in regions:
                if not (0 <= left < right and 0 <= top < bottom):
                    raise ValueError(f"Invalid excluded rectangle for {camera}")
    except (OSError, ValueError) as error:
        parser.error(str(error))

    # Keep package settings inside this project and avoid online auto-installers.
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent / ".ultralytics"))
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    import av
    import cv2
    import numpy as np
    import openvino as ov
    from ultralytics import YOLO

    reid = ov.Core().compile_model(str(args.reid), "CPU")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        for camera, path in cameras.items():
            extract(camera, path, offsets.get(camera, 0), excluded.get(camera, []),
                    args, output, cv2, np, av, YOLO, reid)


if __name__ == "__main__":
    main()
