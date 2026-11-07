"""Export one completed Dima run into the static files consumed by the UI."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from urllib.parse import quote


STATIC = Path(__file__).resolve().parent / "static"
COLORS = ("#38bdf8", "#f43f5e", "#a3e635", "#fbbf24", "#c084fc", "#2dd4bf")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_rows(path):
    with path.open(encoding="utf-8-sig") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False,
                               separators=(",", ":")) + "\n", encoding="utf-8")


def static_url(root, name):
    path = (root / name).resolve()
    if not path.is_file():
        raise ValueError(f"Файл не найден: {path}")
    return "/static/" + quote(path.relative_to(root).as_posix(), safe="/")


def video_urls(root, entries):
    urls = {}
    for entry in entries:
        camera, separator, path = entry.partition("=")
        if not separator or not camera or not path or camera in urls:
            raise ValueError(f"Ожидается уникальная пара камера=файл: {entry}")
        urls[camera] = static_url(root, path)
    return urls


def source_logs(run, cameras):
    local = [Path(info["input"]) / "frames.jsonl"
             for info in cameras.values() if info.get("input")]
    if len(local) == len(cameras) and all(path.is_file() for path in local):
        return local
    if (run / "frames.jsonl").is_file():
        return [run / "frames.jsonl"]
    raise ValueError("Укажи исходные журналы кадров через --frame-log")


def clock_config(cameras):
    timing = {}
    for camera, info in cameras.items():
        offset = info.get("time_offset_sec", 0)
        anchors = info.get("clock_anchors", [])
        if not math.isfinite(offset):
            raise ValueError(f"Некорректное смещение: {camera}")
        for axis in ("video_time_sec", "timestamp_sec"):
            values = [point[axis] for point in anchors]
            if (values and len(values) < 2 or not all(map(math.isfinite, values))
                    or any(a >= b for a, b in zip(values, values[1:]))):
                raise ValueError(f"Некорректные якоря времени: {camera}")
        timing[camera] = {"time_offset_sec": offset, "clock_anchors": anchors}
    return timing


def project(point, matrix):
    x, y = point
    a, b, c = (row[0] * x + row[1] * y + row[2] for row in matrix)
    if abs(c) < 1e-9:
        return None
    return a / c, b / c


def export(args):
    run = args.run
    root = args.static_dir.resolve()
    manifest = read_json(run / "manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError("Запуск Dima должен иметь status=complete")
    if manifest.get("stage") != "local_import" or not manifest.get("cameras"):
        raise ValueError("Нужен манифест импорта Dima с данными камер")
    cameras = manifest["cameras"]
    videos = video_urls(root, args.video)
    if videos.keys() != cameras.keys() or len(videos) < 2:
        raise ValueError("Укажи --video для каждой камеры запуска, минимум две")

    tracks = read_json(run / "tracks.json")
    if tracks["total_tracks"] != len({track["global_id"] for track in tracks["tracks"]}):
        raise ValueError("Счётчик треков не совпадает со списком ID")

    # The optional raw frame logs also retain the footpoint status from newer misha runs.
    frames = defaultdict(list)
    frame_sizes = {}
    frame_times = {}
    raw_points = {}
    for path in args.frame_log or source_logs(run, cameras):
        for row in read_rows(path):
            camera, index = row["camera_id"], row["frame_index"]
            if camera not in cameras or row["recording_id"] != manifest["recording_id"]:
                raise ValueError(f"Журнал из другой записи или камеры: {path}")
            if (camera, index) in frame_times:
                raise ValueError(f"Повтор кадра: {camera}/{index}")
            frame_times[camera, index] = round(row["timestamp_sec"] * 1000)
            size = row["frame_width"], row["frame_height"]
            if frame_sizes.setdefault(camera, size) != size or min(size) <= 0:
                raise ValueError(f"Некорректный размер кадра: {camera}/{index}")
            frames[camera].append({key: row[key] for key in
                                   ("frame_index", "video_time_sec", "timestamp_sec")})
            if args.calibration_dir:
                for obs in row.get("observations", []):
                    raw_points[camera, index, obs["observation_id"]] = obs
    if frames.keys() != cameras.keys():
        raise ValueError("Журналы должны содержать обработанные кадры всех камер")

    start, end = args.window_sec or (
        max(min(row["timestamp_sec"] for row in rows) for rows in frames.values()),
        min(max(row["timestamp_sec"] for row in rows) for rows in frames.values()),
    )
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError("Некорректное окно. Укажи --window-sec НАЧАЛО КОНЕЦ")

    if bool(args.calibration_dir) != bool(args.plan):
        raise ValueError("Для карты задай --calibration-dir и --plan вместе")
    calibration = {}
    map_info = None
    if args.calibration_dir:
        for camera in cameras:
            data = read_json(args.calibration_dir / f"{camera}_calibration.json")
            if tuple(data["camera_size"]) != frame_sizes[camera] or data["lens_correction_applied"]:
                raise ValueError(f"Калибровка не соответствует координатам {camera}")
            width, height = data["plan_size"]
            info = {"url": static_url(root, args.plan), "width_px": width, "height_px": height}
            if map_info and info != map_info:
                raise ValueError("Камеры должны быть откалиброваны на одном плане")
            map_info = info
            calibration[camera] = data["homography"]

    assignments = {}
    for row in read_rows(run / "assignments.jsonl"):
        key = row["camera_id"], row["local_id"], row["frame_index"]
        if key in assignments:
            raise ValueError(f"Повтор назначения: {key}")
        assignments[key] = row["global_id"]

    track_ids = {track["global_id"] for track in tracks["tracks"]}
    overlays = defaultdict(list)
    map_points = defaultdict(list)
    window_track_ids = set()
    for obs in read_rows(run / "observations.jsonl"):
        camera, index = obs["camera_id"], obs["frame_index"]
        key = camera, obs["local_id"], index
        if key not in assignments or frame_times.get((camera, index)) != obs["timestamp_ms"]:
            raise ValueError(f"Наблюдение без назначения или кадра: {key}")
        track_id = assignments.pop(key)
        if track_id is None:
            continue
        if track_id not in track_ids:
            raise ValueError(f"Сквозной ID отсутствует в tracks.json: {track_id}")
        overlays[camera, index].append({
            "global_id": track_id, "silhouette_polygons": obs["silhouette_polygons"],
        })
        if start <= obs["timestamp_ms"] / 1000 <= end:
            window_track_ids.add(track_id)
        if map_info and start <= obs["timestamp_sec"] <= end:
            geometry = raw_points.get((camera, index, obs["observation_id"]), obs)
            point = geometry.get("footpoint_px")
            status = geometry.get("footpoint_status", "estimated")
            if point is None or status not in ("observed", "estimated"):
                continue
            xy = project(point, calibration[camera])
            if xy and 0 <= xy[0] < map_info["width_px"] and 0 <= xy[1] < map_info["height_px"]:
                map_points[track_id].append({
                    "t": obs["timestamp_sec"] - start, "x": xy[0], "y": xy[1],
                    "status": status, "cams": [camera],
                })
    if assignments:
        raise ValueError("В assignments.jsonl остались назначения без наблюдений")

    summaries = {}
    for track in tracks["tracks"]:
        track_id = track["global_id"]
        if track_id not in window_track_ids:
            continue
        segments = sorted(
            ({"cam": camera, "start": max(first / 1000, start) - start,
              "end": min(last / 1000, end) - start}
             for camera, intervals in track["cameras"].items()
             for first, last in intervals if first / 1000 <= end and last / 1000 >= start),
            key=lambda item: (item["start"], item["cam"]),
        )
        if not segments:
            continue
        # Show one projected sample per 100 ms, preferring confirmed geometry.
        buckets = {}
        for point in sorted(map_points[track_id], key=lambda point: point["t"]):
            bucket = int(point["t"] * 10)
            if bucket not in buckets or buckets[bucket]["status"] != "observed":
                buckets[bucket] = point
        summaries[str(track_id)] = {
            "id": track_id, "start_time": max(track["start_ms"] / 1000, start) - start,
            "end_time": min(track["end_ms"] / 1000, end) - start,
            "camera_segments": segments,
            "cameras_visited": list(dict.fromkeys(seg["cam"] for seg in segments)),
            "color": COLORS[len(summaries) % len(COLORS)], "map_path": list(buckets.values()),
        }

    timing = clock_config(cameras)
    urls = {}
    for camera, rows in frames.items():
        width, height = frame_sizes[camera]
        for row in rows:
            row["observations"] = overlays.get((camera, row["frame_index"]), [])
        path = root / "overlays" / f"{quote(camera, safe='')}.json"
        write_json(path, {
            "schema_version": 1, "camera_id": camera, "frame_width": width,
            "frame_height": height,
            "frames": sorted(rows, key=lambda row: row["video_time_sec"]),
        })
        urls[camera] = static_url(root, path.relative_to(root))
    write_json(root / "clock_alignment.json", {"cameras": timing})
    write_json(root / "tracks_metadata.json", {
        "schema_version": 1, "recording_id": manifest["recording_id"],
        "timeline_start_sec": start, "duration_sec": end - start,
        "total_unique_tracks": len(summaries), "cameras": videos,
        "clock_alignment_url": "/static/clock_alignment.json",
        "overlay_urls": urls, "tracks_summary": summaries, "map": map_info,
    })
    print(f"Камер: {len(cameras)}. Треков в окне: {len(summaries)}. Данные: {root}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Каталог завершённого запуска Dima")
    parser.add_argument("--static-dir", type=Path, default=STATIC)
    parser.add_argument("--video", action="append", required=True, metavar="CAM=FILE")
    parser.add_argument("--frame-log", action="append", type=Path, metavar="JSONL")
    parser.add_argument("--window-sec", type=float, nargs=2, metavar=("START", "END"))
    parser.add_argument("--calibration-dir", type=Path)
    parser.add_argument("--plan", help="Путь к картинке плана относительно static")
    args = parser.parse_args()
    try:
        export(args)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Ошибка экспорта: {error}\n")


if __name__ == "__main__":
    main()
