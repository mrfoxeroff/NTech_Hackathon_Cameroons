"""Independent local UI and commands for video-to-plan homography."""

import argparse
import json
import mimetypes
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import RLock
from urllib.parse import unquote, urlsplit

import cv2

from features import prepare
from geometry import fit_calibration, project_observation, project_point

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def serve(directory, port):
    directory = Path(directory).resolve()
    session = json.loads((directory / "session.json").read_text(encoding="utf-8"))
    if session["status"] != "complete":
        raise ValueError("Preparation did not complete")
    latest = {}
    lock = RLock()
    media = {"plan.png"} | {c["url"].split("/")[-1] for c in session["cameras"].values()}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def send(self, code, payload):
            data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            route = unquote(urlsplit(self.path).path)
            if route == "/api/session":
                return self.send(200, session)
            if route == "/":
                path = ROOT / "index.html"
            elif route.startswith("/media/") and route.removeprefix("/media/") in media:
                path = directory / route.removeprefix("/media/")
            else:
                return self.send(404, {"error": "Not found"})
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 1024 * 1024:
                    return self.send(413, {"error": "Invalid request size"})
                data = json.loads(self.rfile.read(length))
                camera = data["camera"]
                if camera not in session["cameras"]:
                    raise ValueError("Unknown camera")
                with lock:
                    if self.path == "/api/fit":
                        if len(data["pairs"]) > 200:
                            raise ValueError("At most 200 landmarks")
                        result = fit_calibration(
                            data["pairs"],
                            session["cameras"][camera]["size"],
                            session["plan"]["size"],
                            threshold_px=float(data.get("threshold_px", 4)),
                            max_check_error_px=float(data.get("max_check_error_px", 6)),
                            intrinsics=data.get("intrinsics"),
                            meters_per_pixel=data.get("meters_per_pixel"),
                        )
                        result.update(
                            camera_id=camera,
                            source_sha256=session["cameras"][camera]["source_sha256"],
                            plan_sha256=session["plan"]["sha256"],
                            reference_image_sha256=session["cameras"][camera].get("reference_image_sha256"),
                            provenance="confirmed_landmark_pairs_from_local_ui",
                        )
                        if session["cameras"][camera]["background"].get("camera_motion_detected"):
                            result["status"] = "draft"
                            result["quality"]["warnings"].append(
                                "Обнаружено движение камеры: нужна отдельная калибровка каждого неподвижного участка"
                            )
                        latest[camera] = result
                        return self.send(200, result)
                    if camera not in latest:
                        raise ValueError("Fit the calibration first")
                    if self.path == "/api/project":
                        return self.send(
                            200,
                            project_point(
                                data["point"], latest[camera], allow_draft=True, allow_extrapolation=True
                            ),
                        )
                    if self.path == "/api/save":
                        target = directory / "exports"
                        target.mkdir(exist_ok=True)
                        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
                        path = target / f"{camera}-{stamp}.json"
                        write_json(path, latest[camera])
                        return self.send(200, {"file": str(path), "calibration": latest[camera]})
                    return self.send(404, {"error": "Not found"})
            except (ValueError, TypeError, KeyError, OSError, cv2.error) as error:
                self.send(400, {"error": str(error)})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def camera_inputs(values):
    result = {}
    for value in values:
        name, path = value.split("=", 1)
        if name in result:
            raise ValueError("Duplicate camera ID")
        result[name] = Path(path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    prep = subs.add_parser("prepare")
    prep.add_argument("--plan", type=Path, required=True)
    prep.add_argument("--video", action="append", default=[], metavar="CAMERA=PATH")
    prep.add_argument("--image", action="append", default=[], metavar="CAMERA=PATH")
    prep.add_argument("--output", type=Path, required=True)
    web = subs.add_parser("serve")
    web.add_argument("--session", type=Path, required=True)
    web.add_argument("--port", type=int, default=8765)
    fit = subs.add_parser("fit")
    fit.add_argument("--input", type=Path, required=True, help="JSON with pairs, camera_size, plan_size")
    fit.add_argument("--output", type=Path, required=True)
    project = subs.add_parser("project")
    project.add_argument("--calibration", type=Path, required=True)
    project.add_argument("--input", type=Path, required=True, help="JSONL observations with footpoint_px")
    project.add_argument("--output", type=Path, required=True)
    project.add_argument("--allow-draft", action="store_true")
    project.add_argument("--allow-extrapolation", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare(args.plan, camera_inputs(args.video), camera_inputs(args.image), args.output)
        elif args.command == "serve":
            serve(args.session, args.port)
        elif args.command == "fit":
            if args.output.exists():
                raise FileExistsError("Use a new output file")
            data = json.loads(args.input.read_text(encoding="utf-8"))
            result = fit_calibration(
                data["pairs"],
                data["camera_size"],
                data["plan_size"],
                intrinsics=data.get("intrinsics"),
                meters_per_pixel=data.get("meters_per_pixel"),
                threshold_px=data.get("threshold_px", 4),
                max_check_error_px=data.get("max_check_error_px", 6),
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, result)
            print(json.dumps({"status": result["status"], "quality": result["quality"]}, ensure_ascii=False))
        else:
            calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
            if args.output.exists():
                raise FileExistsError("Use a new output file")
            if args.input.resolve() == args.output.resolve():
                raise ValueError("Input and output must be different files")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.input.open(encoding="utf-8") as source, args.output.open("x", encoding="utf-8") as out:
                for line in source:
                    if line.strip():
                        row = project_observation(
                            json.loads(line),
                            calibration,
                            allow_draft=args.allow_draft,
                            allow_extrapolation=args.allow_extrapolation,
                        )
                        out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
