"""Read-only dashboard for automatic neural runs; no manual landmark editor."""

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent


def serve(output, port=8766):
    output = Path(output).resolve()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = unquote(urlsplit(self.path).path)
            if route == "/":
                path = ROOT / "neural.html"
            elif route == "/api/result":
                path = output / "manifest.json"
            elif route.startswith("/media/"):
                name = route.removeprefix("/media/")
                if Path(name).name != name or Path(name).suffix not in (".jpg", ".png", ".json"):
                    self.send_error(404)
                    return
                path = output / name
            else:
                self.send_error(404)
                return
            if not path.is_file():
                self.send_error(404)
                return
            if route == "/api/result":
                payload = json.loads(path.read_text(encoding="utf-8"))
                audit = output / "evaluation.json"
                if audit.is_file():
                    evaluations = json.loads(audit.read_text(encoding="utf-8"))
                    for camera, review in evaluations.get("cameras", {}).items():
                        if camera in payload["cameras"]:
                            payload["cameras"][camera]["review"] = review
                data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            else:
                data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    serve(args.output, args.port)
