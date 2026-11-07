"""Read-only API for one prepared camera tracking session."""

import json
import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

ROOT = Path(__file__).resolve().parent


def create_app(session_root):
    session_root = Path(session_root).resolve()
    server = FastAPI(title="NTech camera tracking")

    @lru_cache(maxsize=1)
    def session():
        try:
            data = json.loads((session_root / "session.json").read_text(encoding="utf-8"))
            if data["schema_version"] != "1.0" or not data["cameras"]:
                raise ValueError("Unsupported session schema")
            urls = [data["map"]["url"]]
            for camera in data["cameras"].values():
                urls.append(camera["url"])
                if "overlay_url" in camera:
                    urls.append(camera["overlay_url"])
            if any(not url.startswith("/media/") for url in urls):
                raise ValueError("Expected local media URLs")
            names = {url.removeprefix("/media/") for url in urls}
            if any(not name or Path(name).name != name or name in {".", ".."} for name in names):
                raise ValueError("Invalid media name")
            return data, names
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise HTTPException(503, "Подготовьте данные командой python -m tools.run_web_demo.") from error

    @server.get("/", include_in_schema=False)
    def index():
        return FileResponse(ROOT / "index.html")

    @server.get("/viewer.js", include_in_schema=False)
    def viewer():
        return FileResponse(ROOT / "viewer.js", media_type="text/javascript")

    @server.get("/api/session")
    def read_session():
        data, _ = session()
        return JSONResponse(data, headers={"Cache-Control": "no-store"})

    @server.get("/media/{name}")
    def media(name: str):
        _, names = session()
        path = (session_root / name).resolve()
        if name not in names or path.parent != session_root or not path.is_file():
            raise HTTPException(404, "Файл сеанса не найден.")
        return FileResponse(path)

    return server


app = create_app(os.environ.get("NTECH_SESSION_DIR", ROOT / "outputs_full/session"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
