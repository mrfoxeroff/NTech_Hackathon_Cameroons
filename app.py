"""Player API for prepared and uploaded camera tracking sessions."""

import json
import math
import os
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

ROOT = Path(__file__).resolve().parent


def create_app(session_root, projects=None):
    session_root = Path(session_root).resolve()
    server = FastAPI(title="NTech camera tracking")
    routes = APIRouter()

    def root_for(request):
        if "project_id" not in request.path_params:
            return session_root
        from project_uploads import project_path, read_state

        path = project_path(projects, request.path_params["project_id"])
        if read_state(path)["state"] != "ready":
            raise HTTPException(409, "Сеанс ещё не готов.")
        return path / "result" / "session"

    @lru_cache(maxsize=8)
    def session(root):
        try:
            data = json.loads((root / "session.json").read_text(encoding="utf-8"))
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

    @routes.get("/", include_in_schema=False)
    def index(request: Request):
        if request.path_params:
            root_for(request)
        return FileResponse(ROOT / "index.html")

    @server.get("/viewer.js", include_in_schema=False)
    def viewer():
        return FileResponse(ROOT / "viewer.js", media_type="text/javascript")

    @routes.get("/api/session")
    def read_session(request: Request):
        data, _ = session(root_for(request))
        if "project_id" in request.path_params:
            prefix = "/projects/" + request.path_params["project_id"]
            data = data | {
                "map": data["map"] | {"url": prefix + data["map"]["url"]},
                "cameras": {
                    camera: info | {key: prefix + info[key] for key in ("url", "overlay_url") if key in info}
                    for camera, info in data["cameras"].items()
                },
            }
            if "state_index" in data:
                data["state_index"] = data["state_index"] | {"url": prefix + data["state_index"]["url"]}
        return JSONResponse(data, headers={"Cache-Control": "no-store"})

    @lru_cache(maxsize=4)
    def world_state(root):
        try:
            return json.loads((root / "world_state.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise HTTPException(503, "Индекс состояний сеанса недоступен.") from error

    @routes.get("/api/state")
    def read_state(t: float, request: Request):
        root = root_for(request)
        data, _ = session(root)
        if "state_index" not in data:
            raise HTTPException(404, "Этот сеанс не содержит индекс состояний.")
        if not math.isfinite(t) or t < data["start_time"] or t > data["end_time"]:
            raise HTTPException(422, "Время выходит за границы сеанса.")
        state = world_state(root)
        step = state["step_ms"]
        bucket = math.floor(t * 1000 + 1e-7) // step
        return {
            "start_sec": bucket * step / 1000,
            "end_sec": (bucket + 1) * step / 1000,
            "tracks": state["buckets"].get(str(bucket), []),
        }

    @routes.get("/media/{name}")
    def media(name: str, request: Request):
        root = root_for(request)
        _, names = session(root)
        path = (root / name).resolve()
        if name not in names or path.parent != root or not path.is_file():
            raise HTTPException(404, "Файл сеанса не найден.")
        return FileResponse(path)

    server.include_router(routes)
    if projects is not None:
        from project_uploads import attach_projects

        projects = Path(projects).resolve()
        attach_projects(server, projects, ROOT)
        server.include_router(routes, prefix="/projects/{project_id}")
    return server


default_session = ROOT / "outputs_all/session"
if not (default_session / "session.json").is_file():
    default_session = ROOT / "outputs_full/session"
app = create_app(os.environ.get("NTECH_SESSION_DIR", default_session), os.environ.get("NTECH_PROJECTS_DIR"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
