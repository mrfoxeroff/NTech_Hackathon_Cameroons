"""Local uploads and one-at-a-time processing for project sessions."""

import json
import logging
import re
from pathlib import Path
from threading import Lock, Thread
from uuid import uuid4

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

DEFAULT_ROI = (0, 0, 0.42, 0.07)
SUFFIXES = {"plan": {".jpg", ".jpeg", ".png"}, "video": {".mkv", ".mp4", ".mov", ".avi", ".webm"}}
STATE_LOCK = Lock()


class NewProject(BaseModel):
    cameras: int = Field(ge=1, le=32)
    clock_roi: tuple[float, float, float, float] = DEFAULT_ROI


def project_path(directory, project_id):
    if not re.fullmatch(r"[a-f0-9]{32}", project_id):
        raise HTTPException(404, "Проект не найден.")
    path = directory / project_id
    if not path.is_dir():
        raise HTTPException(404, "Проект не найден.")
    return path


def read_state(path):
    with STATE_LOCK:
        return json.loads((path / "status.json").read_text(encoding="utf-8"))


def write_state(path, **status):
    with STATE_LOCK:
        temp = path / "status.tmp"
        temp.write_text(json.dumps(status, ensure_ascii=False), encoding="utf-8")
        temp.replace(path / "status.json")


def process_project(plan, videos, output, roi, progress):
    from tools.run_uploaded_project import run_project

    return run_project(plan, videos, output, roi=roi, progress=progress)


def attach_projects(server, directory, assets):
    active = Lock()
    running = set()

    @server.get("/api/projects/enabled")
    def enabled():
        return {"enabled": True}

    @server.get("/new", include_in_schema=False)
    def new_page():
        return FileResponse(assets / "upload.html")

    @server.get("/upload.js", include_in_schema=False)
    def upload_js():
        return FileResponse(assets / "upload.js", media_type="text/javascript")

    @server.post("/api/projects", status_code=201)
    def create_project(body: NewProject):
        x1, y1, x2, y2 = body.clock_roi
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1 and x2 - x1 >= 0.05 and y2 - y1 >= 0.01):
            raise HTTPException(422, "Некорректная область часов.")
        path = directory / uuid4().hex
        path.mkdir(parents=True)
        (path / "inputs").mkdir()
        (path / "request.json").write_text(body.model_dump_json(), encoding="utf-8")
        write_state(path, state="uploading")
        return {"project_id": path.name}

    @server.get("/api/projects/{project_id}")
    def project_status(project_id: str):
        path = project_path(directory, project_id)
        status = read_state(path)
        if status["state"] == "processing" and project_id not in running:
            return {"state": "failed", "message": "Обработка прервана. Создайте новый проект."}
        return status

    @server.put("/api/projects/{project_id}/files/{slot}")
    async def upload_file(project_id: str, slot: str, filename: str, request: Request):
        path = project_path(directory, project_id)
        body = NewProject.model_validate_json((path / "request.json").read_text(encoding="utf-8"))
        if read_state(path)["state"] != "uploading":
            raise HTTPException(409, "Приём файлов завершён.")
        names = {str(i): f"video_{i}" for i in range(1, body.cameras + 1)} | {"plan": "plan"}
        if slot not in names:
            raise HTTPException(404, "Неизвестное место для файла.")
        kind = "plan" if slot == "plan" else "video"
        suffix = Path(filename).suffix.lower()
        if suffix not in SUFFIXES[kind]:
            raise HTTPException(415, "Этот формат файла не поддерживается.")
        name = names[slot]
        inputs = path / "inputs"
        partial = inputs / f"{name}.part"
        limit = 25 * 1024**2 if kind == "plan" else 4 * 1024**3
        size = 0
        try:
            stream = partial.open("xb")
        except FileExistsError as error:
            raise HTTPException(409, "Этот файл уже загружается.") from error
        try:
            with stream:
                if any(file != partial for file in inputs.glob(f"{name}.*")):
                    raise HTTPException(409, "Файл уже загружен.")
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(413, "Превышен размер файла.")
                    stream.write(chunk)
            if size == 0:
                raise HTTPException(422, "Файл пустой.")
            partial.replace(inputs / f"{name}{suffix}")
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        return {"bytes": size}

    @server.post("/api/projects/{project_id}/start", status_code=202)
    def start_project(project_id: str):
        path = project_path(directory, project_id)
        if read_state(path)["state"] != "uploading":
            raise HTTPException(409, "Обработка уже запущена.")
        body = NewProject.model_validate_json((path / "request.json").read_text(encoding="utf-8"))
        inputs = path / "inputs"
        files = {file.stem: file for file in inputs.iterdir()}
        expected = {"plan", *(f"video_{i}" for i in range(1, body.cameras + 1))}
        if files.keys() != expected or any(file.suffix == ".part" for file in inputs.iterdir()):
            raise HTTPException(422, "Сначала загрузите план и все видео.")
        if not active.acquire(blocking=False):
            raise HTTPException(409, "Другой проект ещё обрабатывается. Повторите запуск позже.")
        if read_state(path)["state"] != "uploading":
            active.release()
            raise HTTPException(409, "Обработка уже запущена.")

        def run():
            try:
                process_project(
                    files["plan"],
                    [files[f"video_{i}"] for i in range(1, body.cameras + 1)],
                    path / "result",
                    body.clock_roi,
                    lambda message: write_state(path, state="processing", message=message),
                )
                write_state(path, state="ready", url=f"/projects/{project_id}/")
            except Exception as error:
                logging.exception("Processing failed for project %s", project_id)
                write_state(path, state="failed", message=str(error) or "Ошибка обработки.")
            finally:
                running.discard(project_id)
                active.release()

        try:
            running.add(project_id)
            write_state(path, state="processing", message="Подготовка")
            Thread(target=run, name="ntech-project", daemon=True).start()
        except BaseException:
            running.discard(project_id)
            active.release()
            raise
        return {"state": "processing"}
