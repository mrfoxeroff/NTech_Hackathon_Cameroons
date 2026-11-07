from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


ROOT = Path(__file__).resolve().parent
app = FastAPI()
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
def read_index():
    return FileResponse(ROOT / "index.html")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
