import asyncio
import json
from fractions import Fraction

import av
import numpy as np
import pytest


def test_session_available_after_export_and_media_is_allowlisted(tmp_path):
    pytest.importorskip("fastapi")
    httpx = pytest.importorskip("httpx")
    from app import create_app

    server = create_app(tmp_path)

    async def verify():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server), base_url="http://test"
        ) as client:
            assert (await client.get("/")).status_code == 200
            assert (await client.get("/api/session")).status_code == 503
            (tmp_path / "camera_1.mp4").write_bytes(b"0123456789")
            (tmp_path / "camera_1_overlay.json").write_text('{"frames":[]}')
            (tmp_path / "observations.jsonl").write_text("internal records")
            payload = {
                "schema_version": "1.0",
                "map": {"url": "/media/map.png"},
                "cameras": {
                    "camera_1": {"url": "/media/camera_1.mp4", "overlay_url": "/media/camera_1_overlay.json"}
                },
            }
            (tmp_path / "session.json").write_text(json.dumps(payload))
            assert (await client.get("/api/session")).json() == payload
            part = await client.get("/media/camera_1.mp4", headers={"Range": "bytes=2-5"})
            assert part.status_code == 206 and part.content == b"2345"
            assert (await client.get("/media/camera_1_overlay.json")).json() == {"frames": []}
            assert (await client.get("/media/observations.jsonl")).status_code == 404

    asyncio.run(verify())


def test_world_state_api_is_bounded_and_does_not_extrapolate(tmp_path):
    pytest.importorskip("fastapi")
    httpx = pytest.importorskip("httpx")
    from app import create_app

    (tmp_path / "session.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "start_time": 0.12,
                "end_time": 1.25,
                "map": {"url": "/media/map.png"},
                "cameras": {"camera_1": {"url": "/media/camera_1.mp4"}},
                "state_index": {"url": "/api/state", "step_ms": 120},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "world_state.json").write_text(
        json.dumps(
            {
                "step_ms": 120,
                "buckets": {"1": [{"id": 7, "position": {"x": 2, "y": 3}, "sources": []}]},
            }
        ),
        encoding="utf-8",
    )

    async def verify():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(tmp_path)), base_url="http://test"
        ) as client:
            response = await client.get("/api/state", params={"t": 0.12})
            assert response.json()["tracks"][0]["id"] == 7
            assert (await client.get("/api/state", params={"t": 0.24})).json()["tracks"] == []
            assert (await client.get("/api/state", params={"t": 1.26})).status_code == 422

    asyncio.run(verify())


def test_non_h264_web_video_keeps_source_frames_and_overlay_separate(tmp_path):
    from local_tracking.pipeline import file_hash
    from tools.export_web_session import package_camera

    source = tmp_path / "camera.mkv"
    timestamps = [400, 450, 501, 548, 604, 650]
    with av.open(str(source), "w") as video:
        stream = video.add_stream("ffv1", rate=20)
        stream.width, stream.height, stream.pix_fmt = 160, 120, "yuv420p"
        stream.codec_context.time_base = Fraction(1, 1000)
        for timestamp in timestamps:
            frame = av.VideoFrame.from_ndarray(np.zeros((120, 160, 3), dtype=np.uint8), format="bgr24")
            frame.pts, frame.time_base = timestamp, Fraction(1, 1000)
            video.mux(stream.encode(frame))
        video.mux(stream.encode())

    manifest = {
        "source": {"path": str(source), "first_frame_time_sec": 0.4, "width": 160, "height": 120, "fps": 20},
        "source_sha256": file_hash(source),
        "clock_anchors": [],
        "time_offset_sec": 4,
    }
    catalog = [(0, 0, 4), (3, 0.148, 4.148)]
    path = tmp_path / "camera_4.mp4"
    metadata = package_camera("camera_4", catalog, manifest, {3: [(1, [2, 3, 20, 30])]}, path)
    with av.open(str(path)) as video:
        assert video.streams.video[0].codec_context.name == "h264"
        assert [float(frame.pts * frame.time_base) for frame in video.decode(video=0)] == pytest.approx(
            [(time - timestamps[0]) / 1000 for time in timestamps]
        )
    assert metadata["coverage"] == [[4, 4.148]]
    overlay = json.loads((tmp_path / "camera_4_overlay.json").read_text(encoding="utf-8"))
    assert overlay["frames"] == [[0, []], [0.148, [[1, 2, 3, 20, 30]]]]


def test_uploaded_project_serves_its_own_session(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    httpx = pytest.importorskip("httpx")
    from app import create_app

    def process(plan, videos, output, roi, progress):
        assert plan.read_bytes() == b"plan"
        assert [video.read_bytes() for video in videos] == [b"first", b"second"]
        progress("building")
        session = output / "session"
        session.mkdir(parents=True)
        (session / "camera_1.mp4").write_bytes(b"0123456789")
        (session / "session.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "map": {"url": "/media/map.png"},
                    "cameras": {"camera_1": {"url": "/media/camera_1.mp4"}},
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr("project_uploads.process_project", process)
    server = create_app(tmp_path / "default", tmp_path / "uploads")

    async def verify():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server), base_url="http://test"
        ) as client:
            created = await client.post("/api/projects", json={"cameras": 2})
            assert created.status_code == 201
            job = "/api/projects/" + created.json()["project_id"]
            assert (await client.post(job + "/start")).status_code == 422
            for slot, name, data in (
                ("plan", "map.png", b"plan"),
                ("1", "a.mp4", b"first"),
                ("2", "b.mkv", b"second"),
            ):
                response = await client.put(job + "/files/" + slot, params={"filename": name}, content=data)
                assert response.status_code == 200
            assert (await client.post(job + "/start")).status_code == 202
            for _ in range(100):
                status = (await client.get(job)).json()
                if status["state"] in {"ready", "failed"}:
                    break
                await asyncio.sleep(0.02)
            assert status["state"] == "ready", status
            url = status["url"]
            session = await client.get(url + "api/session")
            assert session.status_code == 200
            assert (await client.get("/api/session")).status_code == 503
            video = session.json()["cameras"]["camera_1"]["url"]
            assert video == url + "media/camera_1.mp4"
            assert (await client.get(video, headers={"Range": "bytes=2-5"})).content == b"2345"
            assert (await client.get(url + "media/status.json")).status_code == 404
            assert (await client.post(job + "/start")).status_code == 409

    asyncio.run(verify())
