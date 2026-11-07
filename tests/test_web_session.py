import asyncio
import json

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
            (tmp_path / "observations.jsonl").write_text("internal records")
            payload = {
                "schema_version": "1.0",
                "map": {"url": "/media/map.png"},
                "cameras": {"camera_1": {"url": "/media/camera_1.mp4"}},
            }
            (tmp_path / "session.json").write_text(json.dumps(payload))
            assert (await client.get("/api/session")).json() == payload
            part = await client.get("/media/camera_1.mp4", headers={"Range": "bytes=2-5"})
            assert part.status_code == 206 and part.content == b"2345"
            assert (await client.get("/media/observations.jsonl")).status_code == 404

    asyncio.run(verify())
