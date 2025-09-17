import asyncio
from typing import List, Tuple
import httpx
import pytest

from sgorch.router.runtime import (
    RouterRuntime,
    RouterRuntimeConfig,
    WorkerStatus,
    create_router_app,
)


def test_router_add_list_remove(tmp_path):

    asyncio.run(_run_router_add_list_remove(tmp_path))


async def _run_router_add_list_remove(tmp_path):
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(200)
        return httpx.Response(200, json={"echo": request.url.host})

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as proxy_client, httpx.AsyncClient(transport=transport) as health_client:
        runtime = RouterRuntime(
            RouterRuntimeConfig(probe_interval_s=0.1, failure_cooldown_s=0.1),
            proxy_client=proxy_client,
            health_client=health_client,
        )
        await runtime.start()
        app = create_router_app(runtime)

        router_transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=router_transport, base_url="http://router") as client:
            resp = await client.get("/workers/list")
            assert resp.json() == {"workers": []}

            resp = await client.post("/workers/add", params={"url": "http://worker-a:8000"})
            assert resp.status_code == 200

            resp = await client.get("/workers/list")
            assert resp.json()["workers"] == ["http://worker-a:8000"]

            resp = await client.post("/workers/remove", params={"url": "http://worker-a:8000"})
            assert resp.status_code == 200

            resp = await client.get("/workers/list")
            assert resp.json()["workers"] == []

        await runtime.stop()


@pytest.mark.usefixtures("seed_random")
def test_router_retries_and_marks_unhealthy():
    asyncio.run(_run_router_retries())


async def _run_router_retries():
    request_log: List[Tuple[str, str]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        request_log.append((request.url.host, request.url.path))
        if request.url.path.endswith("/health"):
            return httpx.Response(200)
        if request.url.host == "worker-a":
            return httpx.Response(503, json={"error": "boom"})
        return httpx.Response(200, json={"worker": request.url.host})

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as proxy_client, httpx.AsyncClient(transport=transport) as health_client:
        runtime = RouterRuntime(
            RouterRuntimeConfig(probe_interval_s=1000.0, failure_cooldown_s=1000.0, max_retries=3),
            proxy_client=proxy_client,
            health_client=health_client,
        )
        await runtime.start()
        app = create_router_app(runtime)

        router_transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=router_transport, base_url="http://router") as client:
            for url in ("http://worker-a:8000", "http://worker-b:8000"):
                resp = await client.post("/workers/add", params={"url": url})
                assert resp.status_code == 200

            resp = await client.post("/v1/embeddings", json={"input": "hello"})
            assert resp.status_code == 200
            assert resp.json()["worker"] == "worker-b"

        async with runtime._lock:  # direct inspection for test assertions
            entry_a = runtime._workers["http://worker-a:8000"]
            entry_b = runtime._workers["http://worker-b:8000"]
            assert entry_a.fail_count >= 1
            assert entry_b.status == WorkerStatus.HEALTHY

        upstream_attempts = [entry for entry in request_log if not entry[1].endswith("/health")]
        assert ("worker-a", "/v1/embeddings") in upstream_attempts
        assert ("worker-b", "/v1/embeddings") in upstream_attempts

        await runtime.stop()
