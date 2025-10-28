import asyncio
from typing import List, Tuple
import httpx
import pytest
from prometheus_client import CollectorRegistry

from sgorch.router.runtime import (
    RouterRuntime,
    RouterRuntimeConfig,
    WorkerStatus,
    create_router_app,
)
from sgorch.router.metrics import RouterMetrics


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


def test_router_metrics_update_without_server():
    asyncio.run(_run_router_metrics_update())


async def _run_router_metrics_update():
    registry = CollectorRegistry()
    metrics = RouterMetrics(router_name="unit-test", registry=registry)
    runtime = RouterRuntime(RouterRuntimeConfig(probe_interval_s=0.1, failure_cooldown_s=0.1), metrics=metrics)

    await runtime.add_worker("http://worker-a:8000")
    await runtime.record_success("http://worker-a:8000")
    await runtime.record_failure("http://worker-a:8000")
    await runtime.remove_worker("http://worker-a:8000")

    labels_unknown = {"router": "unit-test", "status": "unknown"}
    labels_healthy = {"router": "unit-test", "status": "healthy"}
    labels_unhealthy = {"router": "unit-test", "status": "unhealthy"}

    assert registry.get_sample_value("sgorch_router_workers", labels_unknown) == 0.0
    assert registry.get_sample_value("sgorch_router_workers", labels_healthy) == 0.0
    assert registry.get_sample_value("sgorch_router_workers", labels_unhealthy) == 0.0

    state_labels_unknown = {
        "router": "unit-test",
        "worker": "http://worker-a:8000",
        "status": "unknown",
    }
    state_labels_healthy = dict(state_labels_unknown)
    state_labels_healthy["status"] = "healthy"
    state_labels_unhealthy = dict(state_labels_unknown)
    state_labels_unhealthy["status"] = "unhealthy"

    assert registry.get_sample_value("sgorch_router_worker_state_changes_total", state_labels_unknown) == 1.0
    assert registry.get_sample_value("sgorch_router_worker_state_changes_total", state_labels_healthy) == 1.0
    assert registry.get_sample_value("sgorch_router_worker_state_changes_total", state_labels_unhealthy) == 1.0

    registration_labels_add = {"router": "unit-test", "action": "add"}
    registration_labels_remove = {"router": "unit-test", "action": "remove"}
    assert registry.get_sample_value("sgorch_router_workers_registered_total", registration_labels_add) == 1.0
    assert registry.get_sample_value("sgorch_router_workers_registered_total", registration_labels_remove) == 1.0


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
