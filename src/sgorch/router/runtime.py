from __future__ import annotations

import asyncio
import random
import time
from time import perf_counter
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Optional, Set
from urllib.parse import urljoin

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response

from ..logging_setup import get_logger
from .metrics import RouterMetrics


logger = get_logger(__name__)


class WorkerStatus(str, Enum):
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


@dataclass
class WorkerEntry:
    url: str
    status: WorkerStatus = WorkerStatus.UNKNOWN
    fail_count: int = 0
    success_count: int = 0
    last_checked: float = 0.0
    last_failure: float = 0.0
    next_check: float = 0.0


@dataclass
class RouterRuntimeConfig:
    bind_host: str = "0.0.0.0"
    bind_port: int = 8080
    health_path: str = "/health"
    probe_interval_s: float = 10.0
    probe_timeout_s: float = 5.0
    request_timeout_s: float = 30.0
    max_retries: int = 3
    failure_cooldown_s: float = 5.0
    prometheus_port: Optional[int] = None
    prometheus_host: Optional[str] = None
    router_name: Optional[str] = None

    def normalized_health_path(self) -> str:
        path = self.health_path or "/health"
        if not path.startswith("/"):
            path = "/" + path
        return path


class ProxyError(Exception):
    pass


class RouterRuntime:
    def __init__(
        self,
        config: RouterRuntimeConfig,
        *,
        proxy_client: Optional[httpx.AsyncClient] = None,
        health_client: Optional[httpx.AsyncClient] = None,
        metrics: Optional[RouterMetrics] = None,
    ) -> None:
        self.config = config
        self._workers: Dict[str, WorkerEntry] = {}
        self._lock = asyncio.Lock()
        self._running = False
        self._health_task: Optional[asyncio.Task] = None
        self._proxy_client = proxy_client
        self._health_client = health_client
        self._owns_proxy_client = proxy_client is None
        self._owns_health_client = health_client is None
        if metrics is not None:
            self._metrics = metrics
        elif config.prometheus_port is not None:
            self._metrics = RouterMetrics(router_name=config.router_name)
        else:
            self._metrics = None

    async def start(self) -> None:
        if self._running:
            return
        logger.info("Starting router runtime")
        if self._proxy_client is None:
            self._proxy_client = httpx.AsyncClient(timeout=self.config.request_timeout_s)
        if self._health_client is None:
            self._health_client = httpx.AsyncClient(timeout=self.config.probe_timeout_s)
        if self._metrics and self.config.prometheus_port is not None:
            host = self.config.prometheus_host or self.config.bind_host
            self._metrics.start_http_server(host, self.config.prometheus_port)
        self._running = True
        self._health_task = asyncio.create_task(self._health_loop(), name="router-health-loop")
        if self._metrics:
            async with self._lock:
                self._update_worker_metrics_locked()

    async def stop(self) -> None:
        if not self._running:
            return
        logger.info("Stopping router runtime")
        self._running = False
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
        if self._owns_proxy_client and self._proxy_client:
            await self._proxy_client.aclose()
            self._proxy_client = None
        if self._owns_health_client and self._health_client:
            await self._health_client.aclose()
            self._health_client = None

    async def list_workers(self) -> Set[str]:
        async with self._lock:
            return set(self._workers.keys())

    async def add_worker(self, url: str) -> None:
        normalized = url.strip()
        if not normalized:
            raise ValueError("Worker URL must be provided")
        if normalized.endswith("/"):
            normalized = normalized.rstrip("/")
        async with self._lock:
            entry = self._workers.get(normalized)
            if entry:
                previous_status = entry.status
                entry.status = WorkerStatus.UNKNOWN
                entry.next_check = 0.0
                entry.fail_count = 0
                logger.info(f"Worker already registered, resetting state: {normalized}")
                if self._metrics and previous_status != WorkerStatus.UNKNOWN:
                    self._metrics.record_worker_state(normalized, WorkerStatus.UNKNOWN.value)
            else:
                self._workers[normalized] = WorkerEntry(url=normalized)
                logger.info(f"Registered new worker: {normalized}")
                if self._metrics:
                    self._metrics.record_worker_state(normalized, WorkerStatus.UNKNOWN.value)
            if self._metrics:
                self._metrics.record_worker_registration("add")
            self._update_worker_metrics_locked()

    async def remove_worker(self, url: str) -> None:
        normalized = url.strip().rstrip("/")
        async with self._lock:
            if normalized in self._workers:
                self._workers.pop(normalized, None)
                logger.info(f"Removed worker: {normalized}")
                if self._metrics:
                    self._metrics.record_worker_registration("remove")
                self._update_worker_metrics_locked()

    async def choose_worker(self, *, exclude: Iterable[str] = ()) -> Optional[WorkerEntry]:
        excluded = {u.rstrip("/") for u in exclude}
        async with self._lock:
            candidates = [w for w in self._workers.values() if w.url not in excluded]
        if not candidates:
            return None

        healthy = [w for w in candidates if w.status == WorkerStatus.HEALTHY]
        if healthy:
            return random.choice(healthy)

        # Fall back to unknown workers when no healthy targets exist
        unknown = [w for w in candidates if w.status == WorkerStatus.UNKNOWN]
        if unknown:
            return random.choice(unknown)

        return None

    async def record_success(self, url: str) -> None:
        normalized = url.rstrip("/")
        now = time.monotonic()
        async with self._lock:
            entry = self._workers.get(normalized)
            if not entry:
                return
            previous_status = entry.status
            entry.status = WorkerStatus.HEALTHY
            entry.fail_count = 0
            entry.success_count += 1
            entry.last_checked = now
            entry.next_check = now + self.config.probe_interval_s
            if self._metrics and previous_status != WorkerStatus.HEALTHY:
                self._metrics.record_worker_state(normalized, WorkerStatus.HEALTHY.value)
            self._update_worker_metrics_locked()

    async def record_failure(self, url: str, *, reason: str | None = None) -> None:
        normalized = url.rstrip("/")
        now = time.monotonic()
        async with self._lock:
            entry = self._workers.get(normalized)
            if not entry:
                return
            previous_status = entry.status
            entry.status = WorkerStatus.UNHEALTHY
            entry.fail_count += 1
            entry.last_checked = now
            entry.last_failure = now
            entry.next_check = now + self.config.failure_cooldown_s
            if self._metrics and previous_status != WorkerStatus.UNHEALTHY:
                self._metrics.record_worker_state(normalized, WorkerStatus.UNHEALTHY.value)
            self._update_worker_metrics_locked()
        if reason:
            logger.warning(f"Worker marked unhealthy {normalized}: {reason}")
        else:
            logger.warning(f"Worker marked unhealthy {normalized}")

    def _update_worker_metrics_locked(self) -> None:
        if not self._metrics:
            return
        counts: Dict[str, int] = {status.value: 0 for status in WorkerStatus}
        for entry in self._workers.values():
            counts[entry.status.value] = counts.get(entry.status.value, 0) + 1
        self._metrics.update_worker_counts(counts)

    async def forward_request(
        self,
        worker_url: str,
        request: Request,
        *,
        path: str,
        body: bytes,
        client_host: Optional[str],
    ) -> Response:
        if self._proxy_client is None:
            raise ProxyError("Proxy client is not initialized")

        target = self._build_target_url(worker_url, path, request.query_params)
        headers = self._prepare_outbound_headers(request.headers, client_host)
        method = request.method.upper()

        metrics = self._metrics
        start = perf_counter() if metrics else None
        if metrics:
            metrics.proxy_inflight_inc(method)

        try:
            resp = await self._proxy_client.request(
                method,
                target,
                content=body if body else None,
                headers=headers,
                timeout=self.config.request_timeout_s,
            )
        except httpx.RequestError as exc:
            if metrics and start is not None:
                metrics.record_proxy_attempt(method, "error", "exception", perf_counter() - start, worker_url)
            raise ProxyError(str(exc)) from exc
        else:
            if metrics and start is not None:
                status_code = str(resp.status_code)
                outcome = "success" if resp.status_code < 500 else "error"
                metrics.record_proxy_attempt(method, outcome, status_code, perf_counter() - start, worker_url)
        finally:
            if metrics:
                metrics.proxy_inflight_dec(method)

        return self._build_response(resp)

    async def _health_loop(self) -> None:
        while self._running:
            await self._run_health_checks()
            await asyncio.sleep(self.config.probe_interval_s)

    async def _run_health_checks(self) -> None:
        if self._health_client is None:
            return
        now = time.monotonic()
        async with self._lock:
            entries = list(self._workers.values())

        for entry in entries:
            if now < entry.next_check:
                continue
            await self._probe_worker(entry)

    async def _probe_worker(self, entry: WorkerEntry) -> None:
        assert self._health_client is not None
        target = urljoin(entry.url + "/", self.config.normalized_health_path().lstrip("/"))
        metrics = self._metrics
        start = perf_counter() if metrics else None
        try:
            resp = await self._health_client.get(target, timeout=self.config.probe_timeout_s)
            healthy = 200 <= resp.status_code < 500
        except httpx.RequestError as exc:
            healthy = False
            logger.debug(f"Health probe failed for {entry.url}: {exc}")

        if metrics and start is not None:
            duration = perf_counter() - start
            result = "success" if healthy else "failure"
            metrics.record_health_probe(
                entry.url,
                result,
                duration,
                success_timestamp=time.time() if healthy else None,
            )

        if healthy:
            await self.record_success(entry.url)
        else:
            await self.record_failure(entry.url)

    def _build_target_url(self, base: str, path: str, query_params) -> str:
        if path:
            if not path.startswith("/"):
                path = "/" + path
        else:
            path = "/"
        target = base.rstrip("/") + path
        query = str(query_params)
        if query:
            target = target + "?" + query
        return target

    def _prepare_outbound_headers(self, headers, client_host: Optional[str]) -> Dict[str, str]:
        hop_by_hop = {
            "connection",
            "keep-alive",
            "proxy-authenticate",
            "proxy-authorization",
            "te",
            "trailers",
            "transfer-encoding",
            "upgrade",
            "host",
        }
        outbound = {}
        for key, value in headers.items():
            if key.lower() in hop_by_hop:
                continue
            outbound[key] = value
        if client_host:
            existing = outbound.get("X-Forwarded-For")
            outbound["X-Forwarded-For"] = f"{existing}, {client_host}" if existing else client_host
        return outbound

    def _build_response(self, upstream: httpx.Response) -> Response:
        # FastAPI/Starlette expect str headers; httpx provides a case-insensitive dict
        headers = {
            key: value
            for key, value in upstream.headers.items()
            if key.lower() not in {
                "connection",
                "keep-alive",
                "proxy-authenticate",
                "proxy-authorization",
                "te",
                "trailers",
                "transfer-encoding",
                "upgrade",
            }
        }
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers=headers,
            media_type=upstream.headers.get("content-type"),
        )


def create_router_app(runtime: RouterRuntime) -> FastAPI:
    app = FastAPI()
    app.state.runtime = runtime

    @app.on_event("startup")
    async def _startup() -> None:  # pragma: no cover - handled via tests
        await runtime.start()

    @app.on_event("shutdown")
    async def _shutdown() -> None:  # pragma: no cover - handled via tests
        await runtime.stop()

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        workers = await runtime.list_workers()
        return JSONResponse({"status": "ok", "workers": sorted(workers)})

    @app.get("/workers/list")
    async def list_workers() -> JSONResponse:
        workers = await runtime.list_workers()
        return JSONResponse({"workers": sorted(workers)})

    @app.post("/workers/add")
    async def add_worker(url: str = Query(..., description="Worker URL")) -> JSONResponse:
        try:
            await runtime.add_worker(url)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return JSONResponse({"status": "ok"})

    @app.post("/workers/remove")
    async def remove_worker(url: str = Query(..., description="Worker URL")) -> JSONResponse:
        await runtime.remove_worker(url)
        return JSONResponse({"status": "ok"})

    @app.api_route("/{full_path:path}", methods=[
        "GET",
        "POST",
        "PUT",
        "PATCH",
        "DELETE",
        "OPTIONS",
        "HEAD",
    ], include_in_schema=False)
    async def proxy_request(full_path: str, request: Request):
        body = await request.body()
        attempted: Set[str] = set()
        last_error: Optional[Exception] = None
        last_response: Optional[Response] = None
        client_host = request.client.host if request.client else None
        metrics = runtime._metrics
        attempt = 0

        for attempt in range(max(1, runtime.config.max_retries)):
            worker = await runtime.choose_worker(exclude=attempted)
            if not worker:
                break
            attempted.add(worker.url)
            try:
                response = await runtime.forward_request(
                    worker.url,
                    request,
                    path=full_path,
                    body=body,
                    client_host=client_host,
                )
                if response.status_code >= 500:
                    last_response = response
                    await runtime.record_failure(
                        worker.url,
                        reason=f"upstream status {response.status_code}",
                    )
                    if metrics and attempt + 1 < runtime.config.max_retries:
                        metrics.record_retry(f"upstream_status_{response.status_code}")
                    continue
                await runtime.record_success(worker.url)
                return response
            except ProxyError as exc:
                last_error = exc
                await runtime.record_failure(worker.url, reason=str(exc))
                if metrics and attempt + 1 < runtime.config.max_retries:
                    metrics.record_retry("proxy_error")
                continue

        detail = "No healthy workers available"
        if last_error:
            detail = f"Proxy attempts failed: {last_error}"
            if metrics:
                metrics.record_terminal_failure("proxy_error")
        elif last_response is not None:
            return last_response
        else:
            if metrics:
                metrics.record_terminal_failure("no_healthy_workers")
        raise HTTPException(status_code=503, detail=detail)

    return app
