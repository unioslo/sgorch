from __future__ import annotations

import threading
from typing import Dict, Optional

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, start_http_server

from ..logging_setup import get_logger


logger = get_logger(__name__)


class RouterMetrics:
    """Prometheus metrics instrumentation for the standalone router."""

    def __init__(self, *, router_name: Optional[str] = None, registry: Optional[CollectorRegistry] = None) -> None:
        self.registry = registry or CollectorRegistry()
        self.router_name = router_name or "default"

        # Worker lifecycle
        self._workers = Gauge(
            "sgorch_router_workers",
            "Number of workers tracked by the router, by status",
            ["router", "status"],
            registry=self.registry,
        )
        self._worker_state_changes = Counter(
            "sgorch_router_worker_state_changes_total",
            "Worker status transition events",
            ["router", "worker", "status"],
            registry=self.registry,
        )
        self._worker_registrations = Counter(
            "sgorch_router_workers_registered_total",
            "Worker registration activity",
            ["router", "action"],
            registry=self.registry,
        )

        # Proxy traffic
        self._proxy_requests = Counter(
            "sgorch_router_proxy_requests_total",
            "Total proxy attempts to upstream workers",
            ["router", "method", "outcome", "status_code"],
            registry=self.registry,
        )
        self._proxy_latency = Histogram(
            "sgorch_router_proxy_request_latency_seconds",
            "Latency of proxy attempts to upstream workers",
            ["router", "method", "outcome"],
            registry=self.registry,
        )
        self._proxy_inflight = Gauge(
            "sgorch_router_proxy_inflight_requests",
            "Number of inflight proxy attempts",
            ["router", "method"],
            registry=self.registry,
        )
        self._proxy_retries = Counter(
            "sgorch_router_retries_total",
            "Retries performed while proxying requests",
            ["router", "reason"],
            registry=self.registry,
        )
        self._proxy_failures = Counter(
            "sgorch_router_failed_attempts_total",
            "Terminal proxy failures after retries were exhausted",
            ["router", "reason"],
            registry=self.registry,
        )

        # Health probes
        self._health_probes = Counter(
            "sgorch_router_health_probes_total",
            "Worker health probe results",
            ["router", "worker", "result"],
            registry=self.registry,
        )
        self._health_latency = Histogram(
            "sgorch_router_health_probe_duration_seconds",
            "Latency of worker health probes",
            ["router", "worker"],
            registry=self.registry,
        )
        self._health_last_success = Gauge(
            "sgorch_router_health_last_success_timestamp",
            "Unix timestamp of the last successful health probe",
            ["router", "worker"],
            registry=self.registry,
        )

        self._http_server_port: Optional[int] = None
        self._http_server_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Worker metrics helpers
    def update_worker_counts(self, counts: Dict[str, int]) -> None:
        for status in ("healthy", "unhealthy", "unknown"):
            value = counts.get(status, 0)
            self._workers.labels(router=self.router_name, status=status).set(value)

    def record_worker_state(self, worker: str, status: str) -> None:
        self._worker_state_changes.labels(router=self.router_name, worker=worker, status=status).inc()

    def record_worker_registration(self, action: str) -> None:
        self._worker_registrations.labels(router=self.router_name, action=action).inc()

    # ------------------------------------------------------------------
    # Proxy metrics helpers
    def proxy_inflight_inc(self, method: str) -> None:
        self._proxy_inflight.labels(router=self.router_name, method=method).inc()

    def proxy_inflight_dec(self, method: str) -> None:
        self._proxy_inflight.labels(router=self.router_name, method=method).dec()

    def record_proxy_attempt(self, method: str, outcome: str, status_code: str, duration: float) -> None:
        labels = dict(router=self.router_name, method=method, outcome=outcome)
        self._proxy_requests.labels(status_code=status_code, **labels).inc()
        self._proxy_latency.labels(**labels).observe(duration)

    def record_retry(self, reason: str) -> None:
        self._proxy_retries.labels(router=self.router_name, reason=reason).inc()

    def record_terminal_failure(self, reason: str) -> None:
        self._proxy_failures.labels(router=self.router_name, reason=reason).inc()

    # ------------------------------------------------------------------
    # Health probe metrics helpers
    def record_health_probe(self, worker: str, result: str, duration: float, success_timestamp: Optional[float] = None) -> None:
        labels = dict(router=self.router_name, worker=worker)
        self._health_probes.labels(result=result, **labels).inc()
        self._health_latency.labels(**labels).observe(duration)
        if result == "success" and success_timestamp is not None:
            self._health_last_success.labels(**labels).set(success_timestamp)

    # ------------------------------------------------------------------
    def start_http_server(self, host: str, port: int) -> bool:
        """Start the dedicated metrics HTTP server if not already running."""
        with self._http_server_lock:
            if self._http_server_port is not None:
                return True
            try:
                start_http_server(port, addr=host, registry=self.registry)
                self._http_server_port = port
                logger.info(f"Router Prometheus metrics server listening on {host}:{port}")
                return True
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error(f"Failed to start router metrics server on {host}:{port}: {exc}")
                return False
