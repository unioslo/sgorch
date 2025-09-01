from __future__ import annotations
import threading
from typing import Dict, List, Iterable

import httpx
from prometheus_client.parser import text_string_to_metric_families
from prometheus_client.core import Metric

from ..logging_setup import get_logger


logger = get_logger(__name__)


class WorkerMetricsCollector:
    """
    A Prometheus collector that scrapes /metrics from worker nodes and re-emits
    them with additional labels: deployment and worker.

    The collector does not schedule any background work; it fetches metrics on demand
    when Prometheus scrapes the orchestrator endpoint.
    """

    def __init__(self, endpoints_lookup: callable[[], Dict[str, List[str]]], timeout_s: float = 1.5):
        self._endpoints_lookup = endpoints_lookup
        self._timeout_s = timeout_s

    def collect(self) -> Iterable[Metric]:  # type: ignore[override]
        mapping = self._endpoints_lookup()
        if not mapping:
            return []

        # Fetch metrics for each worker sequentially with short timeouts
        # to avoid long scrape times.
        for deployment, endpoints in mapping.items():
            for base_url in endpoints:
                try:
                    url = base_url.rstrip("/") + "/metrics"
                    with httpx.Client(timeout=self._timeout_s) as client:
                        resp = client.get(url)
                        resp.raise_for_status()
                        text = resp.text
                except Exception as e:
                    logger.debug(f"Worker metrics scrape failed for {base_url}: {e}")
                    continue

                # Derive a concise worker label (host:port)
                worker_label = base_url
                try:
                    from urllib.parse import urlparse
                    parsed = urlparse(base_url)
                    if parsed.hostname and parsed.port:
                        worker_label = f"{parsed.hostname}:{parsed.port}"
                except Exception:
                    pass

                try:
                    for family in text_string_to_metric_families(text):
                        # Rebuild each metric family with extra labels
                        m = Metric(family.name, family.documentation, family.type)
                        for sample in family.samples:
                            labels = dict(sample.labels)
                            labels["deployment"] = deployment
                            labels["worker"] = worker_label
                            m.add_sample(sample.name, labels=labels, value=sample.value, timestamp=sample.timestamp, exemplar=sample.exemplar)
                        yield m
                except Exception as e:
                    logger.debug(f"Failed to parse worker metrics from {base_url}: {e}")


class WorkerMetricsProxy:
    """
    Maintains a mapping of deployment -> list of worker base URLs, and exposes
    a collector that pulls metrics from those endpoints.
    """

    def __init__(self, registry):
        self._registry = registry
        self._lock = threading.Lock()
        # Only deployments with metrics enabled are included
        self._deploy_to_endpoints: Dict[str, List[str]] = {}

        # Register collector with the provided registry
        self._collector = WorkerMetricsCollector(self._snapshot_endpoints)
        try:
            self._registry.register(self._collector)
        except ValueError:
            # Already registered; harmless in most flows
            pass

    def _snapshot_endpoints(self) -> Dict[str, List[str]]:
        with self._lock:
            # Return a shallow copy to avoid race conditions during iteration
            return {k: list(v) for k, v in self._deploy_to_endpoints.items()}

    def set_endpoints(self, deployment: str, endpoints: List[str], enabled: bool) -> None:
        """Update the set of worker endpoints for a deployment.

        If disabled, the deployment is removed from the mapping.
        """
        with self._lock:
            if enabled and endpoints:
                # Normalize URLs (strip trailing slashes)
                normed = []
                seen = set()
                for e in endpoints:
                    u = e.rstrip("/")
                    if u and u not in seen:
                        seen.add(u)
                        normed.append(u)
                self._deploy_to_endpoints[deployment] = normed
            else:
                self._deploy_to_endpoints.pop(deployment, None)

