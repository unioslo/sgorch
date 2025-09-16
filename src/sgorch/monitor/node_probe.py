import os
import time
import threading
from typing import Optional

from ..logging_setup import get_logger
from ..config import DeploymentConfig, NodeProbeConfig
from ..router.client import RouterClient
from ..metrics.prometheus import get_metrics


logger = get_logger(__name__)


"""Per-node probe utilities."""


class NodeProbeWorker(threading.Thread):
    """Background worker that probes a single worker node via OpenAI-compatible API."""

    def __init__(self, deployment_name: str, worker_url: str, cfg: NodeProbeConfig, auth_cfg):
        super().__init__(name=f"node-probe-{deployment_name}-{worker_url}", daemon=True)
        self.deployment_name = deployment_name
        self.worker_url = worker_url.rstrip("/")
        self.cfg = cfg
        self.auth_cfg = auth_cfg
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        metrics = get_metrics()
        interval = max(1, int(self.cfg.interval_s))

        while not self._stop_event.is_set():
            start = time.perf_counter()
            success = False
            try:
                client, model = self._create_openai_client()
                prompt = self.cfg.prompt or "ping"
                result = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1,
                    timeout=self.cfg.timeout_s,
                )
                success = bool(getattr(result, "choices", None))
            except Exception as e:
                logger.debug(
                    f"Node probe failed for {self.deployment_name} {self.worker_url}: {e}"
                )

            latency = time.perf_counter() - start
            try:
                metrics.record_node_probe(self.deployment_name, self.worker_url, success, latency)
            except Exception as e:
                logger.error(f"Recording node probe metric failed: {e}")

            self._stop_event.wait(interval)

    def _create_openai_client(self):
        from openai import OpenAI

        base = self.worker_url
        if not base.endswith("/v1"):
            base = base + "/v1"

        model = self.cfg.model or "gpt-4o-mini"

        default_headers = {}
        api_key: Optional[str] = None
        auth = self.auth_cfg
        if auth and getattr(auth, 'type', None) == "header":
            token_value = os.getenv(auth.header_value_env)
            if token_value:
                if auth.header_name.lower() == "authorization":
                    if token_value.lower().startswith("bearer "):
                        api_key = token_value.split(" ", 1)[1]
                    else:
                        api_key = token_value
                else:
                    default_headers[auth.header_name] = token_value

        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY", "nokey")

        client = OpenAI(base_url=base, api_key=api_key, default_headers=default_headers or None)
        return client, model


class NodeProbeManager:
    """Manages per-deployment per-node probe workers by syncing with the router."""

    def __init__(self, cfg: NodeProbeConfig):
        self.cfg = cfg
        self._syncers: dict[str, threading.Thread] = {}
        self._workers: dict[str, dict[str, NodeProbeWorker]] = {}
        self._stops: dict[str, threading.Event] = {}

    def start_for(self, deployment: DeploymentConfig, router_client: RouterClient | None) -> None:
        if not self.cfg.enabled:
            return
        if router_client is None:
            logger.info(f"Node probe disabled for {deployment.name}: backend has no router")
            return
        name = deployment.name
        if name in self._syncers:
            return
        stop_evt = threading.Event()
        self._stops[name] = stop_evt
        self._workers[name] = {}

        def _sync_loop():
            while not stop_evt.is_set():
                try:
                    urls = set()
                    try:
                        urls = router_client.list()
                    except Exception as e:
                        logger.debug(f"Node probe sync: list failed for {name}: {e}")

                    current = self._workers.get(name, {})
                    # Start new workers
                    for url in urls:
                        if url not in current:
                            w = NodeProbeWorker(name, url, self.cfg, deployment.router.auth)
                            current[url] = w
                            w.start()
                            logger.info(f"Node probe started for {name} {url}")
                    # Stop removed workers
                    for url in list(current.keys()):
                        if url not in urls:
                            try:
                                current[url].stop()
                                current[url].join(timeout=5.0)
                            except Exception:
                                pass
                            current.pop(url, None)
                except Exception as e:
                    logger.error(f"Node probe sync error for {name}: {e}")
                # Resync at same cadence as probe interval, min 10s
                stop_evt.wait(max(10, int(self.cfg.interval_s)))

        t = threading.Thread(target=_sync_loop, name=f"node-probe-sync-{name}", daemon=True)
        self._syncers[name] = t
        t.start()

    def stop_all(self) -> None:
        # Stop syncers
        for name, evt in list(self._stops.items()):
            try:
                evt.set()
            except Exception:
                pass
        for name, t in list(self._syncers.items()):
            try:
                t.join(timeout=5.0)
            except Exception:
                pass
        self._syncers.clear()
        # Stop workers
        for name, workers in list(self._workers.items()):
            for url, w in list(workers.items()):
                try:
                    w.stop()
                except Exception:
                    pass
        for name, workers in list(self._workers.items()):
            for url, w in list(workers.items()):
                try:
                    w.join(timeout=5.0)
                except Exception:
                    pass
        self._workers.clear()
