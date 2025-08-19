import os
import time
import threading
from typing import Optional

from ..logging_setup import get_logger
from ..config import DeploymentConfig, RouterProbeConfig
from ..metrics.prometheus import get_metrics


logger = get_logger(__name__)


class RouterProbeWorker(threading.Thread):
    """Background worker that probes a deployment's router via OpenAI-compatible API."""

    def __init__(self, deployment: DeploymentConfig, cfg: RouterProbeConfig):
        super().__init__(name=f"router-probe-{deployment.name}", daemon=True)
        self.deployment = deployment
        self.cfg = cfg
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
                # Prepare OpenAI client
                client, model = self._create_openai_client()
                # Minimal, cheap chat.completions request
                prompt = self.cfg.prompt or "ping"
                result = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1,
                    timeout=self.cfg.timeout_s,
                )
                # Consider success if a choice is returned
                success = bool(getattr(result, "choices", None))
            except Exception as e:
                logger.warning(
                    f"Router probe failed for deployment {self.deployment.name}: {e}"
                )

            latency = time.perf_counter() - start
            try:
                metrics.record_router_probe(self.deployment.name, success, latency)
            except Exception as e:
                logger.error(f"Recording router probe metric failed: {e}")

            # Sleep until next interval or until stopped
            self._stop_event.wait(interval)

    def _create_openai_client(self):
        """Create an OpenAI client configured to hit the router base_url."""
        from openai import OpenAI  # Lazy import to avoid hard dependency for tests

        # Derive base URL, ensure it ends with /v1 for OpenAI client
        base = self.deployment.router.base_url.rstrip("/")
        if not base.endswith("/v1"):
            base = base + "/v1"

        # Determine model name
        model = self.cfg.model or self.deployment.sglang.model_path

        # Build headers/auth
        default_headers = {}
        api_key: Optional[str] = None
        auth = self.deployment.router.auth
        if auth and auth.type == "header":
            token_value = os.getenv(auth.header_value_env)
            if token_value:
                # If using Authorization, map to api_key if possible
                if auth.header_name.lower() == "authorization":
                    if token_value.lower().startswith("bearer "):
                        api_key = token_value.split(" ", 1)[1]
                    else:
                        api_key = token_value
                else:
                    default_headers[auth.header_name] = token_value
            else:
                logger.warning(
                    f"Router probe: env {auth.header_value_env} not set for deployment {self.deployment.name}"
                )

        # Fallback API key to a dummy if none was provided (client requires something)
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY", "nokey")

        client = OpenAI(base_url=base, api_key=api_key, default_headers=default_headers or None)
        return client, model


class RouterProbeManager:
    """Manages router probe workers for all deployments."""

    def __init__(self, cfg: RouterProbeConfig):
        self.cfg = cfg
        self._workers: dict[str, RouterProbeWorker] = {}

    def start_for(self, deployment: DeploymentConfig) -> None:
        if not self.cfg.enabled:
            return
        if deployment.name in self._workers:
            return
        worker = RouterProbeWorker(deployment, self.cfg)
        self._workers[deployment.name] = worker
        worker.start()
        logger.info(
            f"Started router probe for deployment {deployment.name} at interval {self.cfg.interval_s}s"
        )

    def stop_all(self) -> None:
        for name, w in list(self._workers.items()):
            try:
                w.stop()
            except Exception:
                pass
        for name, w in list(self._workers.items()):
            try:
                w.join(timeout=5.0)
            except Exception:
                pass
        self._workers.clear()

