import threading
from types import SimpleNamespace

from sgorch.orchestrator import Orchestrator
from sgorch.config import Config, DeploymentConfig, ConnectivityConfig, RouterConfig, SlurmConfig, SGLangConfig


def _cfg(tmp_path):
    return Config(
        deployments=[
            DeploymentConfig(
                name="d1",
                replicas=1,
                connectivity=ConnectivityConfig(mode="direct", tunnel_mode="local", orchestrator_host="o", advertise_host="127.0.0.1"),
                router=RouterConfig(base_url="http://r"),
                slurm=SlurmConfig(account="a", partition="p", gres="g", log_dir=str(tmp_path)),
                sglang=SGLangConfig(model_path="/m"),
            )
        ]
    )


def test_orchestrator_creates_reconcilers_and_threads(monkeypatch, tmp_path):
    # stub RouterClient.health_check to True
    monkeypatch.setattr("sgorch.router.client.RouterClient.health_check", lambda self: True)
    # stub get_metrics().start_http_server
    monkeypatch.setattr("sgorch.orchestrator.get_metrics", lambda: SimpleNamespace(start_http_server=lambda *_: True))
    # make Reconciler a no-op
    class _R:
        def __init__(self, *a, **k):
            pass
        def tick(self):
            pass
        def shutdown(self):
            pass
    monkeypatch.setattr("sgorch.orchestrator.Reconciler", _R)

    orch = Orchestrator(_cfg(tmp_path))
    orch._start_reconciler_threads()
    try:
        assert len(orch.reconcilers) == 1
        assert len(orch.threads) == 1
        assert all(t.is_alive() for t in orch.threads)
    finally:
        orch.shutdown()


def test_orchestrator_shutdown_idempotent(monkeypatch, tmp_path):
    monkeypatch.setattr("sgorch.router.client.RouterClient.health_check", lambda self: True)
    monkeypatch.setattr("sgorch.orchestrator.get_metrics", lambda: SimpleNamespace(start_http_server=lambda *_: True))
    class _R:
        def __init__(self, *a, **k):
            pass
        def tick(self):
            pass
        def shutdown(self):
            pass
    monkeypatch.setattr("sgorch.orchestrator.Reconciler", _R)

    orch = Orchestrator(_cfg(tmp_path))
    orch._start_reconciler_threads()
    orch.shutdown()
    # second shutdown should be no-op
    orch.shutdown()

