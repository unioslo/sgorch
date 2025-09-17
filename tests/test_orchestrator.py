import threading
from types import SimpleNamespace

from sgorch.orchestrator import Orchestrator
from sgorch.config import Config, DeploymentConfig, ConnectivityConfig, RouterConfig, SlurmConfig, SGLangConfig, TEIConfig


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


def _tei_cfg(tmp_path):
    return Config(
        deployments=[
            DeploymentConfig(
                name="tei",
                replicas=1,
                connectivity=ConnectivityConfig(mode="direct", tunnel_mode="local", orchestrator_host="o", advertise_host="127.0.0.1"),
                router=None,
                slurm=SlurmConfig(account="a", partition="p", gres="g", log_dir=str(tmp_path)),
                backend=TEIConfig(model_id="model", args=["--hostname", "0.0.0.0", "--port", "{PORT}"]),
            )
        ]
    )


def _tei_cfg_with_router(tmp_path):
    return Config(
        deployments=[
            DeploymentConfig(
                name="tei",
                replicas=1,
                connectivity=ConnectivityConfig(mode="direct", tunnel_mode="local", orchestrator_host="o", advertise_host="127.0.0.1"),
                router=RouterConfig(base_url="http://router:25000"),
                slurm=SlurmConfig(account="a", partition="p", gres="g", log_dir=str(tmp_path)),
                backend=TEIConfig(model_id="model", args=["--hostname", "0.0.0.0", "--port", "{PORT}"]),
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
    orch.running = True
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


def test_orchestrator_handles_routerless_tei(monkeypatch, tmp_path):
    monkeypatch.setattr("sgorch.orchestrator.get_metrics", lambda: SimpleNamespace(start_http_server=lambda *_: True))

    created = {}

    class _R:
        def __init__(self, deployment_config, *a, **k):
            created["deployment"] = deployment_config
        def tick(self):
            pass
        def shutdown(self):
            pass

    monkeypatch.setattr("sgorch.orchestrator.Reconciler", _R)

    orch = Orchestrator(_tei_cfg(tmp_path))
    assert created["deployment"].backend.type == "tei"
    orch.shutdown()


def test_orchestrator_handles_tei_with_router(monkeypatch, tmp_path):
    monkeypatch.setattr("sgorch.orchestrator.get_metrics", lambda: SimpleNamespace(start_http_server=lambda *_: True))

    called = {"health": False}

    def _health(self):
        called["health"] = True
        return True

    monkeypatch.setattr("sgorch.router.client.RouterClient.health_check", _health)

    captured = {}

    class _R:
        def __init__(self, deployment_config, slurm, router_client, notifier, state_store, backend_adapter):
            captured["router_client"] = router_client
        def tick(self):
            pass
        def shutdown(self):
            pass

    monkeypatch.setattr("sgorch.orchestrator.Reconciler", _R)

    orch = Orchestrator(_tei_cfg_with_router(tmp_path))
    try:
        assert called["health"]
        assert captured["router_client"] is not None
    finally:
        orch.shutdown()
