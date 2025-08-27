from types import SimpleNamespace

from sgorch.reconciler import Reconciler
from sgorch.config import (
    DeploymentConfig,
    ConnectivityConfig,
    RouterConfig,
    SlurmConfig,
    SGLangConfig,
    HealthConfig,
    PolicyConfig,
)


def _dep_cfg(tmp_path):
    return DeploymentConfig(
        name="d",
        replicas=2,
        connectivity=ConnectivityConfig(
            mode="direct",
            tunnel_mode="local",
            orchestrator_host="o",
            advertise_host="127.0.0.1",
            local_port_range=(40000, 40010),
        ),
        router=RouterConfig(base_url="http://router"),
        slurm=SlurmConfig(
            account="a",
            partition="p",
            gres="g",
            log_dir=str(tmp_path),
            time_limit="01:00:00",
            time_limit_stagger_s=300,  # 5 minutes
        ),
        sglang=SGLangConfig(model_path="/m"),
        health=HealthConfig(),
        policy=PolicyConfig(restart_backoff_s=1, deregister_grace_s=0),
    )


class _FakeSlurm:
    def submit(self, spec):
        return "1"

    def status(self, job_id):
        return SimpleNamespace(job_id=job_id, state="UNKNOWN", node=None, time_left_s=None)

    def cancel(self, job_id):
        return None

    def list_jobs(self, name_prefix: str):
        return []


class _FakeRouter:
    def list(self):
        return set()

    def add(self, url):
        pass

    def remove(self, url):
        pass


class _FakeMetrics:
    def __getattr__(self, name):
        def _f(*a, **k):
            return None
        return _f


class _FakeStateStore:
    def __init__(self):
        self.snapshots = {}

    def load_deployment(self, name):
        return self.snapshots.get(name)

    def save_deployment(self, snapshot):
        self.snapshots[snapshot.name] = snapshot


def test_time_limit_stagger_applies_per_instance(monkeypatch, tmp_path):
    monkeypatch.setattr("sgorch.reconciler.get_metrics", lambda: _FakeMetrics())
    cfg = _dep_cfg(tmp_path)
    r = Reconciler(cfg, _FakeSlurm(), _FakeRouter(), notifier=None, state_store=_FakeStateStore())  # type: ignore

    # instance 0: base time limit
    spec0 = r._create_job_spec(0, "u0", 40001)
    assert spec0.time_limit == "01:00:00"

    # instance 1: +5 minutes
    spec1 = r._create_job_spec(1, "u1", 40002)
    assert spec1.time_limit == "01:05:00"

