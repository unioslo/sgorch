import threading
from types import SimpleNamespace

from sgorch.reconciler import Reconciler, WorkerState
from sgorch.config import (
    DeploymentConfig,
    ConnectivityConfig,
    RouterConfig,
    SlurmConfig,
    SGLangConfig,
    HealthConfig,
    PolicyConfig,
)
from sgorch.health.http_probe import HealthStatus


def _dep_cfg(tmp_path, mode="direct"):
    return DeploymentConfig(
        name="d",
        replicas=2,
        connectivity=ConnectivityConfig(
            mode=mode,
            tunnel_mode="local",
            orchestrator_host="o",
            advertise_host="127.0.0.1",
            local_port_range=(40000, 40010),
        ),
        router=RouterConfig(base_url="http://router"),
        slurm=SlurmConfig(account="a", partition="p", gres="g", log_dir=str(tmp_path)),
        sglang=SGLangConfig(model_path="/m"),
        health=HealthConfig(),
        policy=PolicyConfig(restart_backoff_s=1, deregister_grace_s=0),
    )


class _FakeSlurm:
    def __init__(self):
        self.submits = []
        self.cancels = []
        self.jobs = []

    def submit(self, spec):
        jid = str(len(self.submits) + 1)
        self.submits.append(spec)
        return jid

    def status(self, job_id):
        return SimpleNamespace(job_id=job_id, state="UNKNOWN", node=None, time_left_s=None)

    def cancel(self, job_id):
        self.cancels.append(job_id)

    def list_jobs(self, name_prefix: str):
        return list(self.jobs)


class _FakeRouter:
    def __init__(self):
        self._workers = set()
        self.added = []
        self.removed = []

    def list(self):
        return set(self._workers)

    def add(self, url):
        self._workers.add(url)
        self.added.append(url)

    def remove(self, url):
        self._workers.discard(url)
        self.removed.append(url)

    def health_check(self):
        return True


class _FakeMetrics:
    def __getattr__(self, name):
        def _f(*a, **k):
            return None
        return _f


class _FakeHealthMonitor:
    def __init__(self):
        self.probes = {}
        self.added = []
        self.removed = []
        self.results = {}

    def add_worker(self, url, cfg):
        self.added.append(url)
        # default unknown; updated on probe
        self.probes[url] = SimpleNamespace(get_status=lambda: HealthStatus.UNKNOWN)

    def remove_worker(self, url):
        self.removed.append(url)
        self.probes.pop(url, None)

    def probe_all_due(self):
        # Reflect results into probe status so Reconciler sees HEALTHY transitions
        for url, res in self.results.items():
            self.probes[url] = SimpleNamespace(get_status=lambda res=res: res.status)
        return dict(self.results)


class _FakeStateStore:
    def __init__(self):
        self.snapshots = {}

    def load_deployment(self, name):
        return self.snapshots.get(name)

    def save_deployment(self, snapshot):
        self.snapshots[snapshot.name] = snapshot

    def delete_deployment(self, name):
        self.snapshots.pop(name, None)


def test_reconciler_scales_up_and_persists(monkeypatch, tmp_path):
    cfg = _dep_cfg(tmp_path)
    fake_slurm = _FakeSlurm()
    fake_router = _FakeRouter()
    fake_state = _FakeStateStore()
    monkeypatch.setattr("sgorch.reconciler.get_metrics", lambda: _FakeMetrics())

    r = Reconciler(cfg, fake_slurm, fake_router, notifier=None, state_store=fake_state)  # type: ignore
    # avoid health/tunnel side effects during this test
    r.health_monitor = _FakeHealthMonitor()
    r.tunnel_manager.ensure = lambda *a, **k: ""

    r.tick()

    assert len(r.workers) == 2
    assert len(fake_slurm.submits) == 2
    # snapshot saved with workers and allocated ports
    snap = fake_state.snapshots.get("d")
    assert snap is not None
    assert len(snap.workers) == 2
    assert len(snap.allocated_ports) >= 2


def test_manage_tunnels_sets_advertised_url_and_health(monkeypatch, tmp_path):
    cfg = _dep_cfg(tmp_path, mode="tunneled")
    fake_slurm = _FakeSlurm()
    fake_router = _FakeRouter()
    fake_state = _FakeStateStore()
    monkeypatch.setattr("sgorch.reconciler.get_metrics", lambda: _FakeMetrics())

    r = Reconciler(cfg, fake_slurm, fake_router, notifier=None, state_store=fake_state)  # type: ignore
    fh = _FakeHealthMonitor()
    r.health_monitor = fh
    # seed a worker
    r.workers["1"] = WorkerState(job_id="1", instance_uuid="u", node="n", remote_port=8000, advertise_port=30000)
    r.tunnel_manager.ensure = lambda *a, **k: "http://127.0.0.1:30000"

    r._manage_tunnels()
    w = r.workers["1"]
    assert w.advertised_url == "http://127.0.0.1:30000"
    assert "http://127.0.0.1:30000" in fh.added


def test_health_changes_register_with_router(monkeypatch, tmp_path):
    cfg = _dep_cfg(tmp_path, mode="tunneled")
    fake_slurm = _FakeSlurm()
    fake_router = _FakeRouter()
    fake_state = _FakeStateStore()
    monkeypatch.setattr("sgorch.reconciler.get_metrics", lambda: _FakeMetrics())

    r = Reconciler(cfg, fake_slurm, fake_router, notifier=None, state_store=fake_state)  # type: ignore
    fh = _FakeHealthMonitor()
    r.health_monitor = fh
    # worker with advertised URL
    w = WorkerState(job_id="1", instance_uuid="u", node="n", remote_port=8000, advertise_port=30000,
                    worker_url="http://127.0.0.1:30000", advertised_url="http://127.0.0.1:30000")
    r.workers[w.job_id] = w
    # prevent immediate deletion due to grace period check
    import time as _t
    r.workers[w.job_id].submitted_at = _t.time()

    # mark healthy and reconcile router directly
    r.workers[w.job_id].health_status = HealthStatus.HEALTHY
    r._reconcile_router()
    assert fake_router.added == ["http://127.0.0.1:30000"]


def test_failed_worker_cleanup(monkeypatch, tmp_path):
    from sgorch.slurm.base import JobInfo

    cfg = _dep_cfg(tmp_path)
    fake_slurm = _FakeSlurm()
    fake_router = _FakeRouter()
    fake_state = _FakeStateStore()
    monkeypatch.setattr("sgorch.reconciler.get_metrics", lambda: _FakeMetrics())

    r = Reconciler(cfg, fake_slurm, fake_router, notifier=None, state_store=fake_state)  # type: ignore
    # seed a worker
    w = WorkerState(job_id="1", instance_uuid="u", node="n", remote_port=8000,
                    advertise_port=30000, worker_url="http://127.0.0.1:30000",
                    advertised_url="http://127.0.0.1:30000", health_status=HealthStatus.UNHEALTHY)
    r.workers[w.job_id] = w
    # list_jobs returns FAILED for job
    fake_slurm.jobs = [JobInfo(job_id="1", state="FAILED", node="n", time_left_s=None)]

    # prevent scale-up from reusing job_id '1' after removal
    r.config.replicas = 0
    r.tick()
    # removed
    assert "1" not in r.workers
    # router remove called
    assert fake_router.removed[-1] == "http://127.0.0.1:30000"
