from pathlib import Path

from sgorch.discover.adopt import parse_job_comment, WorkerDiscovery, AdoptionManager, DiscoveredWorker
from sgorch.config import (
    DeploymentConfig,
    ConnectivityConfig,
    RouterConfig,
    SlurmConfig,
    SGLangConfig,
    TEIConfig,
    HealthConfig,
    PolicyConfig,
)
from sgorch.backends import make_backend_adapter
from sgorch.slurm.base import JobInfo
from sgorch.net.ports import PortAllocator


def _dep_cfg(tmp_path):
    return DeploymentConfig(
        name="d",
        replicas=1,
        connectivity=ConnectivityConfig(
            mode="direct",
            tunnel_mode="local",
            orchestrator_host="o",
            advertise_host="127.0.0.1",
        ),
        router=RouterConfig(base_url="http://r"),
        slurm=SlurmConfig(account="a", partition="p", gres="g", log_dir=str(tmp_path)),
        sglang=SGLangConfig(model_path="/m"),
        health=HealthConfig(),
        policy=PolicyConfig(),
    )


def _tei_cfg(tmp_path):
    return DeploymentConfig(
        name="tei",
        replicas=1,
        connectivity=ConnectivityConfig(
            mode="direct",
            tunnel_mode="local",
            orchestrator_host="o",
            advertise_host="127.0.0.1",
        ),
        router=None,
        slurm=SlurmConfig(account="a", partition="p", gres="g", log_dir=str(tmp_path)),
        backend=TEIConfig(model_id="model", args=["--hostname", "0.0.0.0", "--port", "{PORT}"]),
        health=HealthConfig(),
        policy=PolicyConfig(),
    )


def test_parse_job_comment_valid_invalid():
    assert parse_job_comment("sgorch:d:uuid") == ("d", "uuid")
    assert parse_job_comment("") is None
    assert parse_job_comment("wrong:format") is None


def test_extract_from_logs_reads_READY_marker_and_ports(tmp_path):
    cfg = _dep_cfg(tmp_path)
    backend = make_backend_adapter(cfg)
    wd = WorkerDiscovery(cfg, slurm=None, backend=backend)  # slurm not used here

    # create a log file matching patterns with READY line
    log = Path(tmp_path) / f"sgl-{cfg.name}-x_123.out"
    log.write_text("\n".join([
        "noise", 
        "READY URL=http://10.0.0.1:8000 JOB=123 INSTANCE=abc",
    ]))

    job = JobInfo(job_id="123", state="RUNNING", node="n1", time_left_s=None)
    worker = wd._extract_from_logs(job)
    assert worker is not None
    assert worker.worker_url == "http://10.0.0.1:8000"
    assert worker.remote_port == 8000
    assert worker.instance_uuid == "abc"


def test_extract_from_logs_reads_ready_marker_for_tei(tmp_path):
    cfg = _tei_cfg(tmp_path)
    backend = make_backend_adapter(cfg)
    wd = WorkerDiscovery(cfg, slurm=None, backend=backend)

    log = Path(tmp_path) / f"tei-{cfg.name}-0_456.out"
    log.write_text("\n".join([
        "noise",
        "READY URL=http://10.0.0.5:3000 JOB=456 INSTANCE=abc",
    ]))

    job = JobInfo(job_id="456", state="RUNNING", node="n1", time_left_s=None)
    worker = wd._extract_from_logs(job)
    assert worker is not None
    assert worker.worker_url == "http://10.0.0.5:3000"
    assert worker.remote_port == 3000


def test_compute_from_job_info_guesses_ports_with_tcp_probe(monkeypatch, tmp_path):
    cfg = _dep_cfg(tmp_path)
    backend = make_backend_adapter(cfg)
    wd = WorkerDiscovery(cfg, slurm=None, backend=backend)
    job = JobInfo(job_id="1", state="RUNNING", node="node1", time_left_s=None)

    # resolve node ip returns fixed
    monkeypatch.setattr("sgorch.discover.adopt.resolve_slurm_node_ip", lambda n: "1.2.3.4")
    # pretend only port 8001 is open
    def fake_probe(host, port, timeout=2):
        return port == 8001
    monkeypatch.setattr("sgorch.net.hostaddr.test_tcp_connection", fake_probe)

    worker = wd._compute_from_job_info(job)
    assert worker is not None
    assert worker.worker_url == "http://1.2.3.4:8001"


def test_reconcile_router_state_removes_only_scoped_stale_urls(tmp_path):
    cfg = _dep_cfg(tmp_path)
    # limit local port range to a small window
    cfg.connectivity.local_port_range = (30000, 30005)
    pa = PortAllocator(cfg.connectivity.local_port_range)
    removed = []

    class _RC:
        def __init__(self, urls):
            self._urls = set(urls)
        def list(self):
            return set(self._urls)
        def remove(self, u):
            removed.append(u)

    backend = make_backend_adapter(cfg)
    am = AdoptionManager(cfg, slurm=None, router_client=_RC([
        "http://127.0.0.1:30000",  # safe
        "http://127.0.0.1:40000",  # out of range
        "http://10.0.0.1:30001",   # wrong host
    ]), port_allocator=pa, backend=backend)

    adopted = {
        "1": DiscoveredWorker(job_id="1", instance_uuid="u", worker_url="http://127.0.0.1:30000")
    }
    am.reconcile_router_state(adopted)
    # only the in-range 127.0.0.1 one should be removed if not adopted; adopted one stays
    assert removed == []  # none removed because the only in-scope is adopted
