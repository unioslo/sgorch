import httpx

from sgorch.health.http_probe import HealthProbe, HealthStatus
from sgorch.config import HealthConfig


def make_client(status_code=200, body="ok"):
    def handler(request):
        return httpx.Response(status_code, text=body)
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_probe_success_transitions_to_HEALTHY_after_threshold(monkeypatch):
    cfg = HealthConfig(interval_s=5, timeout_s=2, consecutive_ok_for_ready=2, failures_to_unhealthy=2)
    p = HealthProbe(cfg, "http://w")
    # Inject mock client
    p.client = make_client(200, "ok")
    r1 = p.probe()
    assert p.get_status() in (HealthStatus.STARTING, HealthStatus.HEALTHY)
    r2 = p.probe()
    assert p.get_status() == HealthStatus.HEALTHY


def test_probe_failures_transition_to_UNHEALTHY_after_threshold(monkeypatch):
    cfg = HealthConfig(interval_s=5, timeout_s=2, consecutive_ok_for_ready=2, failures_to_unhealthy=2)
    p = HealthProbe(cfg, "http://w")
    p.client = make_client(503, "no")
    p.probe()
    p.probe()
    assert p.get_status() == HealthStatus.UNHEALTHY


def test_backoff_schedules_next_probe_slower_on_failures(monkeypatch):
    cfg = HealthConfig(interval_s=2, timeout_s=1, consecutive_ok_for_ready=1, failures_to_unhealthy=1)
    p = HealthProbe(cfg, "http://w")
    p.client = make_client(503, "no")
    p.probe()
    # After a failure, next probe should not be immediately due
    assert not p.should_probe()


def test_should_probe_respects_next_probe_earliest(monkeypatch):
    cfg = HealthConfig(interval_s=1, timeout_s=1, consecutive_ok_for_ready=1, failures_to_unhealthy=1)
    p = HealthProbe(cfg, "http://w")
    p.client = make_client(200, "ok")
    assert p.should_probe()
    p.probe()
    # Immediately after success, should not probe yet
    assert not p.should_probe()

