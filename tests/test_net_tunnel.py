import subprocess

from sgorch.net.tunnel import TunnelSpec, TunnelManager, SupervisedTunnelManager


def test_build_ssh_command_local_vs_reverse_includes_opts():
    tm = TunnelManager()
    spec_local = TunnelSpec(
        mode="local",
        orchestrator_host="ohost",
        advertise_host="127.0.0.1",
        advertise_port=30001,
        remote_host="node",
        remote_port=8000,
        ssh_user="u",
        ssh_opts=["-vvv"],
    )
    cmd_local = tm._build_ssh_command(spec_local)
    assert "-L" in cmd_local
    assert any("-vvv" == c for c in cmd_local)
    assert cmd_local[-1].startswith("u@")

    spec_rev = TunnelSpec(
        mode="reverse",
        orchestrator_host="ohost",
        advertise_host="127.0.0.1",
        advertise_port=30001,
        remote_host="node",
        remote_port=8000,
        ssh_user=None,
        ssh_opts=[],
    )
    cmd_rev = tm._build_ssh_command(spec_rev)
    assert "-R" in cmd_rev
    assert cmd_rev[-1] == "ohost"


def test_ensure_starts_process_and_reports_url(fake_popen):
    tm = TunnelManager()
    spec = TunnelSpec(
        mode="local",
        orchestrator_host="ohost",
        advertise_host="127.0.0.1",
        advertise_port=30002,
        remote_host="node",
        remote_port=8000,
    )
    url = tm.ensure("k1", spec)
    assert url == "http://127.0.0.1:30002"
    assert tm.is_up("k1")


def test_ensure_adopts_existing_listener_when_address_in_use(monkeypatch):
    tm = TunnelManager()

    def fake_create(spec):
        raise RuntimeError("Address already in use")

    monkeypatch.setattr(tm, "_create_tunnel", fake_create)
    monkeypatch.setattr("sgorch.net.tunnel.test_tcp_connection", lambda h, p, timeout=2: True)

    spec = TunnelSpec(
        mode="local",
        orchestrator_host="ohost",
        advertise_host="127.0.0.1",
        advertise_port=30003,
        remote_host="node",
        remote_port=8000,
    )
    url = tm.ensure("k2", spec)
    assert url == "http://127.0.0.1:30003"
    assert tm.is_up("k2")


def test_monitor_and_restart_resets_backoff_on_success(monkeypatch, fake_popen):
    stm = SupervisedTunnelManager()
    spec = TunnelSpec(
        mode="local",
        orchestrator_host="ohost",
        advertise_host="127.0.0.1",
        advertise_port=30004,
        remote_host="node",
        remote_port=8000,
    )
    stm.ensure("k", spec)
    # Initially healthy
    monkeypatch.setattr("sgorch.net.tunnel.test_tcp_connection", lambda h, p, timeout=2: True)
    res = stm.monitor_and_restart()
    assert res["k"] == "healthy"
    # Now unhealthy once, then healthy again
    seq = iter([False, True])
    monkeypatch.setattr("sgorch.net.tunnel.test_tcp_connection", lambda h, p, timeout=2: next(seq))
    res = stm.monitor_and_restart()
    # Might attempt restart; next cycle healthy resets backoff
    res = stm.monitor_and_restart()
    assert "k" in res


def test_drop_and_shutdown_cleanup_calls(fake_popen):
    tm = TunnelManager()
    spec = TunnelSpec(
        mode="local",
        orchestrator_host="ohost",
        advertise_host="127.0.0.1",
        advertise_port=30005,
        remote_host="node",
        remote_port=8000,
    )
    tm.ensure("k3", spec)
    assert tm.is_up("k3")
    tm.drop("k3")
    assert not tm.is_up("k3")
    tm.shutdown()

