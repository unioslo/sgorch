from sgorch.metrics.prometheus import SGOrchMetrics, get_metrics


def test_start_http_server_respects_enabled_flag(monkeypatch):
    called = {"start": 0}

    def fake_start_http_server(port, addr="", registry=None):
        called["start"] += 1

    monkeypatch.setattr("sgorch.metrics.prometheus.start_http_server", fake_start_http_server)
    m = SGOrchMetrics()
    # Disabled -> no start
    assert m.start_http_server(type("C", (), {"enabled": False, "bind": "0.0.0.0", "port": 9999})()) is False
    assert called["start"] == 0
    # Enabled -> start once
    assert m.start_http_server(type("C", (), {"enabled": True, "bind": "0.0.0.0", "port": 9999})()) is True
    assert called["start"] == 1
    # Second call warns but returns True and not restart
    assert m.start_http_server(type("C", (), {"enabled": True, "bind": "0.0.0.0", "port": 9999})()) is True
    assert called["start"] == 1


def test_recorders_update_labels_without_crash():
    m = get_metrics()
    m.update_worker_counts("d", desired=2, ready=1, starting=1, unhealthy=0)
    m.update_tunnel_count("d", 1)
    m.record_restart("d", "unhealthy")
    m.record_job_submitted("d")
    m.record_job_failed("d", "oops")
    m.record_router_operation("d", "add", True)
    m.record_router_operation("d", "add", False)
    m.record_slurm_operation("d", "submit", True)
    m.record_slurm_operation("d", "submit", False)
    m.record_health_check("d", True, 0.01)
    m.record_reconcile_duration("d", 0.02)
    m.update_port_counts("d", 1, 10)
    m.update_blacklisted_nodes("d", 0)
    assert m.is_running() in (True, False)

