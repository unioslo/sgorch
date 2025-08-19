from sgorch.metrics.prometheus import SGOrchMetrics


def test_is_running_reflects_state(monkeypatch):
    called = {"start": 0}

    def fake_start_http_server(port, addr="", registry=None):
        called["start"] += 1

    monkeypatch.setattr("sgorch.metrics.prometheus.start_http_server", fake_start_http_server)

    m = SGOrchMetrics()
    # initially not running
    assert m.is_running() is False
    # disabled start
    assert m.start_http_server(type("C", (), {"enabled": False, "bind": "0.0.0.0", "port": 8000})()) is False
    assert m.is_running() is False
    # enabled start -> running
    assert m.start_http_server(type("C", (), {"enabled": True, "bind": "0.0.0.0", "port": 8000})()) is True
    assert called["start"] == 1
    assert m.is_running() is True
    # second call keeps running, does not call underlying start again
    assert m.start_http_server(type("C", (), {"enabled": True, "bind": "0.0.0.0", "port": 8000})()) is True
    assert called["start"] == 1

