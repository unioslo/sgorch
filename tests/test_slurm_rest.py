import json
import httpx

from sgorch.slurm.rest import SlurmRestAdapter


def _client_with_routes(routes):
    def handler(request: httpx.Request):
        key = (request.method, request.url.path)
        if key in routes:
            status, data = routes[key]
            return httpx.Response(status, json=data)
        # support prefix match for dynamic job id paths
        for (meth, path), (status, data) in routes.items():
            if meth == request.method and path.endswith("*") and request.url.path.startswith(path[:-1]):
                return httpx.Response(status, json=data)
        return httpx.Response(404, json={"errors": ["not found"]})

    transport = httpx.MockTransport(handler)
    return httpx.Client(transport=transport, base_url="http://slurm")


def test_submit_success_and_error_payloads(monkeypatch):
    routes = {
        ("POST", "/slurm/v0.0.39/job/submit"): (200, {"job_id": 123}),
    }
    c = _client_with_routes(routes)
    adapter = SlurmRestAdapter("http://slurm")
    adapter.client = c

    # success path
    from sgorch.slurm.base import SubmitSpec
    spec = SubmitSpec(
        name="n", account="a", reservation=None, partition="p", qos=None,
        gres="g", constraint=None, time_limit="01:00:00", cpus_per_task=1, mem="1G",
        env={}, stdout="/o", stderr="/e", script="#"
    )
    jid = adapter.submit(spec)
    assert jid == "123"

    # error payload
    routes[("POST", "/slurm/v0.0.39/job/submit")] = (200, {"errors": ["bad"]})
    c = _client_with_routes(routes)
    adapter.client = c
    try:
        adapter.submit(spec)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_status_maps_states_and_node_parsing(monkeypatch, freeze_time):
    # running job with start_time and time_limit minutes -> compute time_left_s
    start = freeze_time.now()
    routes = {
        ("GET", "/slurm/v0.0.39/job/42"): (200, {
            "jobs": [{
                "job_id": 42,
                "job_state": "RUNNING",
                "nodes": "node1,node2",
                "time_limit": 2,  # minutes
                "start_time": start - 30,
            }]
        }),
        ("GET", "/slurm/v0.0.39/job/43"): (200, {"errors": ["x"]}),
        ("GET", "/slurm/v0.0.39/job/44"): (404, {}),
    }
    c = _client_with_routes(routes)
    adapter = SlurmRestAdapter("http://slurm")
    adapter.client = c

    info = adapter.status("42")
    assert info.state == "RUNNING"
    assert info.node == "node1"
    assert 50 <= info.time_left_s <= 90  # approx 2min - 30s

    info2 = adapter.status("43")
    assert info2.state == "UNKNOWN" and info2.node is None

    info3 = adapter.status("44")
    assert info3.state == "UNKNOWN"


def test_cancel_404_is_ignored_others_raise():
    routes = {
        ("DELETE", "/slurm/v0.0.39/job/1"): (404, {}),
        ("DELETE", "/slurm/v0.0.39/job/2"): (500, {"errors": ["boom"]}),
    }
    c = _client_with_routes(routes)
    adapter = SlurmRestAdapter("http://slurm")
    adapter.client = c

    # 404 ignored
    adapter.cancel("1")

    # others raise
    try:
        adapter.cancel("2")
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_list_jobs_filters_by_prefix():
    routes = {
        ("GET", "/slurm/v0.0.39/jobs"): (200, {
            "jobs": [
                {"job_id": 1, "name": "sgl-d-1", "job_state": "PENDING"},
                {"job_id": 2, "name": "other", "job_state": "RUNNING"},
                {"job_id": 3, "name": "sgl-d-2", "job_state": "COMPLETED"},
            ]
        })
    }
    c = _client_with_routes(routes)
    adapter = SlurmRestAdapter("http://slurm")
    adapter.client = c

    jobs = adapter.list_jobs("sgl-d-")
    ids = [j.job_id for j in jobs]
    assert ids == ["1", "3"]

