import httpx
import pytest

from sgorch.router.client import RouterClient, RouterError
from sgorch.config import RouterConfig
from sgorch.slurm.rest import SlurmRestAdapter


def test_router_client_errors_raise(monkeypatch):
    def handler(request: httpx.Request):
        return httpx.Response(500, json={"err": "x"})
    transport = httpx.MockTransport(handler)
    c = RouterClient(RouterConfig(base_url="http://router"))
    c.client = httpx.Client(transport=transport, base_url="http://router")
    with pytest.raises(RouterError):
        c.list()
    with pytest.raises(RouterError):
        c.add("http://w")
    with pytest.raises(RouterError):
        c.remove("http://w")


def test_slurm_rest_malformed_json_and_http_errors():
    # malformed JSON on jobs list
    def handler(request: httpx.Request):
        if request.method == "GET" and request.url.path == "/slurm/v0.0.39/jobs":
            return httpx.Response(200, text="not json")
        return httpx.Response(500, json={})
    transport = httpx.MockTransport(handler)
    a = SlurmRestAdapter("http://slurm")
    a.client = httpx.Client(transport=transport, base_url="http://slurm")
    # list_jobs swallows errors and returns []
    assert a.list_jobs("p") == []

