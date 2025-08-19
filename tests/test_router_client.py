import httpx
import os

from sgorch.router.client import RouterClient
from sgorch.config import RouterConfig, EndpointsConfig, AuthConfig


def _router_client_with_routes(routes):
    def handler(request: httpx.Request):
        key = (request.method, request.url.path)
        if key in routes:
            status, data = routes[key]
            return httpx.Response(status, json=data)
        return httpx.Response(404, json={})

    transport = httpx.MockTransport(handler)
    c = RouterClient(RouterConfig(base_url="http://router"))
    c.client = httpx.Client(transport=transport, base_url="http://router")
    return c


def test_list_accepts_list_or_wrapped_formats():
    # list response
    c = _router_client_with_routes({("GET", "/workers/list"): (200, ["a", "b"])})
    assert c.list() == {"a", "b"}
    # wrapped response
    c = _router_client_with_routes({("GET", "/workers/list"): (200, {"workers": ["x"]})})
    assert c.list() == {"x"}
    # alternative key
    c = _router_client_with_routes({("GET", "/workers/list"): (200, {"urls": ["u1", "u2"]})})
    assert c.list() == {"u1", "u2"}


def test_add_and_remove_success_paths():
    routes = {
        ("POST", "/workers/add"): (200, {}),
        ("POST", "/workers/remove"): (404, {}),  # should be ignored gracefully
    }
    c = _router_client_with_routes(routes)
    c.add("http://w")
    c.remove("http://w")


def test_auth_header_injected_from_env_missing_warns(monkeypatch, caplog):
    cfg = RouterConfig(
        base_url="http://router",
        auth=AuthConfig(type="header", header_name="X-Auth", header_value_env="TOK")
    )
    caplog.set_level("WARNING")
    # ensure env missing
    monkeypatch.delenv("TOK", raising=False)
    client = RouterClient(cfg)
    # creates client; should log warning
    assert any("Auth token environment variable TOK not set" in r.message for r in caplog.records)

