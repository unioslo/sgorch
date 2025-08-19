import socket
import subprocess

from sgorch.net.hostaddr import (
    is_valid_ip, is_ipv4, is_ipv6,
    resolve_hostname_all_ips, test_tcp_connection, get_interface_ip,
)


def test_is_valid_ip_ipv4_ipv6():
    assert is_valid_ip("127.0.0.1")
    assert is_ipv4("127.0.0.1")
    assert not is_ipv6("127.0.0.1")
    assert is_valid_ip("::1")
    assert is_ipv6("::1")
    assert not is_ipv4("::1")
    assert not is_valid_ip("not-an-ip")


def test_resolve_hostname_all_ips_handles_duplicates_and_errors(monkeypatch):
    def fake_getaddrinfo(host, port):
        if host == "bad":
            raise socket.gaierror()
        return [
            (0,0,0,"", ("1.2.3.4",0)),
            (0,0,0,"", ("1.2.3.4",0)),
            (0,0,0,"", ("5.6.7.8",0)),
        ]
    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    assert resolve_hostname_all_ips("bad") == []
    assert resolve_hostname_all_ips("ok") == ["1.2.3.4", "5.6.7.8"]


def test_test_tcp_connection_true_false_paths(monkeypatch):
    class FakeSock:
        def __init__(self):
            self.timeout = None
        def settimeout(self, t):
            self.timeout = t
        def connect_ex(self, addr):
            host, port = addr
            return 0 if port == 1234 else 1
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
    monkeypatch.setattr(socket, "socket", lambda *a, **k: FakeSock())
    assert test_tcp_connection("h", 1234)
    assert not test_tcp_connection("h", 1111)


def test_get_interface_ip_parses_ip_output(monkeypatch):
    out = """
    2: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500
        inet 192.168.1.42/24 brd 192.168.1.255 scope global dynamic eth0
           valid_lft 86393sec preferred_lft 86393sec
        inet 127.0.0.1/8 scope host lo
    """
    def fake_run(cmd, capture_output=True, text=True, timeout=None):
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    assert get_interface_ip("eth0") == "192.168.1.42"

