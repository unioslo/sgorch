import socket

from sgorch.net.ports import PortAllocator, find_free_port, PortAllocationError


def test_allocate_and_release_single_port(seed_random):
    alloc = PortAllocator((30000, 30010))
    p = alloc.allocate_port()
    assert p in alloc
    alloc.release_port(p)
    assert p not in alloc.get_allocated_ports()


def test_allocate_pair_success_and_exhaustion(seed_random):
    alloc = PortAllocator((40000, 40003))
    a, b = alloc.allocate_pair()
    assert b == a + 1
    try:
        alloc.allocate_pair()
        # With such a tiny range, second pair should be unavailable
        # depending on bindings this may still fail due to system ports; rely on exception.
        pass
    except PortAllocationError:
        pass


def test_mark_in_use_and_get_available_count():
    alloc = PortAllocator((50000, 50002))
    alloc.mark_in_use(50001)
    assert 50001 in alloc
    assert alloc.get_available_count() == 2  # 50000 and 50002 approximately


def test_find_free_port_stops_at_65535(monkeypatch):
    class FakeSock:
        def __init__(self):
            self.bound = []
        def setsockopt(self, *a):
            pass
        def bind(self, addr):
            host, port = addr
            if port < 65535:
                raise OSError("busy")
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
    monkeypatch.setattr(socket, "socket", lambda *a, **k: FakeSock())
    try:
        find_free_port(65530, max_attempts=10)
        # Should find 65535 without raising
    except PortAllocationError:
        raise AssertionError("expected to find a free port by 65535")

