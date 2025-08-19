import os
import time
import types
import subprocess
from typing import Callable, Dict, Optional

import pytest


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Make time.sleep a no-op for deterministic tests."""
    monkeypatch.setattr(time, "sleep", lambda *_args, **_kwargs: None)


@pytest.fixture()
def freeze_time(monkeypatch):
    """Freeze time.time() and provide an advance() helper."""
    t = {"now": 1735689600.0}  # 2025-01-01T00:00:00Z

    def now():
        return t["now"]

    def advance(seconds: float):
        t["now"] += float(seconds)

    monkeypatch.setattr(time, "time", now)
    return types.SimpleNamespace(now=now, advance=advance)


@pytest.fixture()
def env(monkeypatch):
    """Helper to set/clear environment variables."""
    def _setter(mapping: Optional[Dict[str, str]] = None, clear: Optional[list[str]] = None):
        if mapping:
            for k, v in mapping.items():
                monkeypatch.setenv(k, v)
        if clear:
            for k in clear:
                monkeypatch.delenv(k, raising=False)
    return _setter


@pytest.fixture()
def seed_random(monkeypatch):
    import random
    monkeypatch.setattr(random, "choice", lambda seq: seq[0])


class _FakeProc:
    def __init__(self, rc: Optional[int] = None):
        self._rc = rc
        self._killed = False
        self._signals = []

    def poll(self):
        return self._rc

    def wait(self, timeout: Optional[float] = None):
        return self._rc or 0

    def communicate(self):
        return (b"", b"")

    def send_signal(self, sig):
        self._signals.append(sig)

    def kill(self):
        self._killed = True
        self._rc = -9


@pytest.fixture()
def fake_popen(monkeypatch):
    """Patch subprocess.Popen with a lightweight fake process."""
    instances = []

    def factory(rc: Optional[int] = None):
        proc = _FakeProc(rc)
        instances.append(proc)
        return proc

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: factory(rc=None))
    return types.SimpleNamespace(instances=instances, factory=factory)


@pytest.fixture()
def fake_subprocess_run(monkeypatch):
    """Patch subprocess.run to return scripted outputs based on argv[0]."""
    scripts: Dict[str, Callable[[list[str]], subprocess.CompletedProcess]] = {}

    def register(cmd0: str, func: Callable[[list[str]], subprocess.CompletedProcess]):
        scripts[cmd0] = func

    def _run(cmd, capture_output=True, text=True, timeout=None):
        key = cmd[0] if cmd else ""
        if key in scripts:
            return scripts[key](cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _run)
    return register

