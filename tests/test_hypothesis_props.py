from hypothesis import given, strategies as st

from sgorch.slurm.parse import _parse_time_remaining
from sgorch.util.backoff import exponential_backoff, linear_backoff


@given(
    m=st.integers(min_value=0, max_value=59),
    s=st.integers(min_value=0, max_value=59),
)
def test_parse_time_mmss(m, s):
    t = f"{m:02d}:{s:02d}"
    assert _parse_time_remaining(t) == m * 60 + s


@given(
    h=st.integers(min_value=0, max_value=99),
    m=st.integers(min_value=0, max_value=59),
    s=st.integers(min_value=0, max_value=59),
)
def test_parse_time_hhmmss(h, m, s):
    t = f"{h:02d}:{m:02d}:{s:02d}"
    assert _parse_time_remaining(t) == h * 3600 + m * 60 + s


@given(
    d=st.integers(min_value=0, max_value=30),
    h=st.integers(min_value=0, max_value=23),
    m=st.integers(min_value=0, max_value=59),
    s=st.integers(min_value=0, max_value=59),
)
def test_parse_time_dd_hhmmss(d, h, m, s):
    t = f"{d}-{h:02d}:{m:02d}:{s:02d}"
    assert _parse_time_remaining(t) == d * 86400 + h * 3600 + m * 60 + s


def test_exponential_backoff_properties(monkeypatch):
    # make jitter deterministic
    monkeypatch.setattr("sgorch.util.backoff.random.uniform", lambda a, b: 0)
    delays = list(exponential_backoff(base_delay=1.0, max_delay=10.0, multiplier=2.0, jitter=True, max_attempts=5))
    assert delays[0] == 0.0
    assert all(d >= 0 for d in delays)
    assert all(delays[i] <= delays[i+1] for i in range(len(delays)-1))
    assert max(delays) <= 10.0


def test_linear_backoff_properties(monkeypatch):
    monkeypatch.setattr("sgorch.util.backoff.random.uniform", lambda a, b: 0)
    delays = list(linear_backoff(initial_delay=1.0, increment=1.0, max_delay=5.0, jitter=True, max_attempts=6))
    assert delays[0] == 0.0
    assert all(d >= 0 for d in delays)
    assert all(delays[i] <= delays[i+1] for i in range(len(delays)-1))
    assert max(delays) <= 5.0

