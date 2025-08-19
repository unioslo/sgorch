from sgorch.policy.failure import NodeBlacklist, FailureTracker, RestartPolicy, CircuitBreaker


def test_node_blacklist_add_expire_clear(freeze_time):
    nb = NodeBlacklist(cooldown_seconds=10)
    assert nb.is_blacklisted("n1") is False
    nb.blacklist_node("n1", reason="r")
    assert nb.is_blacklisted("n1") is True
    freeze_time.advance(11)
    assert nb.is_blacklisted("n1") is False
    nb.blacklist_node("n2")
    nb.clear_blacklist("n2")
    assert nb.is_blacklisted("n2") is False


def test_failure_tracker_blacklists_after_threshold():
    nb = NodeBlacklist(cooldown_seconds=60)
    ft = FailureTracker(nb, max_failures_per_node=2, failure_window_seconds=300)
    ft.record_failure("oops", node="n1")
    assert nb.is_blacklisted("n1") is False
    ft.record_failure("oops", node="n1")
    assert nb.is_blacklisted("n1") is True


def test_restart_policy_backoff_and_reset(no_sleep):
    rp = RestartPolicy(restart_backoff_seconds=5, max_restart_attempts=3)
    key = "w1"
    # initial should restart and provide delay values
    assert rp.should_restart(key) is True
    d1 = rp.get_restart_delay(key)
    assert d1 is not None
    assert rp.wait_for_restart(key) is True
    # subsequent calls until exhausted
    assert rp.should_restart(key) is True
    rp.get_restart_delay(key)
    rp.wait_for_restart(key)
    # after max attempts, should_retry returns False and next_delay None
    assert rp.should_restart(key) is False
    assert rp.get_restart_delay(key) is None
    # reset clears count
    rp.reset_restart_count(key)
    assert rp.should_restart(key) is True


def test_circuit_breaker_transitions_closed_open_halfopen(freeze_time):
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=10, half_open_max_calls=2)
    # closed allows
    assert cb.call_allowed() is True
    cb.record_failure()
    assert cb.get_state() == "closed"
    cb.record_failure()
    assert cb.get_state() == "open"
    assert cb.call_allowed() is False
    # after timeout -> half-open
    freeze_time.advance(11)
    assert cb.call_allowed() is True
    assert cb.get_state() == "half-open"
    # successes increment and then close
    cb.record_success()
    assert cb.call_allowed() is True
    cb.record_success()
    assert cb.get_state() == "closed"
    # failure in half-open reopens
    cb.force_open()
    freeze_time.advance(11)
    assert cb.call_allowed() is True
    assert cb.get_state() == "half-open"
    cb.record_failure()
    assert cb.get_state() == "open"

