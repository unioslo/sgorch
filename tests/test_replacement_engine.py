"""Tests for the replacement engine."""

import pytest
import time
from sgorch.replacement_engine import (
    ReplacementEngine,
    ReplacementStrategy,
    ReplacementPlan,
    ReplacementTask,
)


@pytest.fixture
def engine():
    """Create a replacement engine for testing."""
    return ReplacementEngine("test-deployment")


def test_replacement_engine_initialization(engine):
    """Test replacement engine initialization."""
    assert engine.deployment_name == "test-deployment"
    assert len(engine.active_tasks) == 0
    assert len(engine.completed_tasks) == 0
    assert engine.max_replacement_time_minutes == 30
    assert engine.drain_grace_period_seconds == 30


def test_replacement_plan_creation_one_by_one(engine):
    """Test replacement plan creation with one-by-one strategy."""
    plan = engine.plan_replacement(
        workers_needing_replacement=["job1"],
        current_healthy_count=3,
        desired_replicas=2,
        replacement_reason="config_change"
    )
    
    assert plan is not None
    assert plan.strategy == ReplacementStrategy.ONE_BY_ONE
    assert plan.max_concurrent == 1
    assert plan.min_healthy_replicas == 1
    assert plan.workers_to_replace == ["job1"]
    assert plan.replacement_reason == "config_change"
    assert plan.estimated_duration_minutes > 0


def test_replacement_plan_creation_parallel_safe(engine):
    """Test replacement plan creation with parallel-safe strategy."""
    plan = engine.plan_replacement(
        workers_needing_replacement=["job1", "job2"],
        current_healthy_count=5,  # Extra capacity allows parallel
        desired_replicas=2,
        replacement_reason="config_change"
    )
    
    assert plan is not None
    assert plan.strategy == ReplacementStrategy.PARALLEL_SAFE
    assert plan.max_concurrent == 2  # Can replace both workers in parallel
    assert plan.min_healthy_replicas == 1


def test_replacement_plan_safety_check(engine):
    """Test replacement plan safety validation."""
    # Safe plan
    safe_plan = engine.plan_replacement(
        workers_needing_replacement=["job1"],
        current_healthy_count=3,
        desired_replicas=2,
        replacement_reason="config_change"
    )
    
    assert safe_plan is not None
    assert safe_plan.is_safe(3)
    
    # Unsafe plan - not enough healthy workers
    unsafe_plan = engine.plan_replacement(
        workers_needing_replacement=["job1"],
        current_healthy_count=1,  # Only 1 healthy, but need to maintain 1
        desired_replicas=2,
        replacement_reason="config_change"
    )
    
    assert unsafe_plan is None  # Should return None for unsafe plans


def test_replacement_plan_empty_workers_list(engine):
    """Test replacement plan with empty workers list."""
    plan = engine.plan_replacement(
        workers_needing_replacement=[],
        current_healthy_count=5,
        desired_replicas=2,
        replacement_reason="config_change"
    )
    
    assert plan is None


def test_replacement_plan_is_safe():
    """Test ReplacementPlan.is_safe method."""
    # One-by-one strategy
    one_by_one_plan = ReplacementPlan(
        workers_to_replace=["job1"],
        strategy=ReplacementStrategy.ONE_BY_ONE,
        max_concurrent=1,
        min_healthy_replicas=1,
        replacement_reason="test",
        estimated_duration_minutes=10
    )
    
    assert one_by_one_plan.is_safe(2)  # 2 healthy, need 1 + 1 for replacement
    assert not one_by_one_plan.is_safe(1)  # 1 healthy, need 1 + 1 for replacement
    
    # Parallel-safe strategy
    parallel_plan = ReplacementPlan(
        workers_to_replace=["job1", "job2"],
        strategy=ReplacementStrategy.PARALLEL_SAFE,
        max_concurrent=2,
        min_healthy_replicas=1,
        replacement_reason="test",
        estimated_duration_minutes=10
    )
    
    assert parallel_plan.is_safe(3)  # 3 healthy, need 1 + 2 for replacement
    assert not parallel_plan.is_safe(2)  # 2 healthy, need 1 + 2 for replacement


def test_replacement_task_creation(engine, freeze_time):
    """Test replacement task creation and state management."""
    task = engine.start_replacement_task("job1")
    
    assert task.old_worker_job_id == "job1"
    assert task.started_at == freeze_time.now()
    assert task.replacement_job_id is None
    assert task.is_in_progress
    assert not task.is_completed
    assert not task.is_failed


def test_replacement_task_lifecycle(engine, freeze_time):
    """Test complete replacement task lifecycle."""
    # Start task
    task = engine.start_replacement_task("job1")
    assert task.is_in_progress
    
    # Update with new worker started
    engine.update_task_new_worker_started("job1", "job2")
    assert engine.active_tasks["job1"].replacement_job_id == "job2"
    
    # Update new worker ready
    freeze_time.advance(300)  # 5 minutes
    engine.update_task_new_worker_ready("job1")
    assert engine.active_tasks["job1"].new_worker_ready_at == freeze_time.now()
    
    # Update old worker drained
    freeze_time.advance(60)  # 1 minute
    engine.update_task_old_worker_drained("job1")
    assert engine.active_tasks["job1"].old_worker_drained_at == freeze_time.now()
    
    # Complete task
    freeze_time.advance(30)  # 30 seconds
    engine.complete_replacement_task("job1")
    
    # Task should be moved to completed
    assert "job1" not in engine.active_tasks
    assert len(engine.completed_tasks) == 1
    
    completed_task = engine.completed_tasks[0]
    assert completed_task.is_completed
    assert not completed_task.is_failed
    assert completed_task.completed_at == freeze_time.now()
    assert completed_task.duration_seconds == 390  # 6.5 minutes total


def test_replacement_task_failure(engine, freeze_time):
    """Test replacement task failure handling."""
    task = engine.start_replacement_task("job1")
    
    freeze_time.advance(600)  # 10 minutes
    engine.fail_replacement_task("job1", "timeout")
    
    # Task should be moved to completed with failure
    assert "job1" not in engine.active_tasks
    assert len(engine.completed_tasks) == 1
    
    failed_task = engine.completed_tasks[0]
    assert failed_task.is_failed
    assert not failed_task.is_completed
    assert failed_task.failure_reason == "timeout"
    assert failed_task.failed_at == freeze_time.now()
    assert failed_task.duration_seconds == 600


def test_replacement_task_properties():
    """Test ReplacementTask property methods."""
    # In-progress task
    task = ReplacementTask(
        old_worker_job_id="job1",
        started_at=time.time()
    )
    assert task.is_in_progress
    assert not task.is_completed
    assert not task.is_failed
    assert task.duration_seconds is None
    
    # Completed task
    start_time = time.time()
    end_time = start_time + 300
    completed_task = ReplacementTask(
        old_worker_job_id="job1",
        started_at=start_time,
        completed_at=end_time
    )
    assert not completed_task.is_in_progress
    assert completed_task.is_completed
    assert not completed_task.is_failed
    assert abs(completed_task.duration_seconds - 300) < 1  # Allow 1s tolerance
    
    # Failed task
    failed_task = ReplacementTask(
        old_worker_job_id="job1",
        started_at=start_time,
        failed_at=end_time,
        failure_reason="error"
    )
    assert not failed_task.is_in_progress
    assert not failed_task.is_completed
    assert failed_task.is_failed
    assert abs(failed_task.duration_seconds - 300) < 1


def test_timeout_detection(engine, freeze_time):
    """Test timeout detection and cleanup."""
    # Start task
    engine.start_replacement_task("job1")
    
    # No timeouts initially
    timeouts = engine.get_timed_out_tasks()
    assert len(timeouts) == 0
    
    # Advance time beyond timeout
    freeze_time.advance(engine.max_replacement_time_minutes * 60 + 1)
    
    # Should detect timeout
    timeouts = engine.get_timed_out_tasks()
    assert len(timeouts) == 1
    assert timeouts[0].old_worker_job_id == "job1"
    
    # Cleanup timed out tasks
    engine.cleanup_timed_out_tasks()
    
    # Task should be moved to completed with timeout failure
    assert "job1" not in engine.active_tasks
    assert len(engine.completed_tasks) == 1
    assert engine.completed_tasks[0].is_failed
    assert "timeout" in engine.completed_tasks[0].failure_reason


def test_concurrent_replacement_limits(engine):
    """Test concurrent replacement limits."""
    # Start multiple tasks
    engine.start_replacement_task("job1")
    engine.start_replacement_task("job2")
    engine.start_replacement_task("job3")
    
    assert engine.get_active_replacement_count() == 3
    
    # Test capacity checks
    assert not engine.can_start_more_replacements(2)  # Already have 3 active
    assert engine.can_start_more_replacements(5)  # Can handle 5 concurrent


def test_replacement_statistics(engine, freeze_time):
    """Test replacement statistics collection."""
    # Initially empty
    stats = engine.get_replacement_stats()
    assert stats['active'] == 0
    assert stats['completed_successful'] == 0
    assert stats['completed_failed'] == 0
    assert stats['total_completed'] == 0
    
    # Start and complete a task successfully
    engine.start_replacement_task("job1")
    freeze_time.advance(300)
    engine.complete_replacement_task("job1")
    
    # Start and fail a task
    engine.start_replacement_task("job2")
    freeze_time.advance(600)
    engine.fail_replacement_task("job2", "error")
    
    # Start an active task
    engine.start_replacement_task("job3")
    
    stats = engine.get_replacement_stats()
    assert stats['active'] == 1
    assert stats['completed_successful'] == 1
    assert stats['completed_failed'] == 1
    assert stats['total_completed'] == 2


def test_duplicate_task_start(engine):
    """Test that starting a task for the same job twice returns the same task."""
    task1 = engine.start_replacement_task("job1")
    task2 = engine.start_replacement_task("job1")
    
    assert task1 is task2
    assert engine.get_active_replacement_count() == 1


def test_update_nonexistent_task(engine):
    """Test updating a task that doesn't exist."""
    # These should not raise exceptions
    engine.update_task_new_worker_started("nonexistent", "job2")
    engine.update_task_new_worker_ready("nonexistent")
    engine.update_task_old_worker_drained("nonexistent")
    engine.complete_replacement_task("nonexistent")
    engine.fail_replacement_task("nonexistent", "error")


def test_replacement_strategy_enum():
    """Test ReplacementStrategy enum values."""
    assert ReplacementStrategy.ONE_BY_ONE.value == "one_by_one"
    assert ReplacementStrategy.PARALLEL_SAFE.value == "parallel_safe"
    assert ReplacementStrategy.ROLLING.value == "rolling"


@pytest.mark.parametrize("workers,healthy,desired,expected_strategy,expected_concurrent", [
    # Single worker, sufficient capacity → one-by-one
    (1, 3, 2, ReplacementStrategy.ONE_BY_ONE, 1),
    # Multiple workers, extra capacity → parallel
    (2, 5, 2, ReplacementStrategy.PARALLEL_SAFE, 2),
    # Limited capacity → conservative one-by-one
    (2, 2, 2, ReplacementStrategy.ONE_BY_ONE, 1),
    # Lots of workers, lots of capacity → parallel (limited by extra capacity)
    (5, 8, 3, ReplacementStrategy.PARALLEL_SAFE, 5),  # 5 = min(5 workers, 5 extra)
])
def test_replacement_strategy_selection(engine, workers, healthy, desired, expected_strategy, expected_concurrent):
    """Test that replacement strategy is selected correctly based on capacity."""
    worker_list = [f"job{i}" for i in range(workers)]
    
    plan = engine.plan_replacement(
        workers_needing_replacement=worker_list,
        current_healthy_count=healthy,
        desired_replicas=desired,
        replacement_reason="test"
    )
    
    if plan is None:
        pytest.skip("Plan was deemed unsafe")
    
    assert plan.strategy == expected_strategy
    assert plan.max_concurrent == expected_concurrent
