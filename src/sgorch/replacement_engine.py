"""Gradual worker replacement engine with safety guarantees."""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from .logging_setup import get_logger
from .config import DeploymentConfig
from .config_hash import ConfigSnapshot


logger = get_logger(__name__)


class ReplacementStrategy(Enum):
    """Strategy for worker replacement."""
    ONE_BY_ONE = "one_by_one"           # Replace one worker at a time
    PARALLEL_SAFE = "parallel_safe"     # Replace multiple, but maintain min replicas
    ROLLING = "rolling"                 # Blue-green style replacement


@dataclass
class ReplacementPlan:
    """Plan for replacing workers."""
    workers_to_replace: List[str]  # job_ids
    strategy: ReplacementStrategy
    max_concurrent: int
    min_healthy_replicas: int
    replacement_reason: str
    estimated_duration_minutes: int
    
    def is_safe(self, current_healthy: int) -> bool:
        """Check if replacement plan is safe given current healthy worker count."""
        if self.strategy == ReplacementStrategy.ONE_BY_ONE:
            return current_healthy >= self.min_healthy_replicas + 1
        elif self.strategy == ReplacementStrategy.PARALLEL_SAFE:
            return current_healthy >= self.min_healthy_replicas + self.max_concurrent
        else:  # ROLLING
            return current_healthy >= self.min_healthy_replicas


@dataclass
class ReplacementTask:
    """Individual worker replacement task."""
    old_worker_job_id: str
    replacement_job_id: Optional[str] = None
    started_at: Optional[float] = None
    new_worker_ready_at: Optional[float] = None
    old_worker_drained_at: Optional[float] = None
    completed_at: Optional[float] = None
    failed_at: Optional[float] = None
    failure_reason: Optional[str] = None
    
    @property
    def is_in_progress(self) -> bool:
        return self.started_at is not None and self.completed_at is None and self.failed_at is None
    
    @property 
    def is_completed(self) -> bool:
        return self.completed_at is not None
    
    @property
    def is_failed(self) -> bool:
        return self.failed_at is not None
    
    @property
    def duration_seconds(self) -> Optional[float]:
        if self.started_at and (self.completed_at or self.failed_at):
            end_time = self.completed_at or self.failed_at
            return end_time - self.started_at
        return None


class ReplacementEngine:
    """Manages gradual worker replacement with safety guarantees."""
    
    def __init__(self, deployment_name: str):
        self.deployment_name = deployment_name
        self.logger = logger.bind(deployment=deployment_name)
        self.active_tasks: Dict[str, ReplacementTask] = {}  # old_job_id -> task
        self.completed_tasks: List[ReplacementTask] = []
        
        # Safety settings
        self.max_replacement_time_minutes = 30  # Timeout for individual replacements
        self.drain_grace_period_seconds = 30    # Time to wait before canceling old job
        self.new_worker_ready_timeout_minutes = 10  # Max time to wait for new worker to be healthy
    
    def plan_replacement(
        self, 
        workers_needing_replacement: List[str],
        current_healthy_count: int,
        desired_replicas: int,
        replacement_reason: str
    ) -> Optional[ReplacementPlan]:
        """Create a replacement plan for the given workers."""
        
        if not workers_needing_replacement:
            return None
        
        # Choose strategy based on scale and safety requirements
        min_healthy = max(1, desired_replicas - 1)  # Always keep at least 1 healthy
        
        if len(workers_needing_replacement) == 1 and current_healthy_count > min_healthy:
            # Simple one-by-one replacement
            strategy = ReplacementStrategy.ONE_BY_ONE
            max_concurrent = 1
        elif current_healthy_count >= desired_replicas + 1:
            # We have extra healthy workers - can do parallel replacement
            extra_workers = current_healthy_count - desired_replicas
            max_concurrent = min(extra_workers, len(workers_needing_replacement))
            strategy = ReplacementStrategy.PARALLEL_SAFE
        else:
            # Limited capacity - be conservative
            strategy = ReplacementStrategy.ONE_BY_ONE
            max_concurrent = 1
        
        # Estimate duration (rough heuristic)
        avg_replacement_time = 10  # minutes per worker
        if strategy == ReplacementStrategy.ONE_BY_ONE:
            estimated_duration = len(workers_needing_replacement) * avg_replacement_time
        else:
            estimated_duration = (len(workers_needing_replacement) / max_concurrent) * avg_replacement_time
        
        plan = ReplacementPlan(
            workers_to_replace=workers_needing_replacement.copy(),
            strategy=strategy,
            max_concurrent=max_concurrent,
            min_healthy_replicas=min_healthy,
            replacement_reason=replacement_reason,
            estimated_duration_minutes=int(estimated_duration)
        )
        
        if not plan.is_safe(current_healthy_count):
            self.logger.warning(
                f"Replacement plan not safe: need {plan.min_healthy_replicas} healthy, "
                f"have {current_healthy_count}, plan requires {plan.max_concurrent} concurrent"
            )
            return None
            
        return plan
    
    def start_replacement_task(self, old_worker_job_id: str) -> ReplacementTask:
        """Start replacing a specific worker."""
        if old_worker_job_id in self.active_tasks:
            return self.active_tasks[old_worker_job_id]
        
        task = ReplacementTask(
            old_worker_job_id=old_worker_job_id,
            started_at=time.time()
        )
        
        self.active_tasks[old_worker_job_id] = task
        
        self.logger.info(f"Started replacement task for worker {old_worker_job_id}")
        return task
    
    def update_task_new_worker_started(self, old_worker_job_id: str, new_worker_job_id: str) -> None:
        """Update task when new worker has been started."""
        task = self.active_tasks.get(old_worker_job_id)
        if not task:
            self.logger.warning(f"No active replacement task found for {old_worker_job_id}")
            return
            
        task.replacement_job_id = new_worker_job_id
        self.logger.info(f"New worker {new_worker_job_id} started for replacement of {old_worker_job_id}")
    
    def update_task_new_worker_ready(self, old_worker_job_id: str) -> None:
        """Update task when new worker is healthy and ready."""
        task = self.active_tasks.get(old_worker_job_id)
        if not task:
            return
            
        task.new_worker_ready_at = time.time()
        self.logger.info(f"New worker ready for replacement of {old_worker_job_id}")
    
    def update_task_old_worker_drained(self, old_worker_job_id: str) -> None:
        """Update task when old worker has been drained."""
        task = self.active_tasks.get(old_worker_job_id)
        if not task:
            return
            
        task.old_worker_drained_at = time.time()
    
    def complete_replacement_task(self, old_worker_job_id: str) -> None:
        """Mark replacement task as completed."""
        task = self.active_tasks.pop(old_worker_job_id, None)
        if not task:
            return
            
        task.completed_at = time.time()
        self.completed_tasks.append(task)
        
        duration = task.duration_seconds
        self.logger.info(
            f"Completed replacement of {old_worker_job_id} "
            f"(duration: {duration:.1f}s)" if duration else ""
        )
    
    def fail_replacement_task(self, old_worker_job_id: str, reason: str) -> None:
        """Mark replacement task as failed."""
        task = self.active_tasks.pop(old_worker_job_id, None)
        if not task:
            return
            
        task.failed_at = time.time()
        task.failure_reason = reason
        self.completed_tasks.append(task)
        
        self.logger.error(f"Failed replacement of {old_worker_job_id}: {reason}")
    
    def get_active_replacement_count(self) -> int:
        """Get number of active replacement tasks."""
        return len(self.active_tasks)
    
    def get_timed_out_tasks(self) -> List[ReplacementTask]:
        """Get tasks that have been running too long."""
        timeout_seconds = self.max_replacement_time_minutes * 60
        current_time = time.time()
        
        timed_out = []
        for task in self.active_tasks.values():
            if task.started_at and (current_time - task.started_at) > timeout_seconds:
                timed_out.append(task)
        
        return timed_out
    
    def cleanup_timed_out_tasks(self) -> None:
        """Clean up tasks that have timed out."""
        for task in self.get_timed_out_tasks():
            self.fail_replacement_task(
                task.old_worker_job_id, 
                f"timeout_after_{self.max_replacement_time_minutes}min"
            )
    
    def can_start_more_replacements(self, max_concurrent: int) -> bool:
        """Check if we can start more replacement tasks."""
        return len(self.active_tasks) < max_concurrent
    
    def get_replacement_stats(self) -> Dict[str, int]:
        """Get replacement statistics."""
        completed_successful = len([t for t in self.completed_tasks if t.is_completed])
        completed_failed = len([t for t in self.completed_tasks if t.is_failed])
        
        return {
            'active': len(self.active_tasks),
            'completed_successful': completed_successful,
            'completed_failed': completed_failed,
            'total_completed': len(self.completed_tasks)
        }
