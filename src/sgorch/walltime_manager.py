"""Walltime management for proactive worker replacement."""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

from .logging_setup import get_logger


logger = get_logger(__name__)


@dataclass
class WalltimeInfo:
    """Information about a job's walltime."""
    job_id: str
    time_limit_seconds: int
    submitted_at: float
    estimated_end_time: float
    time_remaining_seconds: float
    
    def is_approaching_walltime(self, predrain_seconds: int) -> bool:
        """Check if job is approaching its walltime limit."""
        return self.time_remaining_seconds <= predrain_seconds
    
    @property
    def minutes_remaining(self) -> float:
        """Get remaining time in minutes."""
        return self.time_remaining_seconds / 60.0
    
    @property
    def percent_complete(self) -> float:
        """Get percentage of walltime consumed."""
        elapsed = time.time() - self.submitted_at
        return min(100.0, (elapsed / self.time_limit_seconds) * 100.0)


def parse_slurm_time_limit(time_str: str) -> int:
    """Parse SLURM time limit string to seconds.
    
    Supports formats:
    - "HH:MM:SS" 
    - "MM:SS"
    - "HH:MM"
    - "DD-HH:MM:SS" (days-hours:minutes:seconds)
    - "INFINITE"
    """
    if time_str.upper() in ("INFINITE", "UNLIMITED"):
        return 365 * 24 * 3600  # 1 year as "infinite"
    
    # Handle days format: "DD-HH:MM:SS"
    if '-' in time_str:
        days_part, time_part = time_str.split('-', 1)
        days = int(days_part)
        time_str = time_part
    else:
        days = 0
    
    # Parse time components
    time_parts = time_str.split(':')
    
    if len(time_parts) == 3:  # HH:MM:SS
        try:
            hours, minutes, seconds = map(int, time_parts)
        except ValueError:
            raise ValueError(f"Invalid time format: {time_str}")
    elif len(time_parts) == 2:  # MM:SS or HH:MM
        try:
            # For two parts, SLURM typically uses MM:SS for short times, HH:MM for longer times
            # We'll assume HH:MM format for values that make sense as hours
            first_part = int(time_parts[0])
            second_part = int(time_parts[1])
            
            if first_part <= 23 and second_part <= 59:  # Could be HH:MM
                hours, minutes, seconds = first_part, second_part, 0
            else:  # Assume MM:SS
                hours, minutes, seconds = 0, first_part, second_part
        except ValueError:
            raise ValueError(f"Invalid time format: {time_str}")
    else:
        raise ValueError(f"Invalid time format: {time_str}")
    
    # Validate time components
    if hours < 0 or minutes < 0 or seconds < 0:
        raise ValueError(f"Invalid time format: {time_str}")
    # For MM:SS format, minutes can be > 60, but seconds must be < 60
    # For HH:MM:SS or HH:MM format, both minutes and seconds must be < 60
    if len(time_parts) == 3:  # HH:MM:SS format
        if minutes >= 60 or seconds >= 60:
            raise ValueError(f"Invalid time format: {time_str}")
    elif len(time_parts) == 2:
        # If we interpreted it as HH:MM (hours <= 23), then minutes must be < 60
        if first_part <= 23 and second_part >= 60:
            raise ValueError(f"Invalid time format: {time_str}")
        # If we interpreted it as MM:SS (first_part > 23), then seconds must be < 60
        elif first_part > 23 and second_part >= 60:
            raise ValueError(f"Invalid time format: {time_str}")
    
    total_seconds = (
        (days * 24 * 3600) + 
        (hours * 3600) + 
        (minutes * 60) + 
        seconds
    )
    
    return total_seconds


class WalltimeManager:
    """Manages walltime tracking and proactive replacement scheduling."""
    
    def __init__(self, deployment_name: str, predrain_seconds: int = 180):
        self.deployment_name = deployment_name
        self.predrain_seconds = predrain_seconds
        self.logger = logger.bind(deployment=deployment_name)
        
        # Track walltime info for workers
        self.walltime_info: Dict[str, WalltimeInfo] = {}  # job_id -> WalltimeInfo
        
    def register_worker(self, job_id: str, time_limit: str, submitted_at: float) -> None:
        """Register a worker's walltime information."""
        try:
            time_limit_seconds = parse_slurm_time_limit(time_limit)
            estimated_end_time = submitted_at + time_limit_seconds
            time_remaining = max(0, estimated_end_time - time.time())
            
            walltime_info = WalltimeInfo(
                job_id=job_id,
                time_limit_seconds=time_limit_seconds,
                submitted_at=submitted_at,
                estimated_end_time=estimated_end_time,
                time_remaining_seconds=time_remaining
            )
            
            self.walltime_info[job_id] = walltime_info
            
            self.logger.debug(
                f"Registered walltime for {job_id}: {time_limit} "
                f"({walltime_info.minutes_remaining:.1f}min remaining)"
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to parse time limit '{time_limit}' for job {job_id}: {e}")
    
    def unregister_worker(self, job_id: str) -> None:
        """Remove walltime tracking for a worker."""
        self.walltime_info.pop(job_id, None)
    
    def update_remaining_times(self) -> None:
        """Update remaining time calculations for all workers."""
        current_time = time.time()
        
        for walltime_info in self.walltime_info.values():
            walltime_info.time_remaining_seconds = max(
                0, 
                walltime_info.estimated_end_time - current_time
            )
    
    def get_workers_approaching_walltime(self) -> List[WalltimeInfo]:
        """Get workers that are approaching their walltime limit."""
        self.update_remaining_times()
        
        approaching = []
        for walltime_info in self.walltime_info.values():
            if walltime_info.is_approaching_walltime(self.predrain_seconds):
                approaching.append(walltime_info)
        
        # Sort by most urgent first (least time remaining)
        approaching.sort(key=lambda w: w.time_remaining_seconds)
        
        return approaching
    
    def get_walltime_statistics(self) -> Dict[str, float]:
        """Get walltime statistics for monitoring."""
        if not self.walltime_info:
            return {}
        
        self.update_remaining_times()
        
        remaining_times = [w.time_remaining_seconds for w in self.walltime_info.values()]
        percent_complete = [w.percent_complete for w in self.walltime_info.values()]
        
        return {
            'workers_tracked': len(self.walltime_info),
            'min_time_remaining_minutes': min(remaining_times) / 60.0 if remaining_times else 0,
            'max_time_remaining_minutes': max(remaining_times) / 60.0 if remaining_times else 0,
            'avg_time_remaining_minutes': sum(remaining_times) / len(remaining_times) / 60.0,
            'min_percent_complete': min(percent_complete) if percent_complete else 0,
            'max_percent_complete': max(percent_complete) if percent_complete else 0,
            'avg_percent_complete': sum(percent_complete) / len(percent_complete) if percent_complete else 0,
            'workers_approaching_walltime': len([w for w in self.walltime_info.values() 
                                              if w.is_approaching_walltime(self.predrain_seconds)])
        }
    
    def should_start_proactive_replacement(self, job_id: str) -> bool:
        """Check if we should start proactive replacement for a worker."""
        walltime_info = self.walltime_info.get(job_id)
        if not walltime_info:
            return False
        
        # Update remaining time
        current_time = time.time()
        walltime_info.time_remaining_seconds = max(
            0,
            walltime_info.estimated_end_time - current_time
        )
        
        return walltime_info.is_approaching_walltime(self.predrain_seconds)
    
    def get_replacement_urgency(self, job_id: str) -> str:
        """Get replacement urgency level for a job."""
        walltime_info = self.walltime_info.get(job_id)
        if not walltime_info:
            return "unknown"
        
        self.update_remaining_times()
        
        remaining_minutes = walltime_info.minutes_remaining
        
        if remaining_minutes <= 5:
            return "critical"
        elif remaining_minutes <= 15:
            return "high"
        elif remaining_minutes <= 30:
            return "medium"
        else:
            return "low"
    
    def estimate_replacement_window(self, job_id: str) -> Optional[Dict[str, float]]:
        """Estimate the time window available for replacement."""
        walltime_info = self.walltime_info.get(job_id)
        if not walltime_info:
            return None
        
        self.update_remaining_times()
        
        # Estimate time needed for replacement
        typical_job_start_time = 5 * 60      # 5 minutes to start new job  
        typical_model_load_time = 10 * 60    # 10 minutes to load model
        drain_grace_period = 2 * 60          # 2 minutes grace period
        
        total_replacement_time = typical_job_start_time + typical_model_load_time + drain_grace_period
        
        return {
            'time_remaining_seconds': walltime_info.time_remaining_seconds,
            'estimated_replacement_time_seconds': total_replacement_time,
            'time_buffer_seconds': walltime_info.time_remaining_seconds - total_replacement_time,
            'can_complete_replacement': walltime_info.time_remaining_seconds > total_replacement_time,
            'urgency_level': self.get_replacement_urgency(job_id)
        }
