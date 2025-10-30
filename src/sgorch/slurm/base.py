from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Literal

from .errors import SlurmUnavailableError

JobState = Literal[
    "PENDING", "RUNNING", "COMPLETED", "FAILED", 
    "CANCELLED", "TIMEOUT", "NODE_FAIL", "UNKNOWN"
]


@dataclass
class SubmitSpec:
    """Specification for submitting a SLURM job."""
    name: str
    account: str
    reservation: Optional[str]
    partition: str
    qos: Optional[str]
    gres: str
    constraint: Optional[str]
    time_limit: str          # "HH:MM:SS"
    cpus_per_task: int
    mem: str                 # "64G"
    env: dict[str, str]
    stdout: str
    stderr: str
    script: str              # full bash script text


@dataclass
class JobInfo:
    """Information about a SLURM job."""
    job_id: str
    state: JobState
    node: Optional[str]      # "cn123" when RUNNING
    time_left_s: Optional[int]


class ISlurm(ABC):
    """Interface for SLURM operations."""
    
    @abstractmethod
    def submit(self, spec: SubmitSpec) -> str:
        """Submit a job and return job ID."""
        pass
    
    @abstractmethod
    def status(self, job_id: str) -> JobInfo:
        """Get status of a job."""
        pass
    
    @abstractmethod
    def cancel(self, job_id: str) -> None:
        """Cancel a job."""
        pass
    
    @abstractmethod
    def list_jobs(self, name_prefix: str) -> list[JobInfo]:
        """List jobs with names starting with the given prefix."""
        pass


__all__ = [
    "JobState",
    "SubmitSpec",
    "JobInfo",
    "ISlurm",
    "SlurmUnavailableError",
]
