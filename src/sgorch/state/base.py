from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


@dataclass
class SerializableWorker:
    """Serializable snapshot of a worker's state."""
    job_id: str
    instance_uuid: str
    # Stable replica slot index (for staggered time limits). Optional for
    # backward compatibility with existing state files.
    instance_idx: int | None = None
    node: Optional[str] = None
    remote_port: Optional[int] = None
    advertise_port: Optional[int] = None
    worker_url: Optional[str] = None
    advertised_url: Optional[str] = None
    health_status: str = "unknown"
    last_seen: Optional[float] = None
    submitted_at: float = 0.0


@dataclass
class DeploymentSnapshot:
    """Serializable snapshot for a deployment."""
    name: str
    workers: List[SerializableWorker]
    allocated_ports: List[int]

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "workers": [asdict(w) for w in self.workers],
            "allocated_ports": list(self.allocated_ports),
        }

    @staticmethod
    def from_dict(data: Dict) -> "DeploymentSnapshot":
        # Allow older snapshots without instance_idx
        workers: list[SerializableWorker] = []
        for w in data.get("workers", []):
            if "instance_idx" not in w:
                w = {**w, "instance_idx": None}
            workers.append(SerializableWorker(**w))
        allocated_ports = data.get("allocated_ports", [])
        return DeploymentSnapshot(
            name=data["name"],
            workers=workers,
            allocated_ports=allocated_ports,
        )


class StateStore(ABC):
    """Abstract state store interface for persistence backends (file, redis, etc.)."""

    @abstractmethod
    def load_deployment(self, name: str) -> Optional[DeploymentSnapshot]:
        pass

    @abstractmethod
    def save_deployment(self, snapshot: DeploymentSnapshot) -> None:
        pass

    @abstractmethod
    def delete_deployment(self, name: str) -> None:
        pass
