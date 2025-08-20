"""Configuration hashing for change detection and worker lifecycle management."""

import hashlib
import json
from typing import Dict, Any
from dataclasses import dataclass
from .config import DeploymentConfig


@dataclass
class ConfigSnapshot:
    """Snapshot of configuration relevant to worker lifecycle."""
    # SLURM job parameters
    gres: str
    cpus_per_task: int
    mem: str
    time_limit: str
    constraint: str | None
    account: str
    partition: str
    reservation: str | None
    env: Dict[str, str]
    
    # SGLang parameters
    model_path: str
    sglang_args: list[str]
    venv_path: str | None
    
    # Worker parameters that affect runtime
    health_config_hash: str
    connectivity_mode: str
    
    @classmethod
    def from_deployment_config(cls, config: DeploymentConfig) -> "ConfigSnapshot":
        """Create snapshot from deployment configuration."""
        # Hash health config separately since it's complex
        health_dict = {
            'path': config.health.path,
            'interval_s': config.health.interval_s,
            'timeout_s': config.health.timeout_s,
            'consecutive_ok_for_ready': config.health.consecutive_ok_for_ready,
            'failures_to_unhealthy': config.health.failures_to_unhealthy,
            'headers': dict(config.health.headers) if config.health.headers else {}
        }
        health_hash = _hash_dict(health_dict)
        
        return cls(
            # SLURM parameters
            gres=config.slurm.gres,
            cpus_per_task=config.slurm.cpus_per_task,
            mem=config.slurm.mem,
            time_limit=config.slurm.time_limit,
            constraint=config.slurm.constraint,
            account=config.slurm.account,
            partition=config.slurm.partition,
            reservation=config.slurm.reservation,
            env=dict(config.slurm.env),
            
            # SGLang parameters
            model_path=config.sglang.model_path,
            sglang_args=list(config.sglang.args),
            venv_path=config.sglang.venv_path,
            
            # Runtime parameters
            health_config_hash=health_hash,
            connectivity_mode=config.connectivity.mode,
        )
    
    def compute_hash(self) -> str:
        """Compute hash of this configuration snapshot."""
        # Convert to dict for consistent hashing
        config_dict = {
            'slurm': {
                'gres': self.gres,
                'cpus_per_task': self.cpus_per_task,
                'mem': self.mem,
                'time_limit': self.time_limit,
                'constraint': self.constraint,
                'account': self.account,
                'partition': self.partition,
                'reservation': self.reservation,
                'env': self.env,
            },
            'sglang': {
                'model_path': self.model_path,
                'args': self.sglang_args,
                'venv_path': self.venv_path,
            },
            'runtime': {
                'health_config_hash': self.health_config_hash,
                'connectivity_mode': self.connectivity_mode,
            }
        }
        
        return _hash_dict(config_dict)
    
    def requires_job_replacement(self, other: "ConfigSnapshot") -> bool:
        """Check if changes require SLURM job replacement."""
        job_affecting_fields = [
            'gres', 'cpus_per_task', 'mem', 'time_limit', 'constraint',
            'account', 'partition', 'reservation', 'env',
            'model_path', 'sglang_args', 'venv_path'
        ]
        
        for field in job_affecting_fields:
            if getattr(self, field, None) != getattr(other, field, None):
                return True
        return False
    
    def requires_health_update(self, other: "ConfigSnapshot") -> bool:
        """Check if changes require health monitoring update."""
        return self.health_config_hash != other.health_config_hash


@dataclass  
class WorkerGeneration:
    """Tracks worker generation and configuration context."""
    config_hash: str
    config_snapshot: ConfigSnapshot
    created_at: float
    generation_id: str  # Incrementing generation for this deployment
    
    def needs_replacement(self, current_snapshot: ConfigSnapshot) -> bool:
        """Check if this worker needs replacement due to config changes."""
        return self.config_snapshot.requires_job_replacement(current_snapshot)
    
    def needs_health_update(self, current_snapshot: ConfigSnapshot) -> bool:
        """Check if this worker needs health monitoring update."""  
        return self.config_snapshot.requires_health_update(current_snapshot)


def _hash_dict(data: Dict[str, Any]) -> str:
    """Create deterministic hash of dictionary."""
    # Sort keys for deterministic ordering
    json_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()[:16]  # 16 chars sufficient
