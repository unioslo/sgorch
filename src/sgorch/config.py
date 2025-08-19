from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, validator
from ruamel.yaml import YAML


class MetricsConfig(BaseModel):
    enabled: bool = True
    bind: str = "0.0.0.0"
    port: int = 9315


class EmailConfig(BaseModel):
    smtp_host: Optional[str] = None
    from_addr: Optional[str] = None
    to_addrs: list[str] = []


class NotificationsConfig(BaseModel):
    type: Literal["log_only", "email"] = "log_only"
    email: EmailConfig = Field(default_factory=EmailConfig)

class StateConfig(BaseModel):
    backend: Literal["file"] = "file"
    file_path: Optional[str] = None


class OrchestratorConfig(BaseModel):
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    notifications: NotificationsConfig = Field(default_factory=NotificationsConfig)
    state: StateConfig = Field(default_factory=StateConfig)


class SSHConfig(BaseModel):
    user: Optional[str] = None
    opts: list[str] = []


class ConnectivityConfig(BaseModel):
    mode: Literal["direct", "tunneled"] = "tunneled"
    tunnel_mode: Literal["local", "reverse"] = "local"
    orchestrator_host: str
    advertise_host: str
    local_port_range: tuple[int, int] = (30000, 30999)
    ssh: SSHConfig = Field(default_factory=SSHConfig)


class AuthConfig(BaseModel):
    type: Literal["header", "none"] = "header"
    header_name: str = "Authorization"
    header_value_env: str


class EndpointsConfig(BaseModel):
    list: str = "/workers/list"
    add: str = "/workers/add"
    remove: str = "/workers/remove"


class RouterConfig(BaseModel):
    base_url: str
    endpoints: EndpointsConfig = Field(default_factory=EndpointsConfig)
    auth: Optional[AuthConfig] = None


class SlurmConfig(BaseModel):
    prefer: Literal["rest", "cli", "auto"] = "auto"
    account: str
    reservation: Optional[str] = None
    partition: str
    qos: Optional[str] = None
    gres: str
    constraint: Optional[str] = None
    time_limit: str = "24:00:00"
    cpus_per_task: int = 16
    mem: str = "64G"
    log_dir: str
    env: dict[str, str] = {}
    sbatch_extra: list[str] = []


class SGLangConfig(BaseModel):
    model_path: str
    venv_path: Optional[str] = None
    args: list[str] = []


class HealthConfig(BaseModel):
    path: str = "/health"
    interval_s: int = 5
    timeout_s: int = 3
    consecutive_ok_for_ready: int = 2
    failures_to_unhealthy: int = 3
    headers: dict[str, str] = {}


class PolicyConfig(BaseModel):
    restart_backoff_s: int = 60
    deregister_grace_s: int = 10
    start_grace_period_s: int = 600
    predrain_seconds_before_walltime: int = 180
    node_blacklist_cooldown_s: int = 600


class DeploymentConfig(BaseModel):
    name: str
    replicas: int
    connectivity: ConnectivityConfig
    router: RouterConfig
    slurm: SlurmConfig
    sglang: SGLangConfig
    health: HealthConfig = Field(default_factory=HealthConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)


class Config(BaseModel):
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    deployments: list[DeploymentConfig]

    @validator("deployments")
    def validate_unique_deployment_names(cls, v):
        names = [d.name for d in v]
        if len(names) != len(set(names)):
            raise ValueError("Deployment names must be unique")
        return v


def expand_env_vars(data: Any) -> Any:
    """Recursively expand ${VAR} patterns in configuration data."""
    if isinstance(data, str):
        # Simple ${VAR} expansion
        if data.startswith("${") and data.endswith("}"):
            var_name = data[2:-1]
            return os.environ.get(var_name, data)
        return data
    elif isinstance(data, dict):
        return {k: expand_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [expand_env_vars(item) for item in data]
    return data


def load_config(config_path: str | Path) -> Config:
    """Load and validate configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    yaml = YAML(typ="safe", pure=True)
    with config_path.open() as f:
        raw_data = yaml.load(f)
    
    # Expand environment variables
    expanded_data = expand_env_vars(raw_data)
    
    # Validate with Pydantic
    return Config(**expanded_data)
