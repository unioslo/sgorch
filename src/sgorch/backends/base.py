from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from ..config import DeploymentConfig, SGLangConfig, TEIConfig


@dataclass
class LaunchPlan:
    """Shell-level launch instructions supplied by a backend."""

    command: List[str]
    log_file_name: str
    setup_lines: List[str] = field(default_factory=list)
    extra_env: Dict[str, str] = field(default_factory=dict)


@dataclass
class LaunchContext:
    """Minimal context passed to backends when constructing launch plans."""

    instance_idx: int
    instance_uuid: str
    remote_port: int


class BackendAdapter:
    """Common interface implemented by backend-specific adapters."""

    def __init__(self, deployment: DeploymentConfig):
        self.deployment = deployment

    @property
    def config(self) -> DeploymentConfig:
        return self.deployment

    @property
    def job_name_prefix(self) -> str:
        raise NotImplementedError

    @property
    def display_name(self) -> str:
        raise NotImplementedError

    @property
    def requires_router(self) -> bool:
        return True

    def validate_worker_metrics(self) -> None:
        """Validate enable_worker_metrics compatibility for this backend."""

    def build_launch_plan(self, ctx: LaunchContext) -> LaunchPlan:
        raise NotImplementedError

    def log_file_template(self) -> str:
        return f"{self.job_name_prefix}_{self.deployment.name}_$SLURM_JOB_ID.log"

    def log_file_path(self, job_id: str) -> str:
        return f"{self.deployment.slurm.log_dir}/{self.job_name_prefix}_{self.deployment.name}_{job_id}.log"

    def discovery_port_hints(self) -> list[int]:
        return [8000, 8001, 8080, 8888, 30000, 30001]


class SGLangBackendAdapter(BackendAdapter):
    @property
    def job_name_prefix(self) -> str:
        return "sgl"

    @property
    def display_name(self) -> str:
        return "SGLang"

    def validate_worker_metrics(self) -> None:
        cfg = self.deployment
        if cfg.enable_worker_metrics:
            args = list(self.deployment.sglang.args)
            if "--enable-metrics" not in args:
                raise ValueError(
                    "enable_worker_metrics is enabled but --enable-metrics is not in backend.args"
                )

    def build_launch_plan(self, ctx: LaunchContext) -> LaunchPlan:
        cfg: SGLangConfig = self.deployment.sglang
        args = list(cfg.args)

        processed_args: List[str] = []
        for arg in args:
            if arg == "{PORT}":
                processed_args.append("$PORT")
            else:
                processed_args.append(arg.replace("{PORT}", "$PORT"))

        if "--host" not in processed_args:
            processed_args.extend(["--host", "0.0.0.0"])
        if "--port" not in processed_args:
            processed_args.extend(["--port", "$PORT"])

        cmd: List[str] = ["python3 -m sglang.launch_server"]
        cmd.append(f"--model-path {cfg.model_path}")
        cmd.extend(processed_args)

        setup_lines: List[str] = []
        if cfg.venv_path:
            setup_lines.append(f"source {cfg.venv_path}/bin/activate || true")
        else:
            setup_lines.append(
                "source /cluster/home/jonalsa/sglang-test/.venv/bin/activate || conda activate sglang || source .venv/bin/activate || true"
            )

        return LaunchPlan(
            command=cmd,
            log_file_name=self.log_file_template(),
            setup_lines=setup_lines,
        )


class TEIBackendAdapter(BackendAdapter):
    @property
    def job_name_prefix(self) -> str:
        return "tei"

    @property
    def display_name(self) -> str:
        return "TEI"

    @property
    def requires_router(self) -> bool:
        return False

    def validate_worker_metrics(self) -> None:
        cfg = self.deployment
        tei_cfg: TEIConfig = cfg.tei
        if cfg.enable_worker_metrics and not tei_cfg.prometheus_port:
            raise ValueError(
                "enable_worker_metrics requires tei.prometheus_port to be set"
            )

    def build_launch_plan(self, ctx: LaunchContext) -> LaunchPlan:
        tei_cfg: TEIConfig = self.deployment.tei

        binary = tei_cfg.binary_path or "text-embeddings-router"
        cmd: List[str] = [binary, f"--model-id {tei_cfg.model_id}"]
        if tei_cfg.revision:
            cmd.append(f"--revision {tei_cfg.revision}")

        processed_args: List[str] = []
        for arg in tei_cfg.args:
            if arg == "{PORT}":
                processed_args.append("$PORT")
            else:
                processed_args.append(arg.replace("{PORT}", "$PORT"))

        if "--hostname" not in processed_args:
            processed_args.extend(["--hostname", "0.0.0.0"])
        if "--port" not in processed_args:
            processed_args.extend(["--port", "$PORT"])
        if tei_cfg.prometheus_port and "--prometheus-port" not in processed_args:
            processed_args.extend(["--prometheus-port", str(tei_cfg.prometheus_port)])

        cmd.extend(processed_args)

        return LaunchPlan(
            command=cmd,
            log_file_name=self.log_file_template(),
            extra_env=dict(tei_cfg.env),
        )

    def discovery_port_hints(self) -> list[int]:
        return [3000, 8080, 80]


def make_backend_adapter(deployment: DeploymentConfig) -> BackendAdapter:
    backend_cfg = deployment.backend
    if isinstance(backend_cfg, SGLangConfig):
        return SGLangBackendAdapter(deployment)
    if isinstance(backend_cfg, TEIConfig):
        return TEIBackendAdapter(deployment)
    raise ValueError(f"Unsupported backend type: {backend_cfg}")
