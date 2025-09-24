import os
import shlex
from typing import Dict, Any, Optional

from ..backends.base import LaunchPlan


def render_sbatch_script(
    deploy_name: str,
    instance_idx: int,
    instance_uuid: str,
    account: str,
    partition: str,
    gres: str,
    cpus: int,
    mem: str,
    time_limit: str,
    job_name_prefix: str,
    display_name: str,
    launch_plan: LaunchPlan,
    log_dir: str = "/tmp",
    reservation: Optional[str] = None,
    qos: Optional[str] = None,
    constraint: Optional[str] = None,
    env_vars: Optional[Dict[str, str]] = None,
    remote_port: int = 8000,
    health_path: str = "/health",
    health_token_env: str = "WORKER_HEALTH_TOKEN",
    sbatch_extra: Optional[list[str]] = None,
) -> str:
    """Render SLURM sbatch script template."""

    env_vars = env_vars or {}
    sbatch_extra = sbatch_extra or []

    script_lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={job_name_prefix}-{deploy_name}-{instance_idx}",
        f"#SBATCH --account={account}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --gres={gres}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --time={time_limit}",
    ]

    if reservation:
        script_lines.append(f"#SBATCH --reservation={reservation}")
    if qos:
        script_lines.append(f"#SBATCH --qos={qos}")
    if constraint:
        script_lines.append(f"#SBATCH --constraint={constraint}")

    script_lines.extend([
        f"#SBATCH -o {log_dir}/%x_%j.out",
        f"#SBATCH -e {log_dir}/%x_%j.err",
        f"#SBATCH --comment=sgorch:{deploy_name}:{instance_uuid}",
    ])

    for extra in sbatch_extra:
        if extra.strip():
            script_lines.append(f"#SBATCH {extra}")

    script_lines.extend([
        "",
        "set -eo pipefail",
    ])

    combined_env: Dict[str, str] = {}
    for source in (env_vars, launch_plan.extra_env):
        if not source:
            continue
        for key, value in source.items():
            resolved = _resolve_env_value(value)
            if resolved is None:
                continue
            combined_env[key] = resolved

    for key, value in combined_env.items():
        script_lines.append(f"export {key}={shlex.quote(value)}")

    health_token_value = _resolve_env_value(os.getenv(health_token_env))
    if health_token_value:
        script_lines.append(f"export {health_token_env}={shlex.quote(health_token_value)}")
    script_lines.append(f"PORT={remote_port}")
    script_lines.append("")

    script_lines.extend([
        "# Activate environment",
        "set +u",
        "source ~/.bashrc || true",
        "set -u",
    ])

    script_lines.extend(launch_plan.setup_lines)

    script_lines.extend([
        "",
        "# Get node hostname and IP",
        'HOSTNAME=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)',
        'IP=$(getent hosts "$HOSTNAME" | awk \'{print $1}\' | head -n1)',
        "",
    ])

    if not launch_plan.command:
        raise ValueError("Backend launch command must not be empty")

    command_display = " ".join(launch_plan.command)
    server_log = f"{log_dir}/{launch_plan.log_file_name}"

    script_lines.extend([
        f"echo \"[$(date '+%Y-%m-%d %H:%M:%S')] Starting {display_name} worker on $HOSTNAME:$PORT...\"",
        f"echo \"[$(date '+%Y-%m-%d %H:%M:%S')] Command: {command_display}\" | tee {server_log}",
        f"echo \"[$(date '+%Y-%m-%d %H:%M:%S')] Logs: {server_log}\"",
        "",
    ])

    command_block = " \\\n  ".join(launch_plan.command)
    script_lines.append(command_block + " \\")
    script_lines.append(f"  >> {server_log} 2>&1 &")

    script_lines.extend([
        "",
        "PID=$!",
        f"echo \"[$(date '+%Y-%m-%d %H:%M:%S')] {display_name} worker started with PID $PID\"",
        f"echo \"[$(date '+%Y-%m-%d %H:%M:%S')] Monitor logs: tail -f {server_log}\"",
        "",
        "# Emit a single-line READY marker for adoption on restart:",
        "for i in {1..180}; do",
        f'  if [ -n "${{{health_token_env}:-}}" ]; then',
        f'    CURL_CMD="curl -fsS -H \\"Authorization: ${{{health_token_env}}}\\" \\"http://$IP:{remote_port}{health_path}\\""',
        "  else",
        f'    CURL_CMD="curl -fsS \\"http://$IP:{remote_port}{health_path}\\""',
        "  fi",
        '  if eval $CURL_CMD >/dev/null 2>&1; then',
        f'    echo "READY URL=http://$IP:{remote_port} JOB=$SLURM_JOB_ID INSTANCE={instance_uuid}"',
        "    break",
        "  fi",
        "  sleep 2",
        "done",
        "",
        "wait $PID",
    ])

    return "\n".join(script_lines)


def render_simple_script_template(template_vars: dict[str, Any]) -> str:
    """Simple string template replacement for scripts."""
    template = """#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --gres={gres}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
{optional_directives}
#SBATCH -o {stdout}
#SBATCH -e {stderr}
#SBATCH --comment={comment}

{script_body}
"""

    optional = []
    if template_vars.get("reservation"):
        optional.append(f"#SBATCH --reservation={template_vars['reservation']}")
    if template_vars.get("qos"):
        optional.append(f"#SBATCH --qos={template_vars['qos']}")
    if template_vars.get("constraint"):
        optional.append(f"#SBATCH --constraint={template_vars['constraint']}")

    template_vars["optional_directives"] = "\n".join(optional)

    return template.format(**template_vars)


def _resolve_env_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        trimmed = value.strip()
        if trimmed == "":
            return None
        if trimmed.startswith("${") and trimmed.endswith("}"):
            env_name = trimmed[2:-1]
            env_val = os.getenv(env_name)
            if env_val is None:
                return None
            env_val = env_val.strip()
            if env_val == "" or (env_val.startswith("${") and env_val.endswith("}")):
                return None
            return env_val
        if trimmed.startswith("$") and "${" in trimmed and trimmed.endswith("}"):
            # Something like ${{VAR}}; treat as unresolved placeholder
            return None
        return value

    resolved = str(value).strip()
    return resolved or None
