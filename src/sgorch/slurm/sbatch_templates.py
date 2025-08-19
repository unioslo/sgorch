from typing import Dict, Any, Optional


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
    reservation: Optional[str] = None,
    qos: Optional[str] = None,
    constraint: Optional[str] = None,
    log_dir: str = "/tmp",
    env_vars: Optional[Dict[str, str]] = None,
    model_path: str = "",
    remote_port: int = 8000,
    sglang_args: Optional[list[str]] = None,
    health_path: str = "/v1/health",
    health_token_env: str = "WORKER_HEALTH_TOKEN",
    sglang_venv_path: Optional[str] = None,
    sbatch_extra: Optional[list[str]] = None,
) -> str:
    """Render SLURM sbatch script template."""
    
    env_vars = env_vars or {}
    sglang_args = sglang_args or []
    sbatch_extra = sbatch_extra or []
    
    # Build sbatch header
    script_lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name=sgl-{deploy_name}-{instance_idx}",
        f"#SBATCH --account={account}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --gres={gres}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --time={time_limit}",
    ]
    
    # Optional SBATCH directives
    if reservation:
        script_lines.append(f"#SBATCH --reservation={reservation}")
    if qos:
        script_lines.append(f"#SBATCH --qos={qos}")
    if constraint:
        script_lines.append(f"#SBATCH --constraint={constraint}")
    
    # Logs and metadata
    script_lines.extend([
        f"#SBATCH -o {log_dir}/%x_%j.out",
        f"#SBATCH -e {log_dir}/%x_%j.err",
        f"#SBATCH --comment=sgorch:{deploy_name}:{instance_uuid}",
    ])
    
    # Add any extra sbatch directives
    for extra in sbatch_extra:
        if extra.strip():
            script_lines.append(f"#SBATCH {extra}")
    
    script_lines.extend([
        "",
        "set -euo pipefail",
    ])
    
    # Environment variables
    for key, value in env_vars.items():
        script_lines.append(f"export {key}={value}")
    
    script_lines.append(f"PORT={remote_port}")
    script_lines.append("")
    
    # Activation logic
    script_lines.extend([
        "# Activate environment",
        "source ~/.bashrc || true",
    ])
    
    if sglang_venv_path:
        script_lines.append(f"source {sglang_venv_path}/bin/activate || true")
    else:
        script_lines.append("source /cluster/home/jonalsa/sglang-test/.venv/bin/activate || conda activate sglang || source .venv/bin/activate || true")
    
    script_lines.extend([
        "",
        "# Get node hostname and IP",
        'HOSTNAME=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)',
        'IP=$(getent hosts "$HOSTNAME" | awk \'{print $1}\' | head -n1)',
        "",
    ])
    
    # Build sglang command
    sglang_cmd = ["python3 -m sglang.launch_server"]
    sglang_cmd.append(f"--model-path {model_path}")
    
    # Process args, replacing {PORT} template
    processed_args = []
    for arg in sglang_args:
        if arg == "{PORT}":
            processed_args.append(f"${remote_port}")
        else:
            processed_args.append(arg.replace("{PORT}", str(remote_port)))
    sglang_cmd.extend(processed_args)
    
    # Add default host and port if not in args
    if "--host" not in processed_args:
        sglang_cmd.extend(["--host", "0.0.0.0"])
    if "--port" not in processed_args:
        sglang_cmd.extend(["--port", f"{remote_port}"])
    
    # Launch sglang server
    script_lines.extend([
        " \\\n  ".join(sglang_cmd) + " \\",
        f"  > {log_dir}/server_$SLURM_JOB_ID.log 2>&1 &",
        "",
        "PID=$!",
        "",
        "# Emit a single-line READY marker for adoption on restart:",
        "for i in {1..180}; do",
        f'  if curl -fsS -H "Authorization: ${{{health_token_env}}}" \\',
        f'      "http://$IP:{remote_port}{health_path}" >/dev/null; then',
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
    
    # Build optional directives
    optional = []
    if template_vars.get("reservation"):
        optional.append(f"#SBATCH --reservation={template_vars['reservation']}")
    if template_vars.get("qos"):
        optional.append(f"#SBATCH --qos={template_vars['qos']}")
    if template_vars.get("constraint"):
        optional.append(f"#SBATCH --constraint={template_vars['constraint']}")
    
    template_vars["optional_directives"] = "\n".join(optional)
    
    return template.format(**template_vars)