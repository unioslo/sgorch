import os

import pytest

from sgorch.backends.base import LaunchPlan
from sgorch.slurm.sbatch_templates import render_sbatch_script


def _make_plan(env=None):
    return LaunchPlan(
        command=["echo", "hello"],
        log_file_name="worker.log",
        extra_env=env or {},
    )


def _render(env=None, backend_env=None):
    plan = _make_plan(backend_env)
    return render_sbatch_script(
        deploy_name="dep",
        instance_idx=0,
        instance_uuid="uuid",
        account="acct",
        partition="part",
        gres="gpu:1",
        cpus=4,
        mem="16G",
        time_limit="01:00:00",
        job_name_prefix="tei",
        display_name="TEI",
        launch_plan=plan,
        log_dir="/tmp",
        env_vars=env,
        remote_port=30000,
        health_path="/health",
    )


def test_env_placeholders_are_dropped_when_unset(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)

    script = _render(env={"HF_TOKEN": "${HF_TOKEN}"})

    assert "export HF_TOKEN" not in script


def test_env_is_resolved_from_process_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "real-token")

    script = _render(env={"HF_TOKEN": "${HF_TOKEN}"})

    assert "export HF_TOKEN=real-token" in script


def test_backend_env_overrides_slurm_env(monkeypatch):
    monkeypatch.setenv("FOO", "from-env")

    script = _render(env={"FOO": "${FOO}"}, backend_env={"FOO": "backend"})

    assert "export FOO=backend" in script
    assert script.count("export FOO") == 1


def test_health_token_only_exported_when_present(monkeypatch):
    monkeypatch.delenv("WORKER_HEALTH_TOKEN", raising=False)
    script = _render()
    assert "export WORKER_HEALTH_TOKEN" not in script

    monkeypatch.setenv("WORKER_HEALTH_TOKEN", "token")
    script_with_token = _render()
    assert "export WORKER_HEALTH_TOKEN=token" in script_with_token
