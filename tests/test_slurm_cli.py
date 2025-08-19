import subprocess

from sgorch.slurm.cli import SlurmCliAdapter
from sgorch.slurm.base import SubmitSpec


def mk_submit_spec(tmp_path):
    return SubmitSpec(
        name="sgl-d1-0",
        account="a",
        reservation=None,
        partition="p",
        qos=None,
        gres="g",
        constraint=None,
        time_limit="01:00:00",
        cpus_per_task=2,
        mem="1G",
        env={},
        stdout=str(tmp_path/"o.out"),
        stderr=str(tmp_path/"e.err"),
        script="# echo hi\n",
    )


def test_submit_parses_job_id_and_cleans_temp_file(monkeypatch, tmp_path):
    def fake_run(cmd, capture_output=True, text=True, timeout=None):
        return subprocess.CompletedProcess(cmd, 0, stdout="Submitted batch job 12345\n", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    cli = SlurmCliAdapter()
    job = cli.submit(mk_submit_spec(tmp_path))
    assert job == "12345"


def test_status_unknown_when_scontrol_nonzero(monkeypatch):
    def fake_run(cmd, capture_output=True, text=True, timeout=None):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="nope")
    monkeypatch.setattr(subprocess, "run", fake_run)
    cli = SlurmCliAdapter()
    info = cli.status("999")
    assert info.state == "UNKNOWN"


def test_list_jobs_filters_by_prefix_and_parses(monkeypatch):
    out = """
12345 p sgl-d1-0 u R 01:23 1 cn001
99999 p other u R 01:23 1 cn002
    """
    def fake_run(cmd, capture_output=True, text=True, timeout=None):
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    cli = SlurmCliAdapter()
    jobs = cli.list_jobs("sgl-d1-")
    assert len(jobs) == 1 and jobs[0].job_id == "12345"


def test_cancel_error_bubbles(monkeypatch):
    def fake_run(cmd, capture_output=True, text=True, timeout=None):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")
    monkeypatch.setattr(subprocess, "run", fake_run)
    cli = SlurmCliAdapter()
    try:
        cli.cancel("1")
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass

