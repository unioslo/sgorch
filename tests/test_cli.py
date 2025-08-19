from pathlib import Path
from typer.testing import CliRunner

from sgorch.main import app


def _write_cfg(tmp_path: Path) -> Path:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        """
        orchestrator: {}
        deployments:
          - name: d
            replicas: 1
            connectivity:
              mode: direct
              tunnel_mode: local
              orchestrator_host: o
              advertise_host: 127.0.0.1
            router: { base_url: http://r }
            slurm: { prefer: cli, account: a, partition: p, gres: g, log_dir: %(tmp)s }
            sglang: { model_path: /m }
        """ % {"tmp": str(tmp_path)}
    )
    return cfg


def test_validate_happy_path(tmp_path):
    cfg = _write_cfg(tmp_path)
    r = CliRunner().invoke(app, ["validate", "-c", str(cfg)])
    assert r.exit_code == 0
    assert "Configuration is valid" in r.stdout


def test_status_json_and_human_output(tmp_path):
    cfg = _write_cfg(tmp_path)
    rj = CliRunner().invoke(app, ["status", "-c", str(cfg), "--json"]) 
    assert rj.exit_code == 0
    assert "deployments" in rj.stdout
    rh = CliRunner().invoke(app, ["status", "-c", str(cfg)])
    assert rh.exit_code == 0
    assert "SGOrch Configuration Status" in rh.stdout


def test_run_missing_config_exits_1(tmp_path):
    missing = tmp_path / "nope.yaml"
    r = CliRunner().invoke(app, ["run", "-c", str(missing)])
    assert r.exit_code != 0

