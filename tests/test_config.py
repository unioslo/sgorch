from pathlib import Path

import os
import textwrap

from sgorch.config import load_config


def write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(textwrap.dedent(content))
    return p


def test_config_loads_and_expands_env(tmp_path, env):
    env({"MY_TOKEN": "sekret"})
    cfg = write(
        tmp_path,
        "cfg.yaml",
        """
        orchestrator:
          metrics:
            enabled: true
        deployments:
          - name: d1
            replicas: 1
            connectivity:
              mode: direct
              tunnel_mode: local
              orchestrator_host: ohost
              advertise_host: 127.0.0.1
              local_port_range: [30000, 30010]
            router:
              base_url: http://router
              auth:
                type: header
                header_name: Authorization
                header_value_env: MY_TOKEN
            slurm:
              prefer: cli
              account: acct
              partition: part
              gres: gpu:1
              log_dir: {tmp}
            sglang:
              model_path: /m
        """.format(tmp=str(tmp_path)),
    )
    conf = load_config(cfg)
    assert conf.deployments[0].router.auth.header_value_env == "MY_TOKEN"


def test_config_unique_deploy_names_enforced(tmp_path):
    cfg = write(
        tmp_path,
        "cfg.yaml",
        """
        orchestrator: {{}}
        deployments:
          - name: d1
            replicas: 1
            connectivity:
              mode: direct
              tunnel_mode: local
              orchestrator_host: ohost
              advertise_host: 127.0.0.1
            router: {{ base_url: http://r }}
            slurm: {{ prefer: cli, account: a, partition: p, gres: g, log_dir: {tmp} }}
            sglang: {{ model_path: /m }}
          - name: d1
            replicas: 1
            connectivity:
              mode: direct
              tunnel_mode: local
              orchestrator_host: ohost
              advertise_host: 127.0.0.1
            router: {{ base_url: http://r }}
            slurm: {{ prefer: cli, account: a, partition: p, gres: g, log_dir: {tmp} }}
            sglang: {{ model_path: /m }}
        """.format(tmp=str(tmp_path)),
    )
    try:
        load_config(cfg)
        raise AssertionError("expected ValueError for duplicate names")
    except ValueError:
        pass


def test_defaults_applied_for_optional_sections(tmp_path):
    cfg = write(
        tmp_path,
        "cfg.yaml",
        """
        deployments:
          - name: d1
            replicas: 1
            connectivity:
              mode: direct
              tunnel_mode: local
              orchestrator_host: ohost
              advertise_host: 127.0.0.1
            router: {{ base_url: http://r }}
            slurm: {{ prefer: cli, account: a, partition: p, gres: g, log_dir: {tmp} }}
            sglang: {{ model_path: /m }}
        """.format(tmp=str(tmp_path)),
    )
    conf = load_config(cfg)
    orch = conf.orchestrator
    assert orch.metrics.enabled is True
    assert orch.state.backend == "file"
    dep = conf.deployments[0]
    assert dep.health.interval_s > 0
    assert dep.policy.restart_backoff_s > 0
