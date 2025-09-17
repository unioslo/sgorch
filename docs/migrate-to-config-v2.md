# Migrating to SGOrch Configuration v2

The v2 configuration schema introduces support for multiple inference backends while remaining backwards-compatible with existing SGLang deployments. This guide explains the changes and shows how to adapt your YAML configs.

## What's new

- **Backend block:** Each deployment now declares a `backend:` section with a `type` discriminator (`sglang` or `tei`). This unifies backend-specific settings and prepares for future engines.
- **Optional router config:** Deployments whose backend does not require a router (e.g. Hugging Face TEI) can omit the `router:` block entirely. SGOrch skips router registration in that case.
- **TEI support:** You can orchestrate [huggingface/text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference) jobs alongside SGLang workers, including custom binary paths, arguments, and Hugging Face credentials.
- **Launch abstraction:** SBATCH scripts and adoption logic are now backend-aware, allowing job names, logs, and READY markers to differ per backend while keeping restart behaviour intact.

Existing configs that use the legacy top-level `sglang:` block continue to load, but SGOrch will emit the new structure when saving or expanding configs. We recommend migrating to the explicit `backend:` form for clarity.

## Step-by-step migration

### 1. Wrap SGLang settings under `backend`

**Before**
```yaml
deployments:
  - name: gpt-oss-20b
    # ...
    sglang:
      model_path: openai/gpt-oss-20b
      venv_path: /path/to/sglang/.venv
      args:
        - "--host"
        - "0.0.0.0"
        - "--port"
        - "{PORT}"
```

**After**
```yaml
deployments:
  - name: gpt-oss-20b
    # ...
    backend:
      type: sglang
      model_path: openai/gpt-oss-20b
      venv_path: /path/to/sglang/.venv
      args:
        - "--host"
        - "0.0.0.0"
        - "--port"
        - "{PORT}"
```

No other fields need to change. The orchestrator still reads router/slurm/health/policy exactly as before.

### 2. (Optional) Enable TEI deployments

To add a Hugging Face TEI deployment, create a new block with `backend.type: tei`. A minimal example:

```yaml
deployments:
  - name: tei-bge
    replicas: 2
    connectivity:
      mode: direct
      tunnel_mode: local
      orchestrator_host: router-vm.internal
      advertise_host: tei-service.internal
      local_port_range: [31000, 31999]
    slurm:
      prefer: cli
      account: embeddings
      partition: gpu
      gres: gpu:1
      log_dir: /var/log/tei
    backend:
      type: tei
      model_id: BAAI/bge-base-en-v1.5
      args:
        - "--hostname"
        - "0.0.0.0"
        - "--port"
        - "{PORT}"
        - "--json-output"
      env:
        HF_TOKEN: "${HF_TOKEN}"
      prometheus_port: 9100
```

Notes:
- `router` is optional; omit it unless you have a router that needs explicit registration.
- Set `env` if TEI requires credentials or cache paths.
- `prometheus_port` is optional but required if `enable_worker_metrics` is set for the deployment.

### 3. Environment variable reminders

- SGLang deployments still rely on `ROUTER_TOKEN` (if router auth is configured) and `WORKER_HEALTH_TOKEN` for `/health` probes.
- TEI deployments typically need `HF_TOKEN` to download weights. Export it on the orchestrator host so it propagates into SLURM jobs via `backend.env`.

### 4. Validate the config

Use the CLI to ensure the schema is satisfied:

```bash
sgorch validate --config path/to/config.yaml
```

If you see an error about missing backend configuration, confirm that every deployment now contains either `backend.type: sglang` or `backend.type: tei`.

### 5. Rolling out

- Restart the orchestrator with the new config.
- Monitor the logs for warnings about ignored router configuration—those indicate a TEI deployment still has a `router:` block that will be skipped.
- Confirm SGLang jobs continue to appear with `sgl-<name>-*` job names and TEI jobs with `tei-<name>-*`.

## Troubleshooting

- **"Deployment config missing backend configuration"**: The migration shim only runs when the `sglang` key exists. Ensure you renamed the block to `backend` if you removed the old key.
- **Router errors for TEI**: Remove `router:` from TEI deployments; otherwise SGOrch will warn about unused router settings.
- **Prometheus metrics missing for TEI**: Set `backend.prometheus_port` and make sure `enable_worker_metrics` is enabled on the deployment.

By landing on the `backend` structure you unlock multi-backend orchestration while keeping existing workflows intact.
