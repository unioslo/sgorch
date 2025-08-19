# SLURM ↔ SGLang Orchestrator — Final Spec

## 0) Scope & assumptions

* Runs as a **user-level systemd service** on a VM inside the SLURM cluster.
* Manages **multiple deployments** (each = model + router + slurm profile).
* Launches SGLang workers as **SLURM jobs**, maintains **SSH tunnels** so the router can reach each worker, registers/deregisters with router, and health-checks workers (SLURM + `/v1/health` with **auth**).
* Uses **slurmrestd** if available; otherwise falls back to CLI. The SLURM layer is an **interface** so it’s swappable later.
* **Stateless** by design, but must **resume gracefully** by discovering current jobs, logs, and router workers.
* Initial scaling is **fixed** per deployment (e.g., `replicas=2`), but architecture allows future dynamic scaling.
* **Prometheus** `/metrics` is included (minimal), plus a **notification hook** (no-op now; pluggable email later).
* “Best practice” logging: structured JSON to stdout/stderr → systemd-journald.

---

## 1) High-level architecture

### Components

* **Reconciler** (per deployment): desired replicas vs. actual (SLURM + router), converge via plan.
* **SLURM adapter (interface)**: `SlurmRestAdapter` (preferred) / `SlurmCliAdapter` (fallback).
* **Router client**: add/remove/list workers, configurable endpoints & auth.
* **Health engine**: periodic HTTP probes (with headers) + SLURM state.
* **Tunnel manager**: creates and supervises SSH port forwards (L/R), maps **router-reachable URL** ↔ **node\:port**.
* **Port allocator**: assigns unique local and remote ports; avoids collisions.
* **State view**: in-memory; on startup, **adopt** existing jobs and reconstruct mappings from SLURM + logs + router.
* **Metrics & notifications**: Prometheus `/metrics`; notifier interface (email later).

### Core flows

1. **Submit**: choose port(s), craft sbatch script, submit.
2. **Discover**: when job RUNNING, detect node & remote port, start SSH tunnel, probe `/v1/health`.
3. **Register**: after `k` healthy probes, `router.add(worker_url)` (the **router-reachable** URL, often via tunnel).
4. **Monitor**: continuous probes; on consecutive failures or SLURM errors → **drain** (router.remove), cancel job, replace.
5. **Walltime**: pre-drain and resubmit before SLURM time limit expires.
6. **Resume**: on orchestrator restart, rebuild state: list SLURM jobs + parse worker logs for URL hints + list router workers + rebuild tunnels.

---

## 2) File / package layout

```
sgorch/
  __init__.py
  main.py                     # entrypoint: CLI + service mode
  config.py                   # pydantic models; load/validate YAML
  logging_setup.py            # JSON logging config
  orchestrator.py             # process manager: loads deployments, runs reconcilers
  reconciler.py               # converge loop per deployment

  slurm/
    __init__.py
    base.py                   # ISlurm interface
    rest.py                   # SlurmRestAdapter
    cli.py                    # SlurmCliAdapter
    sbatch_templates.py       # jinja2-free small Python template strings
    parse.py                  # helpers to parse squeue/sacct output (CLI fallback)

  router/
    __init__.py
    client.py                 # RouterClient (httpx)

  health/
    __init__.py
    http_probe.py             # Auth header support, backoff policies

  net/
    __init__.py
    ports.py                  # PortAllocator
    tunnel.py                 # TunnelManager (ssh -L/-R supervision)
    hostaddr.py               # helper to resolve node IP/hostname

  policy/
    __init__.py
    failure.py                # thresholds, backoff, cooldown, node blacklisting

  discover/
    __init__.py
    adopt.py                  # “resume gracefully” logic

  metrics/
    __init__.py
    prometheus.py             # optional tiny /metrics server

  notify/
    __init__.py
    base.py                   # Notifier interface
    log_only.py               # default no-op / log notifier
    email.py                  # placeholder for future

  util/
    __init__.py
    timeouts.py
    backoff.py

  # Tests & artifacts
  tests/
  scripts/
    systemd-user.service      # unit file example
  examples/
    config.yaml               # example multi-deployment config
```

---

## 3) Interfaces (stable surfaces)

### SLURM (swappable)

```python
# slurm/base.py
from dataclasses import dataclass
from typing import Optional, Literal

JobState = Literal["PENDING","RUNNING","COMPLETED","FAILED","CANCELLED","TIMEOUT","NODE_FAIL","UNKNOWN"]

@dataclass
class SubmitSpec:
    name: str
    account: str
    reservation: Optional[str]
    partition: str
    qos: Optional[str]
    gres: str
    constraint: Optional[str]
    time_limit: str          # "HH:MM:SS"
    cpus_per_task: int
    mem: str                 # "64G"
    env: dict[str,str]
    stdout: str
    stderr: str
    script: str              # full bash script text

@dataclass
class JobInfo:
    job_id: str
    state: JobState
    node: Optional[str]      # "cn123" when RUNNING
    time_left_s: Optional[int]

class ISlurm:
    def submit(self, spec: SubmitSpec) -> str: ...
    def status(self, job_id: str) -> JobInfo: ...
    def cancel(self, job_id: str) -> None: ...
    def list_jobs(self, name_prefix: str) -> list[JobInfo]: ...
```

### Router

```python
# router/client.py
class RouterClient:
    def list(self) -> set[str]: ...
    def add(self, url: str) -> None: ...
    def remove(self, url: str) -> None: ...
```

### Tunnels

```python
# net/tunnel.py
@dataclass
class TunnelSpec:
    mode: Literal["local","reverse"]   # ssh -L vs ssh -R
    orchestrator_host: str             # VM hostname (for reverse tunnels)
    advertise_host: str                # advertised host (e.g., vm-ip or 127.0.0.1)
    advertise_port: int                # local (for -L) or router port (for -R)
    remote_host: str                   # compute node IP/host
    remote_port: int                   # sglang worker port
    ssh_user: Optional[str]
    ssh_opts: list[str]                # e.g., KeepAlive, ProxyJump

class TunnelManager:
    def ensure(self, key: str, spec: TunnelSpec) -> None: ...
    def is_up(self, key: str) -> bool: ...
    def drop(self, key: str) -> None: ...
    def gc(self) -> None: ...
```

### Reconciler (per deployment)

```python
# reconciler.py
class Reconciler:
    def tick(self) -> None: ...  # runs the converge plan once
```

---

## 4) Configuration schema (YAML)

```yaml
orchestrator:
  metrics:
    enabled: true
    bind: "0.0.0.0"
    port: 9315
  notifications:
    type: log_only           # future: email
    email:                   # for future
      smtp_host: null
      from_addr: null
      to_addrs: []

deployments:
  - name: llama-8b
    replicas: 2
    connectivity:
      # Either "direct" or "tunneled". If tunneled, choose local or reverse.
      mode: tunneled
      tunnel_mode: local              # local: ssh -L (VM→node); reverse: ssh -R (node→VM)
      orchestrator_host: "router-vm.internal"  # VM hostname (for reverse tunnels)
      advertise_host: "router-vm.internal"     # what router will advertise to clients
      local_port_range: [30000, 30999]        # for -L or -R port allocation
      ssh:
        user: "jonas"
        opts: ["-o","ServerAliveInterval=15","-o","ExitOnForwardFailure=yes"]
        # optional: ProxyJump, IdentityFile, etc.

    router:
      base_url: "http://router-vm.internal:8080"
      endpoints:
        list: "/workers/list"
        add: "/workers/add"
        remove: "/workers/remove"
      auth:
        type: header
        header_name: "Authorization"
        header_value_env: "ROUTER_TOKEN"

    slurm:
      prefer: rest            # rest|cli|auto
      account: "acctX"
      reservation: "resY"     # optional
      partition: "gpu"
      qos: "normal"
      gres: "gpu:1"
      constraint: "a100|h100" # from your config
      time_limit: "24:00:00"
      cpus_per_task: 16
      mem: "64G"
      log_dir: "/home/jonas/sg-logs"
      env:
        HF_HOME: "/mnt/hf-cache"
        HUGGINGFACE_HUB_CACHE: "/mnt/hf-cache"
      sbatch_extra: []        # free-form extras

    sglang:
      # args appended to: python -m sglang.launch_server
      model_path: "meta-llama/Llama-3.1-8B-Instruct"
      args:
        - "--host"
        - "0.0.0.0"
        - "--port"
        - "{PORT}"
        - "--ctx-len"
        - "8192"

    health:
      path: "/v1/health"
      interval_s: 5
      timeout_s: 3
      consecutive_ok_for_ready: 2
      failures_to_unhealthy: 3
      # HEALTH AUTH REQUIRED (your answer to Q8)
      headers:
        Authorization: "${WORKER_HEALTH_TOKEN}"

    policy:
      restart_backoff_s: 60
      deregister_grace_s: 10
      start_grace_period_s: 600
      predrain_seconds_before_walltime: 180
      node_blacklist_cooldown_s: 600
```

Notes

* **{PORT}** is templated by the orchestrator per job.
* `${ENV}` values are expanded at load time via pydantic-settings.

---

## 5) SLURM submission & discovery

### Job script template (rendered per job)

* Minimal, robust, no external templater required.

```bash
#!/usr/bin/env bash
#SBATCH --job-name=sgl-${DEPLOY}-${IDX}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=${GRES}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME_LIMIT}
% if RESERVATION: # (only if set)
#SBATCH --reservation=${RESERVATION}
% endif
% if QOS:
#SBATCH --qos=${QOS}
% endif
% if CONSTRAINT:
#SBATCH --constraint=${CONSTRAINT}
% endif
#SBATCH -o ${LOG_DIR}/%x_%j.out
#SBATCH -e ${LOG_DIR}/%x_%j.err
#SBATCH --comment=sgorch:${DEPLOY}:${INSTANCE_UUID}

set -euo pipefail
export HF_HOME=${HF_HOME}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE}
PORT=${REMOTE_PORT}

# Activate environment
source ~/.bashrc || true
conda activate sglang || source ${VENV}/bin/activate || true

HOSTNAME=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
IP=$(getent hosts "$HOSTNAME" | awk '{print $1}' | head -n1)

python -m sglang.launch_server \
  --model-path ${MODEL_PATH} \
  ${EXTRA_ARGS} \
  --host 0.0.0.0 \
  --port ${REMOTE_PORT} \
  > ${LOG_DIR}/server_${SLURM_JOB_ID}.log 2>&1 &

PID=$!

# Emit a single-line READY marker for adoption on restart:
for i in {1..180}; do
  if curl -fsS -H "Authorization: ${WORKER_HEALTH_TOKEN}" \
      "http://$IP:${REMOTE_PORT}${HEALTH_PATH}" >/dev/null; then
    echo "READY URL=http://$IP:${REMOTE_PORT} JOB=$SLURM_JOB_ID INSTANCE=${INSTANCE_UUID}"
    break
  fi
  sleep 2
done

wait $PID
```

### Discovery / resume

* On startup, for each deployment:

  1. `slurm.list_jobs(name_prefix="sgl-${DEPLOY}-")`
  2. For RUNNING jobs: tail `server_${JOBID}.log` and `%x_%j.out` for `READY URL=... INSTANCE=...`.
  3. If missing, compute from node IP + configured remote port (kept in orchestrator mapping if it submitted the job).
  4. Mark discovered ports as "in-use" in the `PortAllocator` to prevent collisions.
  5. Recreate tunnels for discovered jobs and re-register with router if needed (reconciliation).

---

## 6) Tunnels & addressing

* **Connectivity profiles**:

  * `direct`: router reaches node directly → advertised worker URL is `http://node_ip:remote_port` (no tunnels).
  * `tunneled/local (-L)`: VM listens on `localhost:local_port` and forwards to `node_ip:remote_port`; **advertised URL** becomes `http://advertise_host:local_port` (typically the VM's routable hostname).
  * `tunneled/reverse (-R)`: the node opens a reverse tunnel to `orchestrator_host:advertise_port`; the job prologue can start `ssh -N -R`. (Supported but default to local -L from the orchestrator; reverse is useful when compute nodes can't be reached from the VM.)

* **Supervision**: `TunnelManager` spawns `ssh` via `subprocess.Popen` with `-N -L` (or `-R`) and `ExitOnForwardFailure=yes`, tracks PIDs, restarts with backoff if broken, and validates by a local TCP connect + HTTP probe.

* **Mapping**: `key = f"{deployment}:{job_id}"` → `{remote_host, remote_port, advertise_host, advertise_port, mode, instance_uuid}`.

---

## 7) Health, registration, draining

* Probe `GET {worker_url}{health.path}` with configured **headers** (auth). Timeout & interval as config.
* **Ready** = `consecutive_ok_for_ready` successes; then `router.add(advertised_url)`.
* On **failure** `>= failures_to_unhealthy`: `router.remove(advertised_url)` → wait `deregister_grace_s` → cancel SLURM job → schedule replacement (respect `restart_backoff_s`).
* **Walltime**: from `status(job_id).time_left_s`. If `< predrain_seconds_before_walltime`, pre-drain & resubmit; when replacement is **ready**, cancel the old job.

---

## 8) Policies & failure handling

* **Node blacklist** on `NODE_FAIL` or repeated boot failures: skip node for `node_blacklist_cooldown_s`.
* **Backoff with jitter** for submissions & router calls.
* **Circuit breaker** (optional): if replacement fails too often, hold at lower replica count and notify.

---

## 9) Metrics & notifications

* **Prometheus** (enabled by config): counters & gauges only:

  * `sgorch_workers_desired{deployment}`, `sgorch_workers_ready{deployment}`
  * `sgorch_tunnels_up{deployment}`, `sgorch_restarts_total{deployment,reason}`
  * `sgorch_router_errors_total`, `sgorch_submit_errors_total`
* Serve on `orchestrator.metrics.bind:port` via `prometheus_client.start_http_server(...)`.
* **Notifications**: `Notifier` interface. Default `log_only`. Later add `EmailNotifier` using SMTP creds from config; emit on state transitions: unhealthy, replacement loop, circuit breaker trip.

---

## 10) Logging (best practice)

* **Stdlib `logging`** with a tiny JSON formatter:

  * Fields: `ts`, `level`, `msg`, `deployment`, `job_id`, `worker_url`, `event`, `details`.
* Write to stdout; rely on systemd-journald for rotation & persistence.
* Log levels: `INFO` for state changes, `DEBUG` for polling, `WARNING/ERROR` for failures.

---

## 11) CLI & service

* **CLI (typer)**:

  * `sgorch run --config config.yaml` (foreground; used by systemd)
  * `sgorch status [--deployment X]`
  * `sgorch adopt` (force reconcile & adoption)
  * `sgorch scale DEPLOY N` (future)
* **systemd user unit (`scripts/systemd-user.service`)**:

  ```
  [Unit]
  Description=SGOrch (SLURM-SGLang Orchestrator)

  [Service]
  ExecStart=%h/sgorch/.venv/bin/sgorch run --config %h/sgorch/config.yaml
  Restart=always
  Environment=ROUTER_TOKEN=...
  Environment=WORKER_HEALTH_TOKEN=...
  # Hardening (optional):
  NoNewPrivileges=true
  PrivateTmp=true
  ProtectSystem=full
  ProtectHome=read-only

  [Install]
  WantedBy=default.target
  ```

---

## 12) Dependencies & project management

**Using `uv` for dependency management:**

```toml
# pyproject.toml
[project]
name = "sgorch"
version = "0.1.0"
description = "SLURM ↔ SGLang Orchestrator"
requires-python = ">=3.10"
dependencies = [
    "httpx>=0.27",
    "tenacity>=8.2",
    "pydantic>=2",
    "ruamel.yaml>=0.18",
    "typer>=0.9",
    "prometheus_client>=0.20",
]

[project.scripts]
sgorch = "sgorch.main:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

**Installation:**
```bash
# Install uv if not already available
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install
uv venv
source .venv/bin/activate  # or: .venv\Scripts\activate on Windows
uv pip install -e .
```

(SSH handled via system `ssh`; no extra deps.)

---

## 13) Test plan (practical)

* **Dry-run adapter**: a fake SLURM that “runs” jobs instantly → validates reconcile logic and router interactions without a cluster.
* **Cluster smoke**: single deployment, replicas=1, tunneled `-L`, confirm READY → registered → traffic OK → kill process → observe drain/replace.
* **Walltime**: short time limit (e.g., 5m) with `predrain=60s` → verify new worker becomes ready before old is cancelled.
* **Resume**: kill orchestrator, restart; verify adoption of existing jobs + tunnel recreation + router reconciliation.

---

## 14) Security

* Keep tokens in env (`ROUTER_TOKEN`, `WORKER_HEALTH_TOKEN`) not in the YAML; YAML can reference env with `${...}`.
* Prefer HTTPS to router if available; configurable CA bundle.
* SSH options: `-o StrictHostKeyChecking=accept-new` (or manage known\_hosts explicitly).

---

## 15) Implementation notes (coherence & pitfalls)

* **Advertised URL** must be **reachable by the router**. In tunneled mode, that's the VM host/port, not the node's IP.
* **Health auth** is mandatory (your setup). Make probes succeed even when API auth is enabled.
* **No durable DB**: adoption logic + log markers (`READY URL=... INSTANCE=...`) are what make stateless resume work.
* **Endpoint drift**: router endpoints are config-driven; detect capabilities at startup (e.g., probe `list`, noop add/remove).
* **Backpressure**: if submission is refused (quota/reservation), log + notify; don't thrash.
* **State consistency**: The reconciler's `tick()` method should be idempotent and operate on a consistent snapshot of state. SLURM state can change between calls, so design for level-triggered behavior where the next `tick` corrects any drift.
* **Unique job identity**: Each worker instance gets a UUID (`INSTANCE_UUID`) for tracking throughout its lifecycle, even across job restarts. This helps with log correlation and debugging.
* **Port allocation on resume**: During adoption, discovered ports from `READY URL=` markers are marked as "in-use" in the `PortAllocator` to prevent collisions with new allocations.

---

## 16) Quick class skeletons (just enough to start)

```python
# orchestrator.py
class Orchestrator:
    def __init__(self, cfg): ...
    def run(self):  # main loop
        while True:
            for rec in self.reconcilers: rec.tick()
            time.sleep(1)

# reconciler.py
class Reconciler:
    def tick(self):
        # compute desired vs. actual
        # submit jobs, start tunnels, health-check, register/deregister
        pass
```