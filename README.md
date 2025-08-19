# SGOrch - SLURM ↔ SGLang Orchestrator

SGOrch is a production-ready orchestrator that manages SGLang worker deployments on SLURM clusters. It automatically handles job submission, health monitoring, SSH tunneling, router registration, and failure recovery with graceful resumption capabilities.

## Architecture Overview

SGOrch follows a microservices-like architecture with clear separation of concerns:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   SGLang Router │◄──►│     SGOrch       │◄──►│  SLURM Cluster  │
│                 │    │   Orchestrator   │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   Prometheus     │
                    │    Metrics       │
                    └──────────────────┘
```

### Core Components

- **Orchestrator**: Main process that manages multiple deployments
- **Reconciler**: Per-deployment controller that maintains desired state
- **SLURM Adapters**: Interface to SLURM (REST API or CLI)
- **Router Client**: Manages worker registration with SGLang router
- **Health Monitor**: Monitors worker health via HTTP probes
- **Tunnel Manager**: Creates and supervises SSH tunnels
- **Port Allocator**: Manages port allocation to avoid conflicts
- **Failure Policies**: Node blacklisting, backoff, circuit breakers
- **Discovery System**: Graceful resumption by adopting existing workers

### Key Features

- **Multi-deployment Management**: Manage multiple SGLang model deployments
- **Automatic Scaling**: Maintains desired replica count per deployment  
- **Health Monitoring**: HTTP health checks with authentication
- **SSH Tunneling**: Both local (-L) and reverse (-R) tunnel modes
- **Failure Recovery**: Node blacklisting, exponential backoff, graceful draining
- **Stateless Resumption**: Automatically adopts existing workers on restart
- **Prometheus Metrics**: Comprehensive monitoring and alerting
- **Structured Logging**: JSON logs for observability
- **Production Ready**: Systemd integration, signal handling, graceful shutdown

## Installation

### Prerequisites

- Python 3.10+
- SLURM cluster access with `squeue`, `sbatch`, `scancel` commands
- SGLang installed in a virtual environment
- Network access between orchestrator VM and compute nodes
- SSH access to compute nodes (for tunneled mode)

### Setup

1. **Clone and install SGOrch**:
```bash
git clone <repository-url> sgorch
cd sgorch
uv venv
source .venv/bin/activate
uv pip install -e .
```

2. **Create log directory**:
```bash
mkdir -p ~/sg-logs
```

3. **Set up environment variables**:
```bash
export ROUTER_TOKEN="your-router-authentication-token"
export WORKER_HEALTH_TOKEN="your-worker-health-token"
```

4. **Configure your deployment** (see Configuration section)

## Configuration

Create a YAML configuration file based on `src/sgorch/examples/config.yaml`:

```yaml
orchestrator:
  metrics:
    enabled: true
    bind: "0.0.0.0" 
    port: 9315
  notifications:
    type: log_only

deployments:
  - name: gpt-oss-20b
    replicas: 2
    connectivity:
      mode: tunneled              # or "direct"
      tunnel_mode: local          # "local" or "reverse" 
      orchestrator_host: "vm.example.com"
      advertise_host: "vm.example.com"
      local_port_range: [30000, 30999]
      ssh:
        user: "your-username"
        opts: ["-o", "ServerAliveInterval=15"]

    router:
      base_url: "http://router.example.com:8080"
      endpoints:
        list: "/workers/list"
        add: "/workers/add"
        remove: "/workers/remove"
      auth:
        type: header
        header_name: "Authorization"
        header_value_env: "ROUTER_TOKEN"

    slurm:
      prefer: auto               # rest|cli|auto
      account: "your-account"
      reservation: "your-reservation"
      partition: "GPUQ"
      gres: "gpu:1"
      constraint: "h100"
      time_limit: "08:00:00"
      cpus_per_task: 24
      mem: "128G"
      log_dir: "/path/to/logs"
      env:
        HF_HOME: "/path/to/hf-cache"

    sglang:
      model_path: "openai/gpt-oss-20b"
      venv_path: "/path/to/sglang/.venv"
      args:
        - "--host"
        - "0.0.0.0"
        - "--port" 
        - "{PORT}"
        - "--reasoning-parser"
        - "gpt-oss"
        - "--context-length"
        - "64000"

    health:
      path: "/v1/health"
      interval_s: 5
      timeout_s: 3
      consecutive_ok_for_ready: 2
      failures_to_unhealthy: 3
      headers:
        Authorization: "${WORKER_HEALTH_TOKEN}"

    policy:
      restart_backoff_s: 60
      deregister_grace_s: 10
      start_grace_period_s: 600
      predrain_seconds_before_walltime: 180
      node_blacklist_cooldown_s: 600
```

### Configuration Sections

- **orchestrator**: Global settings for metrics and notifications
- **deployments**: List of SGLang deployments to manage
- **connectivity**: Network setup (direct vs tunneled access)
- **router**: SGLang router connection and authentication
- **slurm**: SLURM job parameters matching your cluster setup
- **sglang**: Model path, virtual environment, and launch arguments
- **health**: Health check configuration with authentication
- **policy**: Failure handling, backoff, and recovery policies

## Usage

### Command Line Interface

```bash
# Validate configuration
sgorch validate --config config.yaml

# Show deployment status
sgorch status --config config.yaml

# Show specific deployment status
sgorch status --config config.yaml --deployment gpt-oss-20b

# Get JSON output
sgorch status --config config.yaml --json

# Force adoption of existing workers
sgorch adopt --config config.yaml

# Run orchestrator (foreground)
sgorch run --config config.yaml --log-level INFO
```

### Running as a Service

1. **Install systemd service**:
```bash
# Copy and customize the service file
cp src/sgorch/scripts/systemd-user.service ~/.config/systemd/user/sgorch.service

# Edit the service file to set correct paths and environment variables
nano ~/.config/systemd/user/sgorch.service

# Enable and start
systemctl --user daemon-reload
systemctl --user enable sgorch.service
systemctl --user start sgorch.service

# Check status
systemctl --user status sgorch.service

# View logs
journalctl --user -u sgorch.service -f
```

2. **Service file configuration**:
```ini
[Unit]
Description=SGOrch (SLURM-SGLang Orchestrator)
After=network.target

[Service]
Type=simple
ExecStart=/path/to/sgorch/.venv/bin/sgorch run --config /path/to/config.yaml
Restart=always
RestartSec=5
Environment=ROUTER_TOKEN=your-token
Environment=WORKER_HEALTH_TOKEN=your-health-token
WorkingDirectory=/path/to/sgorch

[Install]
WantedBy=default.target
```

## Networking Modes

SGOrch supports two connectivity modes:

### Direct Mode
Workers are directly accessible from the router:
- Router connects directly to `http://compute-node:port`
- No tunneling overhead
- Requires network connectivity between router and compute nodes

### Tunneled Mode (Recommended)
SSH tunnels provide connectivity:

**Local Tunnels (-L)**: Orchestrator creates tunnels from VM to compute nodes
```
Router → VM:local_port → SSH tunnel → ComputeNode:remote_port
```

**Reverse Tunnels (-R)**: Compute nodes create tunnels back to VM
```  
Router → VM:advertise_port ← SSH tunnel ← ComputeNode:remote_port
```

## Monitoring and Observability

### Prometheus Metrics

SGOrch exposes metrics on port 9315 by default:

- `sgorch_workers_desired{deployment}`: Desired worker count
- `sgorch_workers_ready{deployment}`: Ready worker count  
- `sgorch_workers_unhealthy{deployment}`: Unhealthy worker count
- `sgorch_tunnels_up{deployment}`: Active tunnel count
- `sgorch_restarts_total{deployment,reason}`: Worker restart count
- `sgorch_router_errors_total{deployment,operation}`: Router API errors
- `sgorch_slurm_errors_total{deployment,operation}`: SLURM operation errors

### Structured Logging

All logs are in JSON format with contextual fields:
```json
{
  "ts": "2025-01-01T12:00:00Z",
  "level": "INFO", 
  "msg": "Worker started successfully",
  "deployment": "gpt-oss-20b",
  "job_id": "12345",
  "worker_url": "http://node01:8000",
  "event": "worker_start"
}
```

### Health Checks

SGOrch continuously monitors worker health via HTTP probes:
- Configurable endpoints, intervals, and timeouts
- Authentication header support
- Consecutive success/failure thresholds
- Automatic deregistration of unhealthy workers

## Failure Handling

### Node Blacklisting
Nodes experiencing repeated failures are temporarily blacklisted:
- Configurable cooldown period
- Prevents job submission to problematic nodes
- Automatic removal when cooldown expires

### Exponential Backoff
Failed operations use exponential backoff with jitter:
- Prevents thundering herd problems
- Configurable base delay and maximum delay
- Applied to job submissions and router operations

### Graceful Draining
Workers are gracefully drained before replacement:
- Remove from router first
- Wait for grace period
- Cancel SLURM job
- Submit replacement

### Walltime Management
Workers are proactively replaced before SLURM time limits:
- Monitor remaining job time
- Pre-drain and submit replacement
- Cancel old job only after new worker is ready

## Troubleshooting

### Common Issues

1. **SLURM job submission failures**:
   - Check partition, account, and resource requirements
   - Verify SLURM commands work manually: `squeue`, `sbatch --test-only`

2. **Router connection errors**:
   - Verify router URL and authentication token
   - Check network connectivity from orchestrator to router

3. **SSH tunnel failures**:
   - Ensure SSH key authentication is set up
   - Test manual SSH connection to compute nodes
   - Check SSH options in configuration

4. **Worker health check failures**:
   - Verify health endpoint and authentication
   - Check SGLang server startup logs
   - Ensure model loading completes successfully

### Debug Mode

Run with debug logging for detailed information:
```bash
sgorch run --config config.yaml --log-level DEBUG
```

### Manual Testing

Test individual components:

```bash
# Test SLURM commands
squeue -u $USER
sbatch --test-only test-script.sh

# Test router connectivity  
curl -H "Authorization: $ROUTER_TOKEN" http://router:8080/workers/list

# Test worker health
curl -H "Authorization: $WORKER_HEALTH_TOKEN" http://worker:8000/v1/health

# Test SSH connectivity
ssh compute-node hostname
```

## Development

### Project Structure

```
sgorch/
├── src/sgorch/
│   ├── config.py              # Configuration management
│   ├── main.py                # CLI entry point  
│   ├── orchestrator.py        # Main orchestrator process
│   ├── reconciler.py          # Per-deployment reconciler
│   ├── logging_setup.py       # Structured logging
│   ├── slurm/                 # SLURM integration
│   │   ├── base.py            # SLURM interface
│   │   ├── cli.py             # CLI adapter
│   │   ├── rest.py            # REST API adapter
│   │   └── sbatch_templates.py # Job script templates
│   ├── router/                # SGLang router integration
│   ├── health/                # Health monitoring
│   ├── net/                   # Networking and tunnels
│   ├── policy/                # Failure policies
│   ├── discover/              # Worker discovery/adoption
│   ├── metrics/               # Prometheus metrics
│   ├── notify/                # Notification system
│   └── util/                  # Utilities
├── examples/
│   └── config.yaml            # Example configuration
├── scripts/
│   └── systemd-user.service   # Systemd service file
└── tests/                     # Test suite
```

### Adding Features

1. **New SLURM Adapter**: Implement `ISlurm` interface in `slurm/`
2. **New Notification Backend**: Implement `Notifier` interface in `notify/`
3. **New Metrics**: Add to `metrics/prometheus.py` 
4. **New Health Checks**: Extend `health/http_probe.py`

### Testing

```bash
# Run tests
python -m pytest tests/

# Validate configuration
sgorch validate --config examples/config.yaml

# Dry run adoption
sgorch adopt --config config.yaml --dry-run
```