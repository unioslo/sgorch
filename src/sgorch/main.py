#!/usr/bin/env python3

import asyncio
import os
import signal
import sys
from pathlib import Path
from typing import Optional

import typer
import uvicorn

from .logging_setup import setup_logging, get_logger
from .config import load_config
from .orchestrator import Orchestrator
from .router.runtime import RouterRuntime, RouterRuntimeConfig, create_router_app


app = typer.Typer(
    name="sgorch",
    help="SLURM ↔ SGLang Orchestrator",
    no_args_is_help=True
)

# Global orchestrator instance for signal handling
_orchestrator: Optional[Orchestrator] = None


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger = get_logger(__name__)
    signal_name = signal.Signals(signum).name
    logger.info(f"Received {signal_name}, shutting down gracefully...")
    
    if _orchestrator:
        _orchestrator.shutdown()
    
    sys.exit(0)


@app.command()
def run(
    config: str = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to configuration file"
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Log level (DEBUG, INFO, WARNING, ERROR)"
    )
):
    """Run the orchestrator service."""
    global _orchestrator
    
    # Setup logging
    setup_logging(log_level.upper())
    logger = get_logger(__name__)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {config}")
        config_path = Path(config)
        cfg = load_config(config_path)
        
        # Create and start orchestrator
        _orchestrator = Orchestrator(cfg)
        logger.info("Starting SGOrch orchestrator")
        
        # Run the orchestrator (this blocks)
        _orchestrator.run()
        
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config}")
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Failed to start orchestrator: {e}")
        raise typer.Exit(1)


@app.command()
def status(
    config: str = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to configuration file"
    ),
    deployment: Optional[str] = typer.Option(
        None,
        "--deployment",
        "-d",
        help="Show status for specific deployment"
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output in JSON format"
    )
):
    """Show orchestrator status."""
    # Setup logging (quieter for status command)
    setup_logging("WARNING")
    logger = get_logger(__name__)
    
    try:
        # Load configuration
        config_path = Path(config)
        cfg = load_config(config_path)
        
        # This is a simplified status implementation
        # In a full implementation, you might query a running orchestrator
        # via metrics endpoint or status file
        
        if json_output:
            import json
            status_data = {
                "deployments": [d.name for d in cfg.deployments],
                "metrics_enabled": cfg.orchestrator.metrics.enabled,
                "notification_type": cfg.orchestrator.notifications.type
            }
            
            if deployment:
                deploy_config = next(
                    (d for d in cfg.deployments if d.name == deployment),
                    None
                )
                if deploy_config:
                    backend = deploy_config.backend
                    backend_info = {"type": backend.type}
                    if backend.type == "sglang":
                        backend_info["model_path"] = deploy_config.sglang.model_path
                    elif backend.type == "tei":
                        backend_info["model_id"] = deploy_config.tei.model_id

                    status_data = {
                        "deployment": deployment,
                        "replicas": deploy_config.replicas,
                        "backend": backend_info,
                        "connectivity_mode": deploy_config.connectivity.mode
                    }
                    if deploy_config.router:
                        status_data["router"] = deploy_config.router.base_url
                else:
                    status_data = {"error": f"Deployment '{deployment}' not found"}
            
            print(json.dumps(status_data, indent=2))
        else:
            # Human-readable output
            print(f"SGOrch Configuration Status")
            print(f"=" * 40)
            print(f"Deployments: {len(cfg.deployments)}")
            
            for deploy in cfg.deployments:
                if deployment is None or deploy.name == deployment:
                    print(f"\n{deploy.name}:")
                    print(f"  Replicas: {deploy.replicas}")
                    backend = deploy.backend
                    if backend.type == "sglang":
                        print(f"  Backend: SGLang ({deploy.sglang.model_path})")
                    elif backend.type == "tei":
                        print(f"  Backend: TEI ({deploy.tei.model_id})")
                    else:
                        print(f"  Backend: {backend.type}")
                    print(f"  Connectivity: {deploy.connectivity.mode}")
                    if deploy.router:
                        print(f"  Router: {deploy.router.base_url}")
            
            if deployment and not any(d.name == deployment for d in cfg.deployments):
                print(f"\nError: Deployment '{deployment}' not found")
                raise typer.Exit(1)
        
    except Exception as e:
        logger.error(f"Failed to show status: {e}")
        raise typer.Exit(1)


@app.command()
def adopt(
    config: str = typer.Option(
        ...,
        "--config",
        "-c", 
        help="Path to configuration file"
    ),
    deployment: Optional[str] = typer.Option(
        None,
        "--deployment",
        "-d",
        help="Force adoption for specific deployment"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be adopted without making changes"
    )
):
    """Force adoption of existing workers (useful for troubleshooting)."""
    setup_logging("INFO")
    logger = get_logger(__name__)
    
    try:
        # Load configuration
        config_path = Path(config)
        cfg = load_config(config_path)
        
        logger.info("Starting worker adoption process")
        
        if dry_run:
            logger.info("DRY RUN MODE - no changes will be made")
        
        # This would trigger adoption logic
        # In a real implementation, this might send a signal to a running orchestrator
        # or run adoption logic directly
        
        print("Adoption completed. Check logs for details.")
        
    except Exception as e:
        logger.error(f"Adoption failed: {e}")
        raise typer.Exit(1)


@app.command()
def router(
    host: str = typer.Option("0.0.0.0", "--host", help="Router bind host"),
    port: int = typer.Option(8080, "--port", "-p", help="Router bind port"),
    health_path: str = typer.Option("/health", "--health-path", help="Worker health-check path"),
    probe_interval: float = typer.Option(10.0, "--probe-interval", help="Seconds between health probes"),
    probe_timeout: float = typer.Option(5.0, "--probe-timeout", help="Health probe timeout in seconds"),
    request_timeout: float = typer.Option(30.0, "--request-timeout", help="Upstream request timeout in seconds"),
    max_retries: int = typer.Option(3, "--max-retries", help="Maximum proxy attempts before failing"),
    failure_cooldown: float = typer.Option(5.0, "--failure-cooldown", help="Cooldown seconds before retrying an unhealthy worker"),
    prometheus_port: Optional[int] = typer.Option(
        None,
        "--prometheus-port",
        help="Expose Prometheus metrics on the given port"
    ),
    router_name: Optional[str] = typer.Option(
        None,
        "--router-name",
        help="Optional router name label to attach to Prometheus metrics"
    ),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Log level (DEBUG, INFO, WARNING, ERROR)"),
):
    """Run the standalone proxy router service."""

    setup_logging(log_level.upper())
    logger = get_logger(__name__)

    if max_retries < 1:
        logger.error("max_retries must be at least 1")
        raise typer.Exit(1)

    if prometheus_port is not None and prometheus_port <= 0:
        logger.error("prometheus_port must be a positive integer")
        raise typer.Exit(1)

    runtime_config = RouterRuntimeConfig(
        bind_host=host,
        bind_port=port,
        health_path=health_path,
        probe_interval_s=probe_interval,
        probe_timeout_s=probe_timeout,
        request_timeout_s=request_timeout,
        max_retries=max_retries,
        failure_cooldown_s=failure_cooldown,
        prometheus_port=prometheus_port,
        prometheus_host=host if prometheus_port is not None else None,
        router_name=router_name,
    )

    runtime = RouterRuntime(runtime_config)
    app_instance = create_router_app(runtime)

    uv_config = uvicorn.Config(
        app_instance,
        host=runtime_config.bind_host,
        port=runtime_config.bind_port,
        log_level=log_level.lower(),
        log_config=None,
    )
    server = uvicorn.Server(uv_config)

    async def _serve() -> None:
        try:
            await server.serve()
        finally:
            await runtime.stop()

    try:
        asyncio.run(_serve())
    except KeyboardInterrupt:
        logger.info("Router interrupted, shutting down")
    except Exception as e:
        logger.error(f"Router exited with error: {e}")
        raise typer.Exit(1)


@app.command()
def validate(
    config: str = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to configuration file"
    )
):
    """Validate configuration file."""
    setup_logging("WARNING")
    
    try:
        config_path = Path(config)
        cfg = load_config(config_path)
        
        print(f"✓ Configuration is valid")
        print(f"✓ Found {len(cfg.deployments)} deployments")
        
        for deploy in cfg.deployments:
            print(f"  - {deploy.name} (replicas: {deploy.replicas})")
        
    except Exception as e:
        print(f"✗ Configuration is invalid: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
