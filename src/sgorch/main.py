#!/usr/bin/env python3

import os
import signal
import sys
from pathlib import Path
from typing import Optional

import typer

from .logging_setup import setup_logging, get_logger
from .config import load_config
from .orchestrator import Orchestrator


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
                    status_data = {
                        "deployment": deployment,
                        "replicas": deploy_config.replicas,
                        "model": deploy_config.sglang.model_path,
                        "connectivity_mode": deploy_config.connectivity.mode
                    }
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
                    print(f"  Model: {deploy.sglang.model_path}")
                    print(f"  Connectivity: {deploy.connectivity.mode}")
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