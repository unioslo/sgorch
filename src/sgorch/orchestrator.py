import time
import threading
from typing import Dict, List

from .logging_setup import get_logger
from .config import Config, DeploymentConfig
from .reconciler import Reconciler
from .slurm.rest import SlurmRestAdapter
from .slurm.cli import SlurmCliAdapter
from .slurm.base import ISlurm
from .router.client import RouterClient
from .notify.base import Notifier
from .notify.log_only import LogOnlyNotifier
from .metrics.prometheus import get_metrics
from .state.file_store import FileStateStore


logger = get_logger(__name__)


class Orchestrator:
    """Main orchestrator that manages multiple deployments."""
    
    def __init__(self, config: Config):
        self.config = config
        self.reconcilers: Dict[str, Reconciler] = {}
        self.running = False
        self.threads: List[threading.Thread] = []
        # Initialize state store (file backend; path from config if provided)
        file_path = getattr(self.config.orchestrator.state, 'file_path', None)
        self.state_store = FileStateStore(file_path=file_path)
        
        # Initialize metrics server
        self._setup_metrics()
        
        # Initialize notifier
        self._setup_notifier()
        
        # Initialize reconcilers for each deployment
        self._setup_reconcilers()
        
        logger.info(f"Orchestrator initialized with {len(self.reconcilers)} deployments")
    
    def _setup_metrics(self) -> None:
        """Initialize metrics server."""
        metrics = get_metrics()
        metrics.start_http_server(self.config.orchestrator.metrics)
    
    def _setup_notifier(self) -> None:
        """Initialize notification system."""
        # For now, just use log-only notifier
        # In the future, this would create the appropriate notifier based on config
        self.notifier = LogOnlyNotifier()
        logger.info(f"Notification system initialized: {self.config.orchestrator.notifications.type}")
    
    def _setup_reconcilers(self) -> None:
        """Initialize reconcilers for each deployment."""
        for deploy_config in self.config.deployments:
            try:
                reconciler = self._create_reconciler(deploy_config)
                self.reconcilers[deploy_config.name] = reconciler
                logger.info(f"Reconciler created for deployment: {deploy_config.name}")
            except Exception as e:
                logger.error(f"Failed to create reconciler for {deploy_config.name}: {e}")
                raise
    
    def _create_reconciler(self, deploy_config: DeploymentConfig) -> Reconciler:
        """Create a reconciler for a deployment."""
        # Create SLURM adapter
        slurm = self._create_slurm_adapter(deploy_config)
        
        # Create router client
        router_client = RouterClient(deploy_config.router)

        # Optional: fast-fail if router is unreachable/misconfigured
        try:
            if not router_client.health_check():
                base = deploy_config.router.base_url
                list_ep = deploy_config.router.endpoints.list
                add_ep = deploy_config.router.endpoints.add
                rm_ep = deploy_config.router.endpoints.remove
                auth = deploy_config.router.auth
                hint = ""
                if auth and auth.type == "header":
                    import os
                    token_env = auth.header_value_env
                    if not os.getenv(token_env):
                        hint = f" (missing env {token_env})"
                raise RuntimeError(
                    f"Router liveness check failed for {base}. Verify endpoints: list={list_ep} add={add_ep} remove={rm_ep}{hint}"
                )
        except Exception as e:
            logger.error(f"Router check failed for deployment {deploy_config.name}: {e}")
            # Fail fast to surface misconfiguration early
            raise
        
        # Create reconciler
        return Reconciler(
            deployment_config=deploy_config,
            slurm=slurm,
            router_client=router_client,
            notifier=self.notifier,
            state_store=self.state_store
        )
    
    def _create_slurm_adapter(self, deploy_config: DeploymentConfig) -> ISlurm:
        """Create SLURM adapter based on configuration."""
        prefer = deploy_config.slurm.prefer
        
        if prefer == "rest":
            # Try REST first
            try:
                # Would need to determine REST endpoint and auth from config/env
                # For now, fall back to CLI
                logger.info(f"SLURM REST not implemented, falling back to CLI")
                return SlurmCliAdapter()
            except Exception:
                logger.warning("SLURM REST failed, falling back to CLI")
                return SlurmCliAdapter()
        
        elif prefer == "cli":
            return SlurmCliAdapter()
        
        elif prefer == "auto":
            # Try REST first, fall back to CLI
            try:
                # Would check for slurmrestd availability
                logger.info("SLURM auto mode: using CLI adapter")
                return SlurmCliAdapter()
            except Exception:
                return SlurmCliAdapter()
        
        else:
            raise ValueError(f"Unknown SLURM adapter preference: {prefer}")
    
    def run(self) -> None:
        """Run the orchestrator (blocks until shutdown)."""
        if self.running:
            logger.warning("Orchestrator is already running")
            return
        
        self.running = True
        logger.info("Starting orchestrator main loop")
        
        try:
            # Start reconciler threads
            self._start_reconciler_threads()
            
            # Main monitoring loop
            while self.running:
                try:
                    # Basic health check of reconcilers
                    self._monitor_reconcilers()
                    
                    # Sleep between monitoring cycles
                    time.sleep(5.0)
                    
                except Exception as e:
                    logger.error(f"Error in orchestrator main loop: {e}")
                    time.sleep(1.0)
        
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            self._shutdown()
    
    def _start_reconciler_threads(self) -> None:
        """Start reconciler threads."""
        for name, reconciler in self.reconcilers.items():
            thread = threading.Thread(
                target=self._run_reconciler,
                args=(name, reconciler),
                name=f"reconciler-{name}",
                daemon=True
            )
            thread.start()
            self.threads.append(thread)
            logger.info(f"Started reconciler thread for {name}")
    
    def _run_reconciler(self, name: str, reconciler: Reconciler) -> None:
        """Run a single reconciler in a loop."""
        logger.info(f"Reconciler {name} started")
        
        while self.running:
            try:
                # Run one reconciliation cycle
                reconciler.tick()
                
                # Wait between cycles (1 second by default)
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in reconciler {name}: {e}")
                time.sleep(5.0)  # Back off on errors
        
        logger.info(f"Reconciler {name} stopped")
    
    def _monitor_reconcilers(self) -> None:
        """Monitor reconciler health."""
        # Basic monitoring - check if threads are alive
        for thread in self.threads:
            if not thread.is_alive():
                logger.error(f"Reconciler thread {thread.name} died")
                # In a production system, you might restart the thread here
    
    def shutdown(self) -> None:
        """Shutdown the orchestrator gracefully."""
        if not self.running:
            return
        
        logger.info("Shutting down orchestrator")
        self.running = False
        
        # Shutdown reconcilers
        for name, reconciler in self.reconcilers.items():
            try:
                reconciler.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down reconciler {name}: {e}")
        
        # Wait for threads to finish
        for thread in self.threads:
            try:
                thread.join(timeout=10.0)
                if thread.is_alive():
                    logger.warning(f"Thread {thread.name} did not shutdown cleanly")
            except Exception as e:
                logger.error(f"Error joining thread {thread.name}: {e}")
        
        logger.info("Orchestrator shutdown complete")
    
    def _shutdown(self) -> None:
        """Internal shutdown method."""
        self.shutdown()
