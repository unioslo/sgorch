import time
import uuid
from typing import Dict, Set, Optional
from dataclasses import dataclass

from .logging_setup import get_logger
from .config import DeploymentConfig
from .slurm.base import ISlurm, SubmitSpec
from .slurm.sbatch_templates import render_sbatch_script
from .router.client import RouterClient
from .notify.base import Notifier
from .health.http_probe import HealthMonitor, HealthStatus
from .net.ports import PortAllocator
from .net.tunnel import SupervisedTunnelManager, TunnelSpec
from .policy.failure import NodeBlacklist, FailureTracker, RestartPolicy
from .discover.adopt import AdoptionManager
from .metrics.prometheus import get_metrics


logger = get_logger(__name__)


@dataclass
class WorkerState:
    """State of a managed worker."""
    job_id: str
    instance_uuid: str
    node: Optional[str] = None
    remote_port: Optional[int] = None
    advertise_port: Optional[int] = None
    worker_url: Optional[str] = None
    advertised_url: Optional[str] = None
    health_status: HealthStatus = HealthStatus.UNKNOWN
    last_seen: Optional[float] = None
    submitted_at: float = 0
    
    def is_healthy(self) -> bool:
        return self.health_status == HealthStatus.HEALTHY
    
    def is_unhealthy(self) -> bool:
        return self.health_status == HealthStatus.UNHEALTHY


class Reconciler:
    """Reconciles desired vs actual state for a single deployment."""
    
    def __init__(
        self,
        deployment_config: DeploymentConfig,
        slurm: ISlurm,
        router_client: RouterClient,
        notifier: Notifier
    ):
        self.config = deployment_config
        self.slurm = slurm
        self.router_client = router_client
        self.notifier = notifier
        self.logger = logger.bind(deployment=deployment_config.name)
        
        # State management
        self.workers: Dict[str, WorkerState] = {}  # job_id -> WorkerState
        self.running = True
        
        # Components
        self.port_allocator = PortAllocator(deployment_config.connectivity.local_port_range)
        self.tunnel_manager = SupervisedTunnelManager()
        self.health_monitor = HealthMonitor()
        
        # Policies
        self.node_blacklist = NodeBlacklist(deployment_config.policy.node_blacklist_cooldown_s)
        self.failure_tracker = FailureTracker(self.node_blacklist)
        self.restart_policy = RestartPolicy(
            restart_backoff_seconds=deployment_config.policy.restart_backoff_s,
            deregister_grace_seconds=deployment_config.policy.deregister_grace_s
        )
        
        # Metrics
        self.metrics = get_metrics()
        
        # Perform initial adoption
        self._adopt_existing_workers()
        
        self.logger.info("Reconciler initialized")
    
    def _adopt_existing_workers(self) -> None:
        """Adopt existing workers during startup."""
        try:
            adoption_manager = AdoptionManager(
                self.config,
                self.slurm,
                self.router_client,
                self.port_allocator
            )
            
            adopted = adoption_manager.adopt_existing_workers()
            
            # Convert adopted workers to our state format
            for job_id, discovered in adopted.items():
                worker = WorkerState(
                    job_id=job_id,
                    instance_uuid=discovered.instance_uuid,
                    node=discovered.node,
                    remote_port=discovered.remote_port,
                    worker_url=discovered.worker_url,
                    last_seen=time.time(),
                    submitted_at=time.time()
                )
                
                self.workers[job_id] = worker
                
                # Set up health monitoring
                if worker.worker_url:
                    self.health_monitor.add_worker(worker.worker_url, self.config.health)
            
            # Reconcile router state
            adoption_manager.reconcile_router_state(adopted)
            
            self.logger.info(f"Adopted {len(adopted)} existing workers")
            
        except Exception as e:
            self.logger.error(f"Failed to adopt existing workers: {e}")
    
    def tick(self) -> None:
        """Run one reconciliation cycle."""
        if not self.running:
            return
        
        start_time = time.time()
        
        try:
            # Update worker states from SLURM
            self._update_worker_states()
            
            # Perform health checks
            self._check_worker_health()
            
            # Reconcile with desired state
            self._reconcile_workers()
            
            # Update tunnels
            self._manage_tunnels()
            
            # Update router registrations
            self._reconcile_router()
            
            # Handle failed workers
            self._handle_failed_workers()
            
            # Update metrics
            self._update_metrics()
            
            # Record successful reconciliation
            duration = time.time() - start_time
            self.metrics.record_reconcile_duration(self.config.name, duration)
            
        except Exception as e:
            self.logger.error(f"Error in reconciliation cycle: {e}")
    
    def _update_worker_states(self) -> None:
        """Update worker states from SLURM."""
        # Get all our jobs from SLURM
        job_prefix = f"sgl-{self.config.name}-"
        slurm_jobs = self.slurm.list_jobs(job_prefix)
        
        # Update existing workers
        for worker in list(self.workers.values()):
            slurm_job = next((j for j in slurm_jobs if j.job_id == worker.job_id), None)
            
            if slurm_job:
                # Update from SLURM info
                if slurm_job.state == "RUNNING" and not worker.node:
                    worker.node = slurm_job.node
                
                # Check for failures
                if slurm_job.state in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"):
                    self.logger.warning(f"Worker {worker.job_id} failed with state {slurm_job.state}")
                    self.failure_tracker.record_failure(
                        reason=slurm_job.state,
                        details=f"SLURM job failed",
                        node=worker.node
                    )
                    
                    # Remove failed worker
                    self._remove_worker(worker.job_id)
            else:
                # Job no longer exists in SLURM
                self.logger.warning(f"Worker {worker.job_id} no longer exists in SLURM")
                self._remove_worker(worker.job_id)
    
    def _check_worker_health(self) -> None:
        """Perform health checks on workers."""
        health_results = self.health_monitor.probe_all_due()
        
        for worker_url, result in health_results.items():
            # Find the worker by URL
            worker = next((w for w in self.workers.values() if w.worker_url == worker_url), None)
            if not worker:
                continue
            
            # Update health status
            probe = self.health_monitor.probes.get(worker_url)
            if probe:
                worker.health_status = probe.get_status()
                
                if result.status == HealthStatus.HEALTHY:
                    worker.last_seen = time.time()
                
                # Record metrics
                self.metrics.record_health_check(
                    self.config.name,
                    result.status == HealthStatus.HEALTHY,
                    result.response_time_ms / 1000.0 if result.response_time_ms else 0.0
                )
    
    def _reconcile_workers(self) -> None:
        """Reconcile actual vs desired worker count."""
        desired = self.config.replicas
        current_healthy = len([w for w in self.workers.values() if w.is_healthy()])
        current_total = len(self.workers)
        
        self.logger.debug(f"Reconciling: desired={desired}, healthy={current_healthy}, total={current_total}")
        
        # Start new workers if needed
        if current_total < desired:
            needed = desired - current_total
            for i in range(needed):
                if self._can_start_worker():
                    self._start_worker()
        
        # We don't automatically scale down - only remove failed workers
    
    def _can_start_worker(self) -> bool:
        """Check if we can start a new worker."""
        # Check if we have available ports
        if self.port_allocator.get_available_count() < 2:
            self.logger.warning("No available ports for new worker")
            return False
        
        # Add other checks as needed (quotas, etc.)
        return True
    
    def _start_worker(self) -> None:
        """Start a new worker."""
        try:
            # Generate unique instance ID
            instance_uuid = str(uuid.uuid4())
            instance_idx = len(self.workers)
            
            # Allocate ports
            remote_port = self.port_allocator.allocate_port()
            advertise_port = remote_port  # For direct mode
            
            if self.config.connectivity.mode == "tunneled":
                advertise_port = self.port_allocator.allocate_port()
            
            # Create SLURM job specification
            spec = self._create_job_spec(instance_idx, instance_uuid, remote_port)
            
            # Submit job
            job_id = self.slurm.submit(spec)
            
            # Create worker state
            worker = WorkerState(
                job_id=job_id,
                instance_uuid=instance_uuid,
                remote_port=remote_port,
                advertise_port=advertise_port,
                submitted_at=time.time()
            )
            
            self.workers[job_id] = worker
            
            self.logger.info(f"Started worker: job_id={job_id}, uuid={instance_uuid}")
            self.metrics.record_job_submitted(self.config.name)
            
        except Exception as e:
            self.logger.error(f"Failed to start worker: {e}")
            # Release allocated ports on failure
            if 'remote_port' in locals():
                self.port_allocator.release_port(remote_port)
            if 'advertise_port' in locals() and advertise_port != remote_port:
                self.port_allocator.release_port(advertise_port)
    
    def _create_job_spec(self, instance_idx: int, instance_uuid: str, remote_port: int) -> SubmitSpec:
        """Create SLURM job specification."""
        job_name = f"sgl-{self.config.name}-{instance_idx}"
        
        # Render job script
        script = render_sbatch_script(
            deploy_name=self.config.name,
            instance_idx=instance_idx,
            instance_uuid=instance_uuid,
            account=self.config.slurm.account,
            partition=self.config.slurm.partition,
            gres=self.config.slurm.gres,
            cpus=self.config.slurm.cpus_per_task,
            mem=self.config.slurm.mem,
            time_limit=self.config.slurm.time_limit,
            reservation=self.config.slurm.reservation,
            qos=self.config.slurm.qos,
            constraint=self.config.slurm.constraint,
            log_dir=self.config.slurm.log_dir,
            env_vars=self.config.slurm.env,
            model_path=self.config.sglang.model_path,
            remote_port=remote_port,
            sglang_args=self.config.sglang.args,
            health_path=self.config.health.path,
            sglang_venv_path=self.config.sglang.venv_path,
            sbatch_extra=self.config.slurm.sbatch_extra
        )
        
        return SubmitSpec(
            name=job_name,
            account=self.config.slurm.account,
            reservation=self.config.slurm.reservation,
            partition=self.config.slurm.partition,
            qos=self.config.slurm.qos,
            gres=self.config.slurm.gres,
            constraint=self.config.slurm.constraint,
            time_limit=self.config.slurm.time_limit,
            cpus_per_task=self.config.slurm.cpus_per_task,
            mem=self.config.slurm.mem,
            env=self.config.slurm.env,
            stdout=f"{self.config.slurm.log_dir}/{job_name}_%j.out",
            stderr=f"{self.config.slurm.log_dir}/{job_name}_%j.err",
            script=script
        )
    
    def _manage_tunnels(self) -> None:
        """Manage SSH tunnels for workers."""
        if self.config.connectivity.mode != "tunneled":
            return
        
        for worker in self.workers.values():
            if not worker.node or not worker.remote_port or not worker.advertise_port:
                continue
            
            # Create tunnel spec
            tunnel_spec = TunnelSpec(
                mode=self.config.connectivity.tunnel_mode,
                orchestrator_host=self.config.connectivity.orchestrator_host,
                advertise_host=self.config.connectivity.advertise_host,
                advertise_port=worker.advertise_port,
                remote_host=worker.node,
                remote_port=worker.remote_port,
                ssh_user=self.config.connectivity.ssh.user,
                ssh_opts=self.config.connectivity.ssh.opts
            )
            
            # Ensure tunnel exists
            tunnel_key = f"{self.config.name}:{worker.job_id}"
            advertised_url = self.tunnel_manager.ensure(tunnel_key, tunnel_spec)
            worker.advertised_url = advertised_url
    
    def _reconcile_router(self) -> None:
        """Reconcile router registrations."""
        try:
            # Get current router workers
            router_workers = self.router_client.list()
            
            # Register healthy workers
            for worker in self.workers.values():
                if worker.is_healthy() and worker.advertised_url:
                    if worker.advertised_url not in router_workers:
                        try:
                            self.router_client.add(worker.advertised_url)
                            self.logger.info(f"Registered worker with router: {worker.advertised_url}")
                            self.metrics.record_router_operation(
                                self.config.name, "add", True
                            )
                        except Exception as e:
                            self.logger.error(f"Failed to register worker: {e}")
                            self.metrics.record_router_operation(
                                self.config.name, "add", False
                            )
            
            # Deregister unhealthy workers
            our_urls = {w.advertised_url for w in self.workers.values() if w.advertised_url}
            stale_urls = router_workers - our_urls
            
            for url in stale_urls:
                try:
                    self.router_client.remove(url)
                    self.logger.info(f"Deregistered stale worker: {url}")
                    self.metrics.record_router_operation(
                        self.config.name, "remove", True
                    )
                except Exception as e:
                    self.logger.error(f"Failed to deregister worker: {e}")
                    self.metrics.record_router_operation(
                        self.config.name, "remove", False
                    )
        
        except Exception as e:
            self.logger.error(f"Error reconciling router state: {e}")
    
    def _handle_failed_workers(self) -> None:
        """Handle failed workers."""
        for worker in list(self.workers.values()):
            if worker.is_unhealthy():
                self.logger.warning(f"Handling failed worker: {worker.job_id}")
                
                # Deregister from router first
                if worker.advertised_url:
                    try:
                        self.router_client.remove(worker.advertised_url)
                    except Exception as e:
                        self.logger.error(f"Failed to deregister failed worker: {e}")
                
                # Wait grace period
                time.sleep(self.config.policy.deregister_grace_s)
                
                # Cancel SLURM job
                try:
                    self.slurm.cancel(worker.job_id)
                except Exception as e:
                    self.logger.error(f"Failed to cancel job {worker.job_id}: {e}")
                
                # Remove worker
                self._remove_worker(worker.job_id)
                
                # Record restart
                self.metrics.record_restart(self.config.name, "unhealthy")
    
    def _remove_worker(self, job_id: str) -> None:
        """Remove a worker and clean up resources."""
        worker = self.workers.get(job_id)
        if not worker:
            return
        
        # Clean up health monitoring
        if worker.worker_url:
            self.health_monitor.remove_worker(worker.worker_url)
        
        # Clean up tunnels
        tunnel_key = f"{self.config.name}:{job_id}"
        self.tunnel_manager.drop(tunnel_key)
        
        # Release ports
        if worker.remote_port:
            self.port_allocator.release_port(worker.remote_port)
        if worker.advertise_port and worker.advertise_port != worker.remote_port:
            self.port_allocator.release_port(worker.advertise_port)
        
        # Remove from our state
        del self.workers[job_id]
        
        self.logger.info(f"Removed worker: {job_id}")
    
    def _update_metrics(self) -> None:
        """Update Prometheus metrics."""
        healthy = len([w for w in self.workers.values() if w.is_healthy()])
        starting = len([w for w in self.workers.values() if w.health_status == HealthStatus.STARTING])
        unhealthy = len([w for w in self.workers.values() if w.is_unhealthy()])
        
        self.metrics.update_worker_counts(
            self.config.name,
            desired=self.config.replicas,
            ready=healthy,
            starting=starting,
            unhealthy=unhealthy
        )
        
        # Update other metrics
        tunnel_count = len([w for w in self.workers.values() if w.advertised_url])
        self.metrics.update_tunnel_count(self.config.name, tunnel_count)
        
        allocated_ports = len(self.port_allocator.get_allocated_ports())
        available_ports = self.port_allocator.get_available_count()
        self.metrics.update_port_counts(self.config.name, allocated_ports, available_ports)
        
        blacklisted_nodes = len(self.node_blacklist.get_blacklisted_nodes())
        self.metrics.update_blacklisted_nodes(self.config.name, blacklisted_nodes)
    
    def shutdown(self) -> None:
        """Shutdown the reconciler gracefully."""
        self.logger.info("Shutting down reconciler")
        self.running = False
        
        # Clean up tunnels
        self.tunnel_manager.shutdown()
        
        self.logger.info("Reconciler shutdown complete")