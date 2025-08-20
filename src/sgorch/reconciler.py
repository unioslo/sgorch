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
from .state.base import StateStore, DeploymentSnapshot, SerializableWorker
from .config_hash import ConfigSnapshot, WorkerGeneration
from .replacement_engine import ReplacementEngine
from .walltime_manager import WalltimeManager


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
    
    # Configuration generation tracking
    generation: Optional[WorkerGeneration] = None
    replacement_reason: Optional[str] = None  # Why this worker needs replacement
    replacement_scheduled: bool = False  # Is replacement in progress
    
    def is_healthy(self) -> bool:
        return self.health_status == HealthStatus.HEALTHY
    
    def is_unhealthy(self) -> bool:
        return self.health_status == HealthStatus.UNHEALTHY
    
    def needs_replacement(self, current_snapshot: ConfigSnapshot) -> bool:
        """Check if this worker needs replacement due to config changes."""
        if not self.generation:
            # Legacy worker without generation info - assume needs replacement
            return True
        return self.generation.needs_replacement(current_snapshot)
    
    def get_replacement_reason(self, current_snapshot: ConfigSnapshot) -> Optional[str]:
        """Get human-readable reason why worker needs replacement."""
        if not self.generation:
            return "legacy_worker_no_generation_info"
        
        if not self.generation.needs_replacement(current_snapshot):
            return None
            
        old_snap = self.generation.config_snapshot
        
        # Check specific differences
        reasons = []
        if old_snap.gres != current_snapshot.gres:
            reasons.append(f"gres: {old_snap.gres} → {current_snapshot.gres}")
        if old_snap.cpus_per_task != current_snapshot.cpus_per_task:
            reasons.append(f"cpus: {old_snap.cpus_per_task} → {current_snapshot.cpus_per_task}")
        if old_snap.mem != current_snapshot.mem:
            reasons.append(f"mem: {old_snap.mem} → {current_snapshot.mem}")
        if old_snap.model_path != current_snapshot.model_path:
            reasons.append(f"model: {old_snap.model_path} → {current_snapshot.model_path}")
        if old_snap.sglang_args != current_snapshot.sglang_args:
            reasons.append("sglang_args_changed")
        if old_snap.env != current_snapshot.env:
            reasons.append("environment_variables_changed")
            
        return "; ".join(reasons) if reasons else "config_changed"


class Reconciler:
    """Reconciles desired vs actual state for a single deployment."""
    
    def __init__(
        self,
        deployment_config: DeploymentConfig,
        slurm: ISlurm,
        router_client: RouterClient,
        notifier: Notifier,
        state_store: StateStore
    ):
        self.config = deployment_config
        self.slurm = slurm
        self.router_client = router_client
        self.notifier = notifier
        self.state_store = state_store
        self.logger = logger.bind(deployment=deployment_config.name)
        
        # State management
        self.workers: Dict[str, WorkerState] = {}  # job_id -> WorkerState
        self.running = True
        
        # Configuration tracking
        self.current_config_snapshot = ConfigSnapshot.from_deployment_config(deployment_config)
        self.generation_counter = 0  # Increment for each new generation
        self.active_replacements = 0  # Track concurrent replacements
        
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
        
        # Replacement engine
        self.replacement_engine = ReplacementEngine(deployment_config.name)
        
        # Walltime manager
        self.walltime_manager = WalltimeManager(
            deployment_config.name,
            deployment_config.policy.predrain_seconds_before_walltime
        )
        
        # Load persisted state first, then adopt any existing workers
        self._load_persisted_state()

        # Perform initial adoption
        self._adopt_existing_workers()
        
        self.logger.info("Reconciler initialized")

    def _load_persisted_state(self) -> None:
        """Load previously persisted state for this deployment, if any."""
        try:
            snapshot = self.state_store.load_deployment(self.config.name)
            if not snapshot:
                return

            # Seed workers from snapshot
            for sw in snapshot.workers:
                if sw.job_id in self.workers:
                    continue
                worker = WorkerState(
                    job_id=sw.job_id,
                    instance_uuid=sw.instance_uuid,
                    node=sw.node,
                    remote_port=sw.remote_port,
                    advertise_port=sw.advertise_port,
                    worker_url=sw.worker_url,
                    advertised_url=sw.advertised_url,
                    health_status=HealthStatus(sw.health_status) if sw.health_status in HealthStatus._value2member_map_ else HealthStatus.UNKNOWN,
                    last_seen=sw.last_seen,
                    submitted_at=sw.submitted_at,
                )
                self.workers[worker.job_id] = worker

            # Mark ports as in-use (prevents collisions on restart)
            for p in snapshot.allocated_ports:
                self.port_allocator.mark_in_use(p)

            # Recreate health monitoring for known URLs
            for w in list(self.workers.values()):
                if w.worker_url and w.worker_url not in self.health_monitor.probes:
                    self.health_monitor.add_worker(w.worker_url, self.config.health)

            self.logger.info(f"Loaded persisted state: {len(snapshot.workers)} workers")
        except Exception as e:
            self.logger.warning(f"Failed to load persisted state: {e}")
    
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
            
            # Handle configuration-based replacements
            self._handle_config_replacements()
            
            # Update metrics
            self._update_metrics()
            
            # Record successful reconciliation
            duration = time.time() - start_time
            self.metrics.record_reconcile_duration(self.config.name, duration)
            # Persist state at end of successful cycle
            self._persist_state()
            
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
                    self.logger.info(f"Worker {worker.job_id} started running on {worker.node}")
                
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
                # Job no longer exists in SLURM - but check if enough time has passed
                # to avoid timing issues where jobs haven't appeared in squeue yet
                time_since_submit = time.time() - worker.submitted_at
                grace_period = 60  # 60 seconds grace period
                
                if time_since_submit < grace_period:
                    # Too early to declare job missing - it might still be starting
                    self.logger.debug(f"Worker {worker.job_id} not found in SLURM, but within grace period ({time_since_submit:.1f}s < {grace_period}s)")
                    continue
                
                # Enough time has passed - job is genuinely missing
                self.logger.warning(f"Worker {worker.job_id} no longer exists in SLURM after {time_since_submit:.1f}s")
                self._remove_worker(worker.job_id)
    
    def _check_worker_health(self) -> None:
        """Perform health checks on workers."""
        # Only perform health checks if we have workers with URLs
        workers_with_urls = [w for w in self.workers.values() if w.worker_url]
        if not workers_with_urls:
            return
            
        health_results = self.health_monitor.probe_all_due()
        
        for worker_url, result in health_results.items():
            # Find the worker by URL
            worker = next((w for w in self.workers.values() if w.worker_url == worker_url), None)
            if not worker:
                continue
            
            # Update health status
            probe = self.health_monitor.probes.get(worker_url)
            if probe:
                old_status = worker.health_status
                worker.health_status = probe.get_status()
                
                # Log status changes
                if old_status != worker.health_status:
                    self.logger.info(
                        f"Worker {worker.job_id} health changed: {old_status} -> {worker.health_status}",
                        worker_url=worker.worker_url
                    )
                
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
            
            # Create worker generation info
            self.generation_counter += 1
            generation = WorkerGeneration(
                config_hash=self.current_config_snapshot.compute_hash(),
                config_snapshot=self.current_config_snapshot,
                created_at=time.time(),
                generation_id=f"{self.config.name}-gen-{self.generation_counter}"
            )
            
            # Create worker state
            worker = WorkerState(
                job_id=job_id,
                instance_uuid=instance_uuid,
                remote_port=remote_port,
                advertise_port=advertise_port,
                submitted_at=time.time(),
                generation=generation
            )
            
            self.workers[job_id] = worker
            
            # Register with walltime manager
            self.walltime_manager.register_worker(
                job_id=job_id,
                time_limit=self.config.slurm.time_limit,
                submitted_at=worker.submitted_at
            )
            
            # Log where to find the SGLang server logs
            sglang_log_path = f"{self.config.slurm.log_dir}/sglang_{self.config.name}_{job_id}.log"
            self.logger.info(f"Started worker: job_id={job_id}, uuid={instance_uuid}")
            self.logger.info(f"SGLang server logs: {sglang_log_path}")
            self.logger.info(f"Monitor SGLang startup: tail -f {sglang_log_path}")
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
            # For direct mode, set worker URLs directly
            for worker in self.workers.values():
                if worker.node and worker.remote_port and not worker.worker_url:
                    worker.worker_url = f"http://{worker.node}:{worker.remote_port}"
                    worker.advertised_url = worker.worker_url
                    
                    # Start health monitoring for new workers
                    if worker.worker_url not in self.health_monitor.probes:
                        self.health_monitor.add_worker(worker.worker_url, self.config.health)
                        self.logger.info(f"Started health monitoring for worker: {worker.worker_url}")
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
            
            # Set worker URL for health monitoring (tunnel endpoint)
            if not worker.worker_url and advertised_url:
                worker.worker_url = advertised_url
                
                # Start health monitoring for new workers
                if worker.worker_url not in self.health_monitor.probes:
                    self.health_monitor.add_worker(worker.worker_url, self.config.health)
                    self.logger.info(f"Started health monitoring for worker: {worker.worker_url}")
    
    def _reconcile_router(self) -> None:
        """Reconcile router registrations."""
        # Collect workers that should be registered (healthy with advertised URLs)
        healthy_workers = [w for w in self.workers.values() 
                          if w.is_healthy() and w.advertised_url]
        
        # If we have no healthy workers and no workers to potentially deregister, skip router operations
        if not healthy_workers and not self.workers:
            return
            
        try:
            # Only call list() if we have workers to potentially register/deregister
            need_router_check = bool(healthy_workers)
            
            if need_router_check:
                router_workers = self.router_client.list()
                
                # Register healthy workers that aren't registered yet
                for worker in healthy_workers:
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
                
                # Deregister stale workers (in router but not in our healthy set)
                our_healthy_urls = {w.advertised_url for w in healthy_workers}
                stale_urls = router_workers - our_healthy_urls

                for url in stale_urls:
                    # Be conservative: only deregister URLs that look like ours
                    should_remove = False
                    try:
                        from urllib.parse import urlparse
                        parsed = urlparse(url)
                        host = parsed.hostname
                        port = parsed.port
                    except Exception:
                        host, port = None, None

                    if self.config.connectivity.mode == "tunneled":
                        adv_host = self.config.connectivity.advertise_host
                        pr_min, pr_max = self.config.connectivity.local_port_range
                        if host in {adv_host, "127.0.0.1", "localhost"} and port and pr_min <= port <= pr_max:
                            should_remove = True
                    else:
                        # In direct mode, only remove if it's one of our known URLs
                        known_urls = {w.advertised_url for w in self.workers.values() if w.advertised_url}
                        if url in known_urls:
                            should_remove = True

                    if not should_remove:
                        continue

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

        # Proactively deregister from router if we had advertised a URL
        if worker.advertised_url:
            try:
                self.router_client.remove(worker.advertised_url)
                self.logger.info(f"Deregistered worker from router: {worker.advertised_url}")
            except Exception as e:
                # Don't block cleanup if router removal fails
                self.logger.warning(f"Failed to deregister worker from router: {e}")

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
        
        # Unregister from walltime manager
        self.walltime_manager.unregister_worker(job_id)
        
        # Remove from our state
        del self.workers[job_id]
        
        self.logger.info(f"Removed worker: {job_id}")
        self._persist_state()
    
    def _handle_config_replacements(self) -> None:
        """Handle workers that need replacement due to configuration changes."""
        # Clean up timed out replacement tasks
        self.replacement_engine.cleanup_timed_out_tasks()
        
        # Find workers that need replacement
        workers_needing_replacement = []
        
        for worker in self.workers.values():
            if worker.replacement_scheduled:
                continue
                
            replacement_reason = None
            
            # Check for configuration changes
            if worker.needs_replacement(self.current_config_snapshot):
                replacement_reason = worker.get_replacement_reason(self.current_config_snapshot)
            
            # Check for walltime approaching
            elif self.walltime_manager.should_start_proactive_replacement(worker.job_id):
                urgency = self.walltime_manager.get_replacement_urgency(worker.job_id)
                replacement_reason = f"walltime_approaching_{urgency}"
                
                # Log walltime replacement with details
                window = self.walltime_manager.estimate_replacement_window(worker.job_id)
                if window:
                    self.logger.info(
                        f"Worker {worker.job_id} approaching walltime: "
                        f"{window['time_remaining_seconds']/60:.1f}min remaining, "
                        f"urgency={urgency}, can_complete={window['can_complete_replacement']}"
                    )
            
            if replacement_reason:
                worker.replacement_reason = replacement_reason
                workers_needing_replacement.append(worker.job_id)
                self.logger.info(f"Worker {worker.job_id} needs replacement: {replacement_reason}")
        
        if not workers_needing_replacement:
            return
        
        # Get current healthy worker count
        healthy_workers = [w for w in self.workers.values() if w.is_healthy()]
        current_healthy_count = len(healthy_workers)
        
        # Create replacement plan
        plan = self.replacement_engine.plan_replacement(
            workers_needing_replacement=workers_needing_replacement,
            current_healthy_count=current_healthy_count,
            desired_replicas=self.config.replicas,
            replacement_reason="config_change"
        )
        
        if not plan:
            self.logger.warning(f"Cannot safely replace workers: need more healthy workers")
            return
        
        self.logger.info(
            f"Starting replacement plan: {len(plan.workers_to_replace)} workers, "
            f"strategy={plan.strategy.value}, max_concurrent={plan.max_concurrent}, "
            f"estimated_duration={plan.estimated_duration_minutes}min"
        )
        
        # Start replacement tasks for workers we can handle
        workers_to_start = []
        for job_id in plan.workers_to_replace:
            if self.replacement_engine.can_start_more_replacements(plan.max_concurrent):
                workers_to_start.append(job_id)
        
        # Execute replacements
        for job_id in workers_to_start:
            try:
                self._start_worker_replacement(job_id)
            except Exception as e:
                self.logger.error(f"Failed to start replacement for worker {job_id}: {e}")
                self.replacement_engine.fail_replacement_task(job_id, str(e))
        
        # Update replacement tasks for existing workers
        self._update_replacement_tasks()
    
    def _start_worker_replacement(self, old_job_id: str) -> None:
        """Start replacement for a specific worker."""
        old_worker = self.workers.get(old_job_id)
        if not old_worker:
            return
        
        # Mark worker as having replacement scheduled
        old_worker.replacement_scheduled = True
        
        # Start replacement task
        task = self.replacement_engine.start_replacement_task(old_job_id)
        
        # Start new worker
        try:
            self._start_worker()  # This will create a new worker with current config
            
            # Find the newly created worker (it will have the highest submitted_at time)
            newest_worker = max(self.workers.values(), key=lambda w: w.submitted_at)
            
            # Update replacement task
            self.replacement_engine.update_task_new_worker_started(old_job_id, newest_worker.job_id)
            
            self.logger.info(f"Started replacement worker {newest_worker.job_id} for {old_job_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to start replacement worker for {old_job_id}: {e}")
            self.replacement_engine.fail_replacement_task(old_job_id, str(e))
            old_worker.replacement_scheduled = False
    
    def _update_replacement_tasks(self) -> None:
        """Update status of active replacement tasks."""
        for task in list(self.replacement_engine.active_tasks.values()):
            old_worker = self.workers.get(task.old_worker_job_id)
            new_worker = None
            
            if task.replacement_job_id:
                new_worker = self.workers.get(task.replacement_job_id)
            
            # Check if new worker is ready
            if new_worker and new_worker.is_healthy() and not task.new_worker_ready_at:
                self.replacement_engine.update_task_new_worker_ready(task.old_worker_job_id)
            
            # If new worker is ready, start draining old worker
            if (task.new_worker_ready_at and not task.old_worker_drained_at 
                and old_worker and old_worker.advertised_url):
                
                try:
                    # Deregister old worker from router
                    self.router_client.remove(old_worker.advertised_url)
                    self.replacement_engine.update_task_old_worker_drained(task.old_worker_job_id)
                    self.logger.info(f"Drained old worker {task.old_worker_job_id} from router")
                    
                    # Schedule old worker removal after grace period
                    import threading
                    def delayed_removal():
                        time.sleep(self.replacement_engine.drain_grace_period_seconds)
                        if task.old_worker_job_id in self.workers:
                            try:
                                self.slurm.cancel(task.old_worker_job_id)
                                self._remove_worker(task.old_worker_job_id)
                                self.replacement_engine.complete_replacement_task(task.old_worker_job_id)
                                self.logger.info(f"Completed replacement of worker {task.old_worker_job_id}")
                            except Exception as e:
                                self.logger.error(f"Failed to complete replacement: {e}")
                                self.replacement_engine.fail_replacement_task(task.old_worker_job_id, str(e))
                    
                    threading.Thread(target=delayed_removal, daemon=True).start()
                    
                except Exception as e:
                    self.logger.error(f"Failed to drain old worker {task.old_worker_job_id}: {e}")
                    self.replacement_engine.fail_replacement_task(task.old_worker_job_id, str(e))
    
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
        
        # Update replacement metrics
        active_replacements = self.replacement_engine.get_active_replacement_count()
        workers_needing_replacement = len([w for w in self.workers.values() 
                                         if w.needs_replacement(self.current_config_snapshot)])
        
        # Count unique generations
        generations = set()
        for worker in self.workers.values():
            if worker.generation:
                generations.add(worker.generation.generation_id)
        active_generations = len(generations)
        
        self.metrics.update_replacement_metrics(
            self.config.name,
            active_replacements=active_replacements,
            workers_needing_replacement=workers_needing_replacement,
            active_generations=active_generations
        )
        
        # Update walltime metrics
        walltime_stats = self.walltime_manager.get_walltime_statistics()
        if walltime_stats:
            self.metrics.update_walltime_metrics(
                self.config.name,
                workers_approaching_walltime=walltime_stats.get('workers_approaching_walltime', 0),
                min_time_remaining_minutes=walltime_stats.get('min_time_remaining_minutes', 0),
                avg_walltime_percent_complete=walltime_stats.get('avg_percent_complete', 0)
            )
    
    def shutdown(self) -> None:
        """Shutdown the reconciler gracefully."""
        self.logger.info("Shutting down reconciler")
        self.running = False
        
        # Clean up tunnels
        self.tunnel_manager.shutdown()
        # Persist final state
        try:
            self._persist_state()
        except Exception:
            pass
        
        self.logger.info("Reconciler shutdown complete")

    def _persist_state(self) -> None:
        """Persist current deployment state to the state store."""
        try:
            workers = []
            for w in self.workers.values():
                # Include generation info for persistence
                config_hash = None
                generation_id = None  
                created_at = None
                if w.generation:
                    config_hash = w.generation.config_hash
                    generation_id = w.generation.generation_id
                    created_at = w.generation.created_at
                
                workers.append(
                    SerializableWorker(
                        job_id=w.job_id,
                        instance_uuid=w.instance_uuid,
                        node=w.node,
                        remote_port=w.remote_port,
                        advertise_port=w.advertise_port,
                        worker_url=w.worker_url,
                        advertised_url=w.advertised_url,
                        health_status=w.health_status.value,
                        last_seen=w.last_seen,
                        submitted_at=w.submitted_at,
                        config_hash=config_hash,
                        generation_id=generation_id,
                        created_at=created_at,
                    )
                )

            snapshot = DeploymentSnapshot(
                name=self.config.name,
                workers=workers,
                allocated_ports=sorted(list(self.port_allocator.get_allocated_ports())),
            )
            self.state_store.save_deployment(snapshot)
        except Exception as e:
            self.logger.warning(f"Failed to persist state: {e}")
