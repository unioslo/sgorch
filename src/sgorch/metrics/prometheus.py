import time
from typing import Dict, Optional, Set
from threading import Lock

from prometheus_client import Counter, Gauge, Histogram, start_http_server, CollectorRegistry

from ..logging_setup import get_logger
from ..config import MetricsConfig


logger = get_logger(__name__)


class SGOrchMetrics:
    """Prometheus metrics for SGOrch."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        # Use a per-instance registry to avoid global duplication across tests
        self.registry: CollectorRegistry = registry or CollectorRegistry()
        # Deployment-level metrics
        self.workers_desired = Gauge(
            'sgorch_workers_desired',
            'Desired number of workers per deployment',
            ['deployment'],
            registry=self.registry
        )
        
        self.workers_ready = Gauge(
            'sgorch_workers_ready',
            'Number of ready workers per deployment',
            ['deployment'],
            registry=self.registry
        )
        
        self.workers_starting = Gauge(
            'sgorch_workers_starting',
            'Number of starting workers per deployment',
            ['deployment'],
            registry=self.registry
        )
        
        self.workers_unhealthy = Gauge(
            'sgorch_workers_unhealthy',
            'Number of unhealthy workers per deployment',
            ['deployment'],
            registry=self.registry
        )
        
        # Tunnel metrics
        self.tunnels_up = Gauge(
            'sgorch_tunnels_up',
            'Number of active tunnels per deployment',
            ['deployment'],
            registry=self.registry
        )
        
        # Operation counters
        self.restarts_total = Counter(
            'sgorch_restarts_total',
            'Total number of worker restarts',
            ['deployment', 'reason'],
            registry=self.registry
        )
        
        self.jobs_submitted_total = Counter(
            'sgorch_jobs_submitted_total',
            'Total number of SLURM jobs submitted',
            ['deployment'],
            registry=self.registry
        )
        
        self.jobs_failed_total = Counter(
            'sgorch_jobs_failed_total',
            'Total number of failed SLURM jobs',
            ['deployment', 'reason'],
            registry=self.registry
        )
        
        # Router operation metrics
        self.router_errors_total = Counter(
            'sgorch_router_errors_total',
            'Total number of router API errors',
            ['deployment', 'operation'],
            registry=self.registry
        )
        
        self.router_operations_total = Counter(
            'sgorch_router_operations_total',
            'Total number of router operations',
            ['deployment', 'operation', 'status'],
            registry=self.registry
        )
        
        # Note: router-level probe metrics removed per user request; only per-node probes remain

        # Node probe (per-worker) metrics
        self.node_probe_total = Counter(
            'sgorch_node_probe_total',
            'Total number of worker node probe attempts',
            ['deployment', 'worker', 'status'],
            registry=self.registry
        )

        self.node_probe_latency = Histogram(
            'sgorch_node_probe_latency_seconds',
            'Latency of OpenAI worker node test probes',
            ['deployment', 'worker'],
            registry=self.registry
        )

        self.node_probe_success = Gauge(
            'sgorch_node_probe_success',
            '1 if last worker node probe succeeded, else 0',
            ['deployment', 'worker'],
            registry=self.registry
        )

        self.node_probe_timestamp = Gauge(
            'sgorch_node_probe_timestamp',
            'Timestamp of last worker node probe',
            ['deployment', 'worker'],
            registry=self.registry
        )
        
        # SLURM operation metrics
        self.slurm_errors_total = Counter(
            'sgorch_slurm_errors_total',
            'Total number of SLURM errors',
            ['deployment', 'operation'],
            registry=self.registry
        )
        
        self.slurm_operations_total = Counter(
            'sgorch_slurm_operations_total',
            'Total number of SLURM operations',
            ['deployment', 'operation', 'status'],
            registry=self.registry
        )
        
        # Health check metrics
        self.health_checks_total = Counter(
            'sgorch_health_checks_total',
            'Total number of health checks performed',
            ['deployment', 'status'],
            registry=self.registry
        )
        
        self.health_check_duration = Histogram(
            'sgorch_health_check_duration_seconds',
            'Duration of health checks',
            ['deployment'],
            registry=self.registry
        )
        
        # System metrics
        self.reconcile_duration = Histogram(
            'sgorch_reconcile_duration_seconds',
            'Duration of reconciliation cycles',
            ['deployment'],
            registry=self.registry
        )
        
        self.last_successful_reconcile = Gauge(
            'sgorch_last_successful_reconcile_timestamp',
            'Timestamp of last successful reconciliation',
            ['deployment'],
            registry=self.registry
        )
        
        # Port allocation metrics
        self.ports_allocated = Gauge(
            'sgorch_ports_allocated',
            'Number of allocated ports',
            ['deployment'],
            registry=self.registry
        )
        
        self.ports_available = Gauge(
            'sgorch_ports_available',
            'Number of available ports',
            ['deployment'],
            registry=self.registry
        )
        
        # Node blacklist metrics
        self.blacklisted_nodes = Gauge(
            'sgorch_blacklisted_nodes',
            'Number of blacklisted nodes',
            ['deployment'],
            registry=self.registry
        )
        
        # Configuration management metrics
        self.active_replacements = Gauge(
            'sgorch_active_replacements',
            'Number of active replacement tasks',
            ['deployment'],
            registry=self.registry
        )
        
        self.completed_replacements_total = Counter(
            'sgorch_completed_replacements_total',
            'Total number of completed replacements',
            ['deployment', 'reason', 'status'],
            registry=self.registry
        )
        
        self.workers_needing_replacement = Gauge(
            'sgorch_workers_needing_replacement',
            'Number of workers needing replacement',
            ['deployment'],
            registry=self.registry
        )
        
        self.worker_generations_active = Gauge(
            'sgorch_worker_generations_active',
            'Number of different config generations active',
            ['deployment'],
            registry=self.registry
        )
        
        # Walltime metrics
        self.workers_approaching_walltime = Gauge(
            'sgorch_workers_approaching_walltime',
            'Number of workers approaching walltime limit',
            ['deployment'],
            registry=self.registry
        )
        
        self.min_time_remaining_minutes = Gauge(
            'sgorch_min_time_remaining_minutes',
            'Minimum time remaining across all workers in minutes',
            ['deployment'],
            registry=self.registry
        )
        
        self.avg_walltime_percent_complete = Gauge(
            'sgorch_avg_walltime_percent_complete',
            'Average walltime consumption percentage',
            ['deployment'],
            registry=self.registry
        )
        
        self.replacement_duration_seconds = Histogram(
            'sgorch_replacement_duration_seconds',
            'Duration of worker replacements',
            ['deployment', 'reason'],
            registry=self.registry
        )
        
        self._http_server_port: Optional[int] = None
        self._lock = Lock()
    
    def update_worker_counts(
        self,
        deployment: str,
        desired: int = 0,
        ready: int = 0,
        starting: int = 0,
        unhealthy: int = 0
    ) -> None:
        """Update worker count metrics."""
        self.workers_desired.labels(deployment=deployment).set(desired)
        self.workers_ready.labels(deployment=deployment).set(ready)
        self.workers_starting.labels(deployment=deployment).set(starting)
        self.workers_unhealthy.labels(deployment=deployment).set(unhealthy)
    
    def update_tunnel_count(self, deployment: str, count: int) -> None:
        """Update tunnel count metric."""
        self.tunnels_up.labels(deployment=deployment).set(count)
    
    def record_restart(self, deployment: str, reason: str) -> None:
        """Record a worker restart."""
        self.restarts_total.labels(deployment=deployment, reason=reason).inc()
    
    def record_job_submitted(self, deployment: str) -> None:
        """Record a job submission."""
        self.jobs_submitted_total.labels(deployment=deployment).inc()
    
    def record_job_failed(self, deployment: str, reason: str) -> None:
        """Record a job failure."""
        self.jobs_failed_total.labels(deployment=deployment, reason=reason).inc()
    
    def record_router_operation(
        self,
        deployment: str,
        operation: str,
        success: bool
    ) -> None:
        """Record a router operation."""
        status = "success" if success else "failure"
        self.router_operations_total.labels(
            deployment=deployment,
            operation=operation,
            status=status
        ).inc()
        
        if not success:
            self.router_errors_total.labels(
                deployment=deployment,
                operation=operation
            ).inc()

    

    def record_node_probe(self, deployment: str, worker: str, success: bool, latency: float) -> None:
        """Record a per-worker OpenAI-compatible probe result."""
        status = "success" if success else "failure"
        self.node_probe_total.labels(deployment=deployment, worker=worker, status=status).inc()
        if success:
            self.node_probe_latency.labels(deployment=deployment, worker=worker).observe(latency)
        self.node_probe_success.labels(deployment=deployment, worker=worker).set(1.0 if success else 0.0)
        self.node_probe_timestamp.labels(deployment=deployment, worker=worker).set(time.time())
    
    def record_slurm_operation(
        self,
        deployment: str,
        operation: str,
        success: bool
    ) -> None:
        """Record a SLURM operation."""
        status = "success" if success else "failure"
        self.slurm_operations_total.labels(
            deployment=deployment,
            operation=operation,
            status=status
        ).inc()
        
        if not success:
            self.slurm_errors_total.labels(
                deployment=deployment,
                operation=operation
            ).inc()
    
    def record_health_check(self, deployment: str, healthy: bool, duration: float) -> None:
        """Record a health check."""
        status = "healthy" if healthy else "unhealthy"
        self.health_checks_total.labels(
            deployment=deployment,
            status=status
        ).inc()
        
        self.health_check_duration.labels(deployment=deployment).observe(duration)
    
    def record_reconcile_duration(self, deployment: str, duration: float) -> None:
        """Record reconciliation duration."""
        self.reconcile_duration.labels(deployment=deployment).observe(duration)
        self.last_successful_reconcile.labels(deployment=deployment).set(time.time())
    
    def update_port_counts(self, deployment: str, allocated: int, available: int) -> None:
        """Update port allocation metrics."""
        self.ports_allocated.labels(deployment=deployment).set(allocated)
        self.ports_available.labels(deployment=deployment).set(available)
    
    def update_blacklisted_nodes(self, deployment: str, count: int) -> None:
        """Update blacklisted nodes count."""
        self.blacklisted_nodes.labels(deployment=deployment).set(count)
    
    def update_replacement_metrics(
        self,
        deployment: str,
        active_replacements: int,
        workers_needing_replacement: int,
        active_generations: int
    ) -> None:
        """Update configuration management metrics."""
        self.active_replacements.labels(deployment=deployment).set(active_replacements)
        self.workers_needing_replacement.labels(deployment=deployment).set(workers_needing_replacement)
        self.worker_generations_active.labels(deployment=deployment).set(active_generations)
    
    def record_replacement_completed(self, deployment: str, reason: str, success: bool, duration_seconds: float) -> None:
        """Record a completed replacement."""
        status = "success" if success else "failure"
        self.completed_replacements_total.labels(
            deployment=deployment, 
            reason=reason, 
            status=status
        ).inc()
        self.replacement_duration_seconds.labels(deployment=deployment, reason=reason).observe(duration_seconds)
    
    def update_walltime_metrics(
        self,
        deployment: str,
        workers_approaching_walltime: int,
        min_time_remaining_minutes: float,
        avg_walltime_percent_complete: float
    ) -> None:
        """Update walltime-related metrics."""
        self.workers_approaching_walltime.labels(deployment=deployment).set(workers_approaching_walltime)
        self.min_time_remaining_minutes.labels(deployment=deployment).set(min_time_remaining_minutes)
        self.avg_walltime_percent_complete.labels(deployment=deployment).set(avg_walltime_percent_complete)
    
    def cleanup_deployment_metrics(self, deployment: str) -> None:
        """Clean up metrics for a removed deployment."""
        # Note: Prometheus client doesn't support removing metrics by label
        # This is a limitation of the current prometheus_client library
        # In practice, stale metrics will just stop being updated
        logger.debug(f"Would clean up metrics for deployment {deployment}")
    
    def start_http_server(self, config: MetricsConfig) -> bool:
        """Start the Prometheus HTTP server."""
        if not config.enabled:
            logger.info("Metrics disabled in configuration")
            return False
        
        with self._lock:
            if self._http_server_port is not None:
                logger.warning(f"Metrics server already running on port {self._http_server_port}")
                return True
            
            try:
                # Pass the custom registry so SGOrch metrics are included
                start_http_server(config.port, addr=config.bind, registry=self.registry)
                self._http_server_port = config.port
                
                logger.info(f"Started Prometheus metrics server on {config.bind}:{config.port}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to start metrics server: {e}")
                return False
    
    def is_running(self) -> bool:
        """Check if metrics server is running."""
        with self._lock:
            return self._http_server_port is not None


# Global metrics instance
metrics = SGOrchMetrics()


def get_metrics() -> SGOrchMetrics:
    """Get the global metrics instance."""
    return metrics
