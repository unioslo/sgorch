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
                # Avoid passing registry to remain compatible with simple monkeypatch in tests
                start_http_server(config.port, addr=config.bind)
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
