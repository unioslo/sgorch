import time
from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

import httpx

from ..logging_setup import get_logger
from ..config import HealthConfig
from ..util.backoff import BackoffManager


logger = get_logger(__name__)


class HealthStatus(Enum):
    """Health check status."""
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"


@dataclass
class HealthResult:
    """Result of a health check."""
    status: HealthStatus
    response_time_ms: Optional[float] = None
    status_code: Optional[int] = None
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class HealthProbe:
    """HTTP health probe for monitoring worker health."""
    
    def __init__(self, config: HealthConfig, worker_url: str):
        self.config = config
        self.worker_url = worker_url.rstrip('/')
        self.health_url = f"{self.worker_url}{config.path}"
        
        # State tracking
        self.consecutive_successes = 0
        self.consecutive_failures = 0
        self.last_probe_time: Optional[float] = None
        self.current_status = HealthStatus.UNKNOWN
        self.last_successful_probe: Optional[float] = None
        
        # HTTP client with timeout
        self.client = httpx.Client(timeout=config.timeout_s)
        
        # Backoff for failed requests
        self.backoff = BackoffManager(
            strategy="exponential",
            base_delay=1.0,
            max_delay=min(30.0, config.interval_s * 0.8),
            max_attempts=None  # Unlimited attempts
        )
    
    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, 'client'):
            self.client.close()
    
    def probe(self) -> HealthResult:
        """Perform a single health check probe."""
        self.last_probe_time = time.time()
        
        logger.debug(f"Health check for {self.worker_url}")
        
        try:
            start_time = time.time()
            
            # Build headers with auth if configured
            headers = dict(self.config.headers) if self.config.headers else {}
            
            # Perform HTTP request
            response = self.client.get(self.health_url, headers=headers)
            
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            # Check response
            result = self._evaluate_response(response, response_time_ms)
            
            # Update state
            self._update_state(result)
            
            return result
            
        except httpx.TimeoutException:
            error_msg = f"Health check timeout for {self.worker_url}"
            logger.debug(error_msg)
            result = HealthResult(
                status=HealthStatus.UNHEALTHY,
                error=error_msg
            )
            self._update_state(result)
            return result
            
        except httpx.ConnectError:
            error_msg = f"Connection failed to {self.worker_url}"
            logger.debug(error_msg)
            result = HealthResult(
                status=HealthStatus.UNHEALTHY,
                error=error_msg
            )
            self._update_state(result)
            return result
            
        except Exception as e:
            error_msg = f"Health check error for {self.worker_url}: {e}"
            logger.debug(error_msg)
            result = HealthResult(
                status=HealthStatus.UNHEALTHY,
                error=error_msg
            )
            self._update_state(result)
            return result
    
    def _evaluate_response(self, response: httpx.Response, response_time_ms: float) -> HealthResult:
        """Evaluate HTTP response and return health result."""
        status_code = response.status_code
        
        # Generally, 2xx status codes indicate health
        if 200 <= status_code < 300:
            return HealthResult(
                status=HealthStatus.HEALTHY,
                response_time_ms=response_time_ms,
                status_code=status_code
            )
        else:
            return HealthResult(
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time_ms,
                status_code=status_code,
                error=f"HTTP {status_code}: {response.text[:200]}"
            )
    
    def _update_state(self, result: HealthResult) -> None:
        """Update internal state based on health result."""
        if result.status == HealthStatus.HEALTHY:
            self.consecutive_successes += 1
            self.consecutive_failures = 0
            self.last_successful_probe = time.time()
            self.backoff.reset()  # Reset backoff on success
            
            # Determine if we're ready (enough consecutive successes)
            if self.consecutive_successes >= self.config.consecutive_ok_for_ready:
                self.current_status = HealthStatus.HEALTHY
            else:
                self.current_status = HealthStatus.STARTING
                
        else:
            self.consecutive_failures += 1
            self.consecutive_successes = 0
            
            # Determine if we should mark as unhealthy
            if self.consecutive_failures >= self.config.failures_to_unhealthy:
                self.current_status = HealthStatus.UNHEALTHY
            elif self.current_status == HealthStatus.HEALTHY:
                # Give some grace period before marking unhealthy
                self.current_status = HealthStatus.STARTING
    
    def is_ready(self) -> bool:
        """Check if worker is ready (healthy with enough consecutive successes)."""
        return self.current_status == HealthStatus.HEALTHY
    
    def is_unhealthy(self) -> bool:
        """Check if worker is definitely unhealthy."""
        return self.current_status == HealthStatus.UNHEALTHY
    
    def is_starting(self) -> bool:
        """Check if worker is in starting state."""
        return self.current_status == HealthStatus.STARTING
    
    def get_status(self) -> HealthStatus:
        """Get current health status."""
        return self.current_status
    
    def time_since_last_success(self) -> Optional[float]:
        """Get seconds since last successful probe."""
        if self.last_successful_probe is None:
            return None
        return time.time() - self.last_successful_probe
    
    def time_since_last_probe(self) -> Optional[float]:
        """Get seconds since last probe attempt."""
        if self.last_probe_time is None:
            return None
        return time.time() - self.last_probe_time
    
    def should_probe(self) -> bool:
        """Check if it's time for the next probe based on interval."""
        if self.last_probe_time is None:
            return True
        
        time_since_last = self.time_since_last_probe()
        return time_since_last >= self.config.interval_s
    
    def reset(self) -> None:
        """Reset probe state."""
        self.consecutive_successes = 0
        self.consecutive_failures = 0
        self.last_probe_time = None
        self.current_status = HealthStatus.UNKNOWN
        self.last_successful_probe = None
        self.backoff.reset()


class HealthMonitor:
    """Manages health monitoring for multiple workers."""
    
    def __init__(self):
        self.probes: Dict[str, HealthProbe] = {}
    
    def add_worker(self, worker_url: str, config: HealthConfig) -> None:
        """Add a worker for health monitoring."""
        if worker_url in self.probes:
            logger.warning(f"Worker already being monitored: {worker_url}")
            return
        
        self.probes[worker_url] = HealthProbe(config, worker_url)
        logger.info(f"Added health monitoring for worker: {worker_url}")
    
    def remove_worker(self, worker_url: str) -> None:
        """Remove a worker from health monitoring."""
        if worker_url in self.probes:
            del self.probes[worker_url]
            logger.info(f"Removed health monitoring for worker: {worker_url}")
    
    def probe_worker(self, worker_url: str) -> Optional[HealthResult]:
        """Perform health check for a specific worker."""
        probe = self.probes.get(worker_url)
        if not probe:
            return None
        
        return probe.probe()
    
    def probe_all_due(self) -> Dict[str, HealthResult]:
        """Probe all workers that are due for health checks."""
        results = {}
        
        for worker_url, probe in self.probes.items():
            if probe.should_probe():
                results[worker_url] = probe.probe()
        
        return results
    
    def get_worker_status(self, worker_url: str) -> Optional[HealthStatus]:
        """Get current status of a worker."""
        probe = self.probes.get(worker_url)
        return probe.get_status() if probe else None
    
    def get_ready_workers(self) -> set[str]:
        """Get set of worker URLs that are ready."""
        return {
            url for url, probe in self.probes.items()
            if probe.is_ready()
        }
    
    def get_unhealthy_workers(self) -> set[str]:
        """Get set of worker URLs that are unhealthy."""
        return {
            url for url, probe in self.probes.items()
            if probe.is_unhealthy()
        }
    
    def get_starting_workers(self) -> set[str]:
        """Get set of worker URLs that are starting."""
        return {
            url for url, probe in self.probes.items()
            if probe.is_starting()
        }