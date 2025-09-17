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
        
        # Backoff for failed requests influences probing cadence during startup/failure
        # Use a base delay slower than the steady-state interval to reduce noise.
        self.backoff = BackoffManager(
            strategy="exponential",
            base_delay=max(2.0, float(config.interval_s) * 2.0),
            max_delay=max(30.0, float(config.interval_s) * 10.0),
            max_attempts=None  # Unlimited attempts
        )
        # Next earliest time we should probe (set after each result)
        self._next_probe_earliest: Optional[float] = None
    
    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, 'client'):
            self.client.close()
    
    def probe(self) -> HealthResult:
        """Perform a single health check probe."""
        now = time.time()
        self.last_probe_time = now
        
        logger.info(f"Health check attempt {self.consecutive_failures + 1} for {self.worker_url}")
        
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
            old_status = self.current_status
            self._update_state(result)
            
            # Schedule next probe time based on success/failure
            self._schedule_next_probe(result)

            # Log result
            if result.status == HealthStatus.HEALTHY:
                logger.info(f"Health check SUCCESS for {self.worker_url}: {result.status_code} in {response_time_ms:.1f}ms")
            else:
                logger.warning(f"Health check FAILED for {self.worker_url}: {result.status_code} - {result.error}")
            
            # Log status changes
            if old_status != self.current_status:
                logger.info(f"Worker {self.worker_url} status: {old_status} -> {self.current_status} (successes: {self.consecutive_successes}, failures: {self.consecutive_failures})")
            
            return result
            
        except httpx.TimeoutException:
            error_msg = f"Health check timeout after {self.config.timeout_s}s"
            logger.warning(f"Health check TIMEOUT for {self.worker_url}: {error_msg}")
            result = HealthResult(
                status=HealthStatus.UNHEALTHY,
                error=error_msg
            )
            old_status = self.current_status
            self._update_state(result)
            self._schedule_next_probe(result)
            
            if old_status != self.current_status:
                logger.info(f"Worker {self.worker_url} status: {old_status} -> {self.current_status} (failures: {self.consecutive_failures})")
            
            return result
            
        except httpx.ConnectError:
            error_msg = f"Connection refused - backend likely still starting"
            logger.info(f"Health check CONNECTION FAILED for {self.worker_url}: {error_msg}")
            result = HealthResult(
                status=HealthStatus.UNHEALTHY,
                error=error_msg
            )
            old_status = self.current_status
            self._update_state(result)
            self._schedule_next_probe(result)
            
            if old_status != self.current_status:
                logger.info(f"Worker {self.worker_url} status: {old_status} -> {self.current_status} (failures: {self.consecutive_failures})")
            
            return result
            
        except Exception as e:
            error_msg = f"Health check error: {e}"
            logger.warning(f"Health check ERROR for {self.worker_url}: {error_msg}")
            result = HealthResult(
                status=HealthStatus.UNHEALTHY,
                error=error_msg
            )
            old_status = self.current_status
            self._update_state(result)
            self._schedule_next_probe(result)
            
            if old_status != self.current_status:
                logger.info(f"Worker {self.worker_url} status: {old_status} -> {self.current_status} (failures: {self.consecutive_failures})")
            
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
        """Check if it's time for the next probe based on interval/backoff."""
        # First probe happens immediately
        if self.last_probe_time is None or self._next_probe_earliest is None:
            return True

        return time.time() >= self._next_probe_earliest
    
    def reset(self) -> None:
        """Reset probe state."""
        self.consecutive_successes = 0
        self.consecutive_failures = 0
        self.last_probe_time = None
        self.current_status = HealthStatus.UNKNOWN
        self.last_successful_probe = None
        self.backoff.reset()
        self._next_probe_earliest = None

    def _schedule_next_probe(self, result: HealthResult) -> None:
        """Schedule the next probe time using interval or backoff."""
        now = time.time()
        if result.status == HealthStatus.HEALTHY:
            # Normal cadence when healthy
            self._next_probe_earliest = now + float(self.config.interval_s)
            self.backoff.reset()
        else:
            # Increase delay between attempts during startup/failure.
            # BackoffManager yields 0.0 on first attempt; ensure a sensible minimum delay.
            delay = self.backoff.next_delay()
            min_delay = max(5.0, float(self.config.interval_s) * 2.0)
            if delay is None or delay <= 0.01:
                delay = min_delay
            self._next_probe_earliest = now + float(delay)


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
