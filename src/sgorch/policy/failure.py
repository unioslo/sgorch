import time
from typing import Dict, Set, Optional
from dataclasses import dataclass
from threading import Lock

from ..logging_setup import get_logger
from ..util.backoff import BackoffManager


logger = get_logger(__name__)


@dataclass
class FailureEvent:
    """Record of a failure event."""
    timestamp: float
    reason: str
    details: str
    node: Optional[str] = None


class NodeBlacklist:
    """Manages blacklisting of problematic nodes."""
    
    def __init__(self, cooldown_seconds: int = 600):
        self.cooldown_seconds = cooldown_seconds
        self.blacklisted_nodes: Dict[str, float] = {}  # node -> blacklist_until_timestamp
        self.lock = Lock()
    
    def blacklist_node(self, node: str, reason: str = "") -> None:
        """Add a node to the blacklist."""
        if not node:
            return
        
        blacklist_until = time.time() + self.cooldown_seconds
        
        with self.lock:
            self.blacklisted_nodes[node] = blacklist_until
        
        logger.warning(
            f"Blacklisted node {node} until "
            f"{time.strftime('%H:%M:%S', time.localtime(blacklist_until))} "
            f"(reason: {reason})"
        )
    
    def is_blacklisted(self, node: str) -> bool:
        """Check if a node is currently blacklisted."""
        if not node:
            return False
        
        with self.lock:
            blacklist_until = self.blacklisted_nodes.get(node)
            if blacklist_until is None:
                return False
            
            if time.time() >= blacklist_until:
                # Blacklist expired, remove it
                del self.blacklisted_nodes[node]
                logger.info(f"Node {node} blacklist expired")
                return False
            
            return True
    
    def get_blacklisted_nodes(self) -> Set[str]:
        """Get set of currently blacklisted nodes."""
        current_time = time.time()
        
        with self.lock:
            # Clean up expired blacklists
            expired = [
                node for node, until in self.blacklisted_nodes.items()
                if current_time >= until
            ]
            
            for node in expired:
                del self.blacklisted_nodes[node]
                logger.info(f"Node {node} blacklist expired")
            
            return set(self.blacklisted_nodes.keys())
    
    def clear_blacklist(self, node: Optional[str] = None) -> None:
        """Clear blacklist for a specific node or all nodes."""
        with self.lock:
            if node:
                if node in self.blacklisted_nodes:
                    del self.blacklisted_nodes[node]
                    logger.info(f"Cleared blacklist for node {node}")
            else:
                cleared_count = len(self.blacklisted_nodes)
                self.blacklisted_nodes.clear()
                logger.info(f"Cleared blacklist for {cleared_count} nodes")
    
    def get_cooldown_remaining(self, node: str) -> float:
        """Get remaining cooldown time for a node in seconds."""
        with self.lock:
            blacklist_until = self.blacklisted_nodes.get(node)
            if blacklist_until is None:
                return 0.0
            
            remaining = blacklist_until - time.time()
            return max(0.0, remaining)


class FailureTracker:
    """Tracks failure patterns and triggers blacklisting."""
    
    def __init__(
        self,
        node_blacklist: NodeBlacklist,
        max_failures_per_node: int = 3,
        failure_window_seconds: int = 300
    ):
        self.node_blacklist = node_blacklist
        self.max_failures_per_node = max_failures_per_node
        self.failure_window_seconds = failure_window_seconds
        
        # Track failures per node
        self.node_failures: Dict[str, list[FailureEvent]] = {}
        self.lock = Lock()
    
    def record_failure(
        self,
        reason: str,
        details: str = "",
        node: Optional[str] = None
    ) -> None:
        """Record a failure event."""
        event = FailureEvent(
            timestamp=time.time(),
            reason=reason,
            details=details,
            node=node
        )
        
        logger.warning(
            f"Failure recorded: {reason} "
            f"{'on node ' + node if node else ''} - {details}"
        )
        
        if node:
            self._record_node_failure(node, event)
    
    def _record_node_failure(self, node: str, event: FailureEvent) -> None:
        """Record a failure for a specific node and check thresholds."""
        with self.lock:
            if node not in self.node_failures:
                self.node_failures[node] = []
            
            # Add the failure
            self.node_failures[node].append(event)
            
            # Clean up old failures outside the window
            cutoff_time = time.time() - self.failure_window_seconds
            self.node_failures[node] = [
                f for f in self.node_failures[node]
                if f.timestamp > cutoff_time
            ]
            
            # Check if we should blacklist this node
            failure_count = len(self.node_failures[node])
            
            if failure_count >= self.max_failures_per_node:
                reasons = [f.reason for f in self.node_failures[node]]
                reason_summary = ", ".join(set(reasons))
                
                self.node_blacklist.blacklist_node(
                    node,
                    f"{failure_count} failures: {reason_summary}"
                )
    
    def get_node_failure_count(self, node: str, window_seconds: Optional[int] = None) -> int:
        """Get failure count for a node within a time window."""
        if window_seconds is None:
            window_seconds = self.failure_window_seconds
        
        cutoff_time = time.time() - window_seconds
        
        with self.lock:
            failures = self.node_failures.get(node, [])
            return sum(1 for f in failures if f.timestamp > cutoff_time)
    
    def get_recent_failures(self, window_seconds: int = 300) -> list[FailureEvent]:
        """Get all recent failures across all nodes."""
        cutoff_time = time.time() - window_seconds
        recent_failures = []
        
        with self.lock:
            for node_failures in self.node_failures.values():
                for failure in node_failures:
                    if failure.timestamp > cutoff_time:
                        recent_failures.append(failure)
        
        # Sort by timestamp
        recent_failures.sort(key=lambda f: f.timestamp)
        return recent_failures


class RestartPolicy:
    """Manages restart policies and backoff for workers."""
    
    def __init__(
        self,
        restart_backoff_seconds: int = 60,
        max_restart_attempts: Optional[int] = None,
        deregister_grace_seconds: int = 10
    ):
        self.restart_backoff_seconds = restart_backoff_seconds
        self.max_restart_attempts = max_restart_attempts
        self.deregister_grace_seconds = deregister_grace_seconds
        
        # Track restart attempts per worker/job
        self.restart_managers: Dict[str, BackoffManager] = {}
        self.lock = Lock()
    
    def should_restart(self, worker_key: str) -> bool:
        """Check if a worker should be restarted."""
        with self.lock:
            if worker_key not in self.restart_managers:
                # First failure, create backoff manager
                self.restart_managers[worker_key] = BackoffManager(
                    strategy="exponential",
                    base_delay=self.restart_backoff_seconds,
                    max_delay=min(3600, self.restart_backoff_seconds * 10),
                    max_attempts=self.max_restart_attempts
                )
            
            manager = self.restart_managers[worker_key]
            return manager.should_retry()
    
    def get_restart_delay(self, worker_key: str) -> Optional[float]:
        """Get the delay before next restart attempt."""
        with self.lock:
            manager = self.restart_managers.get(worker_key)
            if not manager:
                return None
            
            return manager.next_delay()
    
    def wait_for_restart(self, worker_key: str) -> bool:
        """Wait for the restart backoff period. Returns True if should proceed."""
        with self.lock:
            manager = self.restart_managers.get(worker_key)
            if not manager:
                return True
            
            return manager.wait()
    
    def reset_restart_count(self, worker_key: str) -> None:
        """Reset restart count for a worker (on successful startup)."""
        with self.lock:
            if worker_key in self.restart_managers:
                self.restart_managers[worker_key].reset()
                logger.debug(f"Reset restart count for {worker_key}")
    
    def remove_worker(self, worker_key: str) -> None:
        """Remove restart tracking for a worker."""
        with self.lock:
            if worker_key in self.restart_managers:
                del self.restart_managers[worker_key]


class CircuitBreaker:
    """Circuit breaker to prevent cascading failures."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 300,
        half_open_max_calls: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half-open
        self.half_open_calls = 0
        
        self.lock = Lock()
    
    def call_allowed(self) -> bool:
        """Check if a call is allowed through the circuit breaker."""
        with self.lock:
            current_time = time.time()
            
            if self.state == "closed":
                return True
            
            elif self.state == "open":
                # Check if enough time has passed to try again
                if (self.last_failure_time and 
                    current_time - self.last_failure_time >= self.recovery_timeout):
                    self.state = "half-open"
                    self.half_open_calls = 0
                    logger.info("Circuit breaker entering half-open state")
                    return True
                return False
            
            elif self.state == "half-open":
                # Allow limited calls in half-open state
                return self.half_open_calls < self.half_open_max_calls
            
            return False
    
    def record_success(self) -> None:
        """Record a successful operation."""
        with self.lock:
            if self.state == "half-open":
                self.half_open_calls += 1
                # If we've had enough successful calls, close the circuit
                if self.half_open_calls >= self.half_open_max_calls:
                    self.state = "closed"
                    self.failure_count = 0
                    logger.info("Circuit breaker closed after successful recovery")
            elif self.state == "closed":
                # Reset failure count on success
                self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self) -> None:
        """Record a failed operation."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == "closed" and self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.warning(
                    f"Circuit breaker opened after {self.failure_count} failures"
                )
            elif self.state == "half-open":
                # Failure in half-open state, go back to open
                self.state = "open"
                logger.warning("Circuit breaker reopened after failure in half-open state")
    
    def get_state(self) -> str:
        """Get current circuit breaker state."""
        with self.lock:
            return self.state
    
    def force_open(self) -> None:
        """Manually open the circuit breaker."""
        with self.lock:
            self.state = "open"
            self.last_failure_time = time.time()
            logger.info("Circuit breaker manually opened")
    
    def reset(self) -> None:
        """Reset the circuit breaker to closed state."""
        with self.lock:
            self.state = "closed"
            self.failure_count = 0
            self.half_open_calls = 0
            self.last_failure_time = None
            logger.info("Circuit breaker reset to closed state")