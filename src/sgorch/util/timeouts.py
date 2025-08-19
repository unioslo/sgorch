import asyncio
import time
from contextlib import contextmanager
from typing import Optional, Any, Generator


class TimeoutError(Exception):
    """Raised when an operation times out."""
    pass


@contextmanager
def timeout_context(seconds: Optional[float]) -> Generator[None, None, None]:
    """Context manager for timing out operations."""
    if seconds is None or seconds <= 0:
        yield
        return
    
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        if elapsed > seconds:
            raise TimeoutError(f"Operation timed out after {elapsed:.2f}s (limit: {seconds}s)")


def with_timeout(func, *args, timeout_seconds: Optional[float] = None, **kwargs) -> Any:
    """Execute a function with a timeout."""
    if timeout_seconds is None or timeout_seconds <= 0:
        return func(*args, **kwargs)
    
    start_time = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start_time
    
    if elapsed > timeout_seconds:
        raise TimeoutError(f"Operation timed out after {elapsed:.2f}s (limit: {timeout_seconds}s)")
    
    return result


class Timer:
    """Simple timer utility."""
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.time()
        self.end_time = None
    
    def stop(self) -> float:
        """Stop the timer and return elapsed time."""
        if self.start_time is None:
            raise ValueError("Timer not started")
        
        self.end_time = time.time()
        return self.elapsed()
    
    def elapsed(self) -> float:
        """Get elapsed time (whether timer is still running or stopped)."""
        if self.start_time is None:
            return 0.0
        
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time
    
    def is_running(self) -> bool:
        """Check if timer is currently running."""
        return self.start_time is not None and self.end_time is None