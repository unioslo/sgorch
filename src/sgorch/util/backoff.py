import random
import time
from typing import Optional, Iterator


def exponential_backoff(
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    multiplier: float = 2.0,
    jitter: bool = True,
    max_attempts: Optional[int] = None
) -> Iterator[float]:
    """Generate exponential backoff delays with optional jitter."""
    attempt = 0
    current_delay = base_delay
    
    while max_attempts is None or attempt < max_attempts:
        if attempt > 0:  # No delay on first attempt
            delay = min(current_delay, max_delay)
            
            if jitter:
                # Add ±25% jitter
                jitter_range = delay * 0.25
                delay += random.uniform(-jitter_range, jitter_range)
                delay = max(0, delay)  # Ensure non-negative
            
            yield delay
            current_delay *= multiplier
        else:
            yield 0.0
        
        attempt += 1


def linear_backoff(
    initial_delay: float = 1.0,
    increment: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    max_attempts: Optional[int] = None
) -> Iterator[float]:
    """Generate linear backoff delays with optional jitter."""
    attempt = 0
    current_delay = initial_delay
    
    while max_attempts is None or attempt < max_attempts:
        if attempt > 0:  # No delay on first attempt
            delay = min(current_delay, max_delay)
            
            if jitter:
                # Add ±25% jitter
                jitter_range = delay * 0.25
                delay += random.uniform(-jitter_range, jitter_range)
                delay = max(0, delay)  # Ensure non-negative
            
            yield delay
            current_delay += increment
        else:
            yield 0.0
        
        attempt += 1


class BackoffManager:
    """Manages backoff state for retry operations."""
    
    def __init__(
        self,
        strategy: str = "exponential",
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        multiplier: float = 2.0,
        increment: float = 1.0,
        jitter: bool = True,
        max_attempts: Optional[int] = None
    ):
        self.strategy = strategy
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.increment = increment
        self.jitter = jitter
        self.max_attempts = max_attempts
        
        self.attempt_count = 0
        self.last_attempt_time: Optional[float] = None
        self._backoff_iterator: Optional[Iterator[float]] = None
    
    def reset(self) -> None:
        """Reset backoff state."""
        self.attempt_count = 0
        self.last_attempt_time = None
        self._backoff_iterator = None
    
    def should_retry(self) -> bool:
        """Check if we should retry based on max_attempts."""
        if self.max_attempts is None:
            return True
        return self.attempt_count < self.max_attempts
    
    def next_delay(self) -> Optional[float]:
        """Get the next delay duration, or None if max attempts exceeded."""
        if not self.should_retry():
            return None
        
        if self._backoff_iterator is None:
            if self.strategy == "exponential":
                self._backoff_iterator = exponential_backoff(
                    base_delay=self.base_delay,
                    max_delay=self.max_delay,
                    multiplier=self.multiplier,
                    jitter=self.jitter,
                    max_attempts=self.max_attempts
                )
            elif self.strategy == "linear":
                self._backoff_iterator = linear_backoff(
                    initial_delay=self.base_delay,
                    increment=self.increment,
                    max_delay=self.max_delay,
                    jitter=self.jitter,
                    max_attempts=self.max_attempts
                )
            else:
                raise ValueError(f"Unknown backoff strategy: {self.strategy}")
        
        try:
            delay = next(self._backoff_iterator)
            self.attempt_count += 1
            self.last_attempt_time = time.time()
            return delay
        except StopIteration:
            return None
    
    def wait(self) -> bool:
        """Wait for the next retry delay. Returns True if should continue, False if max attempts reached."""
        delay = self.next_delay()
        if delay is None:
            return False
        
        if delay > 0:
            time.sleep(delay)
        
        return True


def retry_with_backoff(
    func,
    *args,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    strategy: str = "exponential",
    exceptions: tuple = (Exception,),
    **kwargs
):
    """Retry a function with backoff on specified exceptions."""
    backoff = BackoffManager(
        strategy=strategy,
        base_delay=base_delay,
        max_delay=max_delay,
        max_attempts=max_attempts
    )
    
    last_exception = None
    
    while backoff.should_retry():
        try:
            return func(*args, **kwargs)
        except exceptions as e:
            last_exception = e
            
            if not backoff.should_retry():
                break
            
            delay = backoff.next_delay()
            if delay is None:
                break
            
            if delay > 0:
                time.sleep(delay)
    
    # If we get here, all retries failed
    if last_exception:
        raise last_exception
    else:
        raise RuntimeError(f"Function failed after {max_attempts} attempts")