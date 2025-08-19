import random
import socket
from typing import Optional, Set, Tuple
from threading import Lock

from ..logging_setup import get_logger


logger = get_logger(__name__)


class PortAllocationError(Exception):
    """Raised when port allocation fails."""
    pass


class PortAllocator:
    """Manages port allocation within a range to avoid collisions."""
    
    def __init__(self, port_range: Tuple[int, int]):
        self.min_port, self.max_port = port_range
        self.allocated_ports: Set[int] = set()
        self.lock = Lock()
        
        if self.min_port <= 0 or self.max_port <= 0:
            raise ValueError("Port range must contain positive integers")
        if self.min_port >= self.max_port:
            raise ValueError("Invalid port range: min >= max")
        
        logger.info(f"Port allocator initialized for range {self.min_port}-{self.max_port}")
    
    def is_port_available(self, port: int) -> bool:
        """Check if a port is available (not allocated and not in use)."""
        if port < self.min_port or port > self.max_port:
            return False
        
        with self.lock:
            if port in self.allocated_ports:
                return False
        
        # Check if port is actually available on the system
        return self._check_port_free(port)
    
    def _check_port_free(self, port: int, host: str = "127.0.0.1") -> bool:
        """Check if a port is free by attempting to bind to it."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((host, port))
                return True
        except (socket.error, OSError):
            return False
    
    def allocate_port(self, preferred_port: Optional[int] = None) -> int:
        """
        Allocate a port from the range.
        
        Args:
            preferred_port: Try this port first if specified
            
        Returns:
            Allocated port number
            
        Raises:
            PortAllocationError: If no ports are available
        """
        with self.lock:
            # Try preferred port first
            if preferred_port is not None:
                if self.is_port_available(preferred_port):
                    self.allocated_ports.add(preferred_port)
                    logger.debug(f"Allocated preferred port: {preferred_port}")
                    return preferred_port
                else:
                    logger.debug(f"Preferred port {preferred_port} not available")
            
            # Try to find an available port in the range
            available_ports = []
            for port in range(self.min_port, self.max_port + 1):
                if port not in self.allocated_ports and self._check_port_free(port):
                    available_ports.append(port)
            
            if not available_ports:
                raise PortAllocationError(
                    f"No available ports in range {self.min_port}-{self.max_port}"
                )
            
            # Randomly select from available ports to reduce predictability
            selected_port = random.choice(available_ports)
            self.allocated_ports.add(selected_port)
            
            logger.debug(f"Allocated port: {selected_port}")
            return selected_port
    
    def allocate_pair(self) -> Tuple[int, int]:
        """
        Allocate a pair of consecutive ports (useful for some applications).
        
        Returns:
            Tuple of (port1, port2) where port2 = port1 + 1
        """
        with self.lock:
            for port in range(self.min_port, self.max_port):
                if (port not in self.allocated_ports and 
                    (port + 1) not in self.allocated_ports and
                    port + 1 <= self.max_port and
                    self._check_port_free(port) and 
                    self._check_port_free(port + 1)):
                    
                    self.allocated_ports.add(port)
                    self.allocated_ports.add(port + 1)
                    
                    logger.debug(f"Allocated port pair: {port}, {port + 1}")
                    return (port, port + 1)
            
            raise PortAllocationError(
                f"No consecutive port pairs available in range {self.min_port}-{self.max_port}"
            )
    
    def release_port(self, port: int) -> None:
        """Release a previously allocated port."""
        with self.lock:
            if port in self.allocated_ports:
                self.allocated_ports.remove(port)
                logger.debug(f"Released port: {port}")
            else:
                logger.warning(f"Attempted to release non-allocated port: {port}")
    
    def release_ports(self, ports: Set[int]) -> None:
        """Release multiple ports at once."""
        with self.lock:
            for port in ports:
                self.allocated_ports.discard(port)
            logger.debug(f"Released {len(ports)} ports")
    
    def mark_in_use(self, port: int) -> None:
        """
        Mark a port as in use (for ports discovered during adoption).
        This prevents the allocator from allocating this port.
        """
        with self.lock:
            if self.min_port <= port <= self.max_port:
                self.allocated_ports.add(port)
                logger.debug(f"Marked port as in use: {port}")
            else:
                logger.warning(f"Port {port} is outside allocation range")
    
    def mark_ports_in_use(self, ports: Set[int]) -> None:
        """Mark multiple ports as in use."""
        with self.lock:
            for port in ports:
                if self.min_port <= port <= self.max_port:
                    self.allocated_ports.add(port)
            logger.debug(f"Marked {len(ports)} ports as in use")
    
    def get_allocated_ports(self) -> Set[int]:
        """Get a copy of all allocated ports."""
        with self.lock:
            return self.allocated_ports.copy()
    
    def get_available_count(self) -> int:
        """Get the number of available ports (approximate)."""
        total_ports = self.max_port - self.min_port + 1
        with self.lock:
            allocated_count = len(self.allocated_ports)
        return max(0, total_ports - allocated_count)
    
    def clear(self) -> None:
        """Clear all allocated ports (useful for testing)."""
        with self.lock:
            cleared_count = len(self.allocated_ports)
            self.allocated_ports.clear()
            logger.info(f"Cleared {cleared_count} allocated ports")
    
    def __len__(self) -> int:
        """Return number of allocated ports."""
        with self.lock:
            return len(self.allocated_ports)
    
    def __contains__(self, port: int) -> bool:
        """Check if a port is allocated."""
        with self.lock:
            return port in self.allocated_ports


def find_free_port(start_port: int = 49152, max_attempts: int = 100) -> int:
    """
    Find a free port starting from start_port.
    
    Args:
        start_port: Port to start searching from
        max_attempts: Maximum number of ports to try
        
    Returns:
        A free port number
        
    Raises:
        PortAllocationError: If no free port found
    """
    for attempt in range(max_attempts):
        port = start_port + attempt
        if port > 65535:
            break
        
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("127.0.0.1", port))
                return port
        except (socket.error, OSError):
            continue
    
    raise PortAllocationError(f"No free port found after {max_attempts} attempts")