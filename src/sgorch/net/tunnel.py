import subprocess
import time
import signal
import os
from dataclasses import dataclass
from typing import Optional, Dict, List, Literal
from threading import Lock

from ..logging_setup import get_logger
from ..util.backoff import BackoffManager
from .hostaddr import test_tcp_connection


logger = get_logger(__name__)


@dataclass
class TunnelSpec:
    """Specification for an SSH tunnel."""
    mode: Literal["local", "reverse"]   # ssh -L vs ssh -R
    orchestrator_host: str             # VM hostname (for reverse tunnels)
    advertise_host: str                # advertised host (e.g., vm-ip or 127.0.0.1)
    advertise_port: int                # local (for -L) or router port (for -R)
    remote_host: str                   # compute node IP/host
    remote_port: int                   # sglang worker port
    ssh_user: Optional[str] = None
    ssh_opts: List[str] = None
    
    def __post_init__(self):
        if self.ssh_opts is None:
            self.ssh_opts = []


@dataclass
class TunnelInfo:
    """Information about an active tunnel."""
    spec: TunnelSpec
    process: subprocess.Popen
    started_at: float
    last_check: float
    check_failures: int = 0
    advertised_url: str = ""
    
    def __post_init__(self):
        if not self.advertised_url:
            self.advertised_url = f"http://{self.spec.advertise_host}:{self.spec.advertise_port}"


class TunnelManager:
    """Manages SSH tunnels with supervision and automatic restart."""
    
    def __init__(self):
        self.tunnels: Dict[str, TunnelInfo] = {}
        self.lock = Lock()
        self._shutdown = False
    
    def ensure(self, key: str, spec: TunnelSpec) -> str:
        """
        Ensure a tunnel exists and is healthy.
        
        Args:
            key: Unique identifier for the tunnel
            spec: Tunnel specification
            
        Returns:
            The advertised URL for the tunnel
        """
        with self.lock:
            existing = self.tunnels.get(key)
            
            # If tunnel exists and is healthy, return it
            if existing and self._is_tunnel_healthy(existing):
                return existing.advertised_url
            
            # Stop existing tunnel if unhealthy
            if existing:
                self._stop_tunnel(key, existing)
            
            # Create new tunnel
            tunnel_info = self._create_tunnel(spec)
            self.tunnels[key] = tunnel_info
            
            logger.info(f"Tunnel {key} established: {tunnel_info.advertised_url}")
            return tunnel_info.advertised_url
    
    def is_up(self, key: str) -> bool:
        """Check if a tunnel is up and healthy."""
        with self.lock:
            tunnel = self.tunnels.get(key)
            if not tunnel:
                return False
            
            return self._is_tunnel_healthy(tunnel)
    
    def drop(self, key: str) -> None:
        """Drop a tunnel."""
        with self.lock:
            tunnel = self.tunnels.get(key)
            if tunnel:
                self._stop_tunnel(key, tunnel)
                del self.tunnels[key]
                logger.info(f"Tunnel {key} dropped")
    
    def gc(self) -> None:
        """Garbage collect dead tunnels."""
        with self.lock:
            to_remove = []
            
            for key, tunnel in self.tunnels.items():
                if not self._is_process_alive(tunnel.process):
                    logger.warning(f"Tunnel {key} process died, cleaning up")
                    to_remove.append(key)
            
            for key in to_remove:
                tunnel = self.tunnels[key]
                self._cleanup_process(tunnel.process)
                del self.tunnels[key]
    
    def get_tunnel_url(self, key: str) -> Optional[str]:
        """Get the advertised URL for a tunnel."""
        with self.lock:
            tunnel = self.tunnels.get(key)
            return tunnel.advertised_url if tunnel else None
    
    def shutdown(self) -> None:
        """Shutdown all tunnels."""
        self._shutdown = True
        
        with self.lock:
            logger.info(f"Shutting down {len(self.tunnels)} tunnels")
            
            for key, tunnel in self.tunnels.items():
                self._stop_tunnel(key, tunnel)
            
            self.tunnels.clear()
    
    def _create_tunnel(self, spec: TunnelSpec) -> TunnelInfo:
        """Create and start a new SSH tunnel."""
        cmd = self._build_ssh_command(spec)
        
        logger.debug(f"Starting tunnel: {' '.join(cmd)}")
        
        try:
            # Start SSH process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                preexec_fn=os.setsid  # Create new process group
            )
            
            # Give tunnel time to establish
            time.sleep(2.0)
            
            # Check if process is still alive
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                raise RuntimeError(
                    f"SSH tunnel failed to start: {stderr.decode()}"
                )
            
            tunnel_info = TunnelInfo(
                spec=spec,
                process=process,
                started_at=time.time(),
                last_check=time.time(),
                advertised_url=f"http://{spec.advertise_host}:{spec.advertise_port}"
            )
            
            return tunnel_info
            
        except Exception as e:
            logger.error(f"Failed to create tunnel: {e}")
            raise
    
    def _build_ssh_command(self, spec: TunnelSpec) -> List[str]:
        """Build SSH command for the tunnel."""
        cmd = ["ssh", "-N"]  # -N = no remote command
        
        # Add SSH options
        default_opts = [
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            "-o", "ExitOnForwardFailure=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "BatchMode=yes"  # Non-interactive
        ]
        
        cmd.extend(default_opts)
        
        # Add user-specified options
        if spec.ssh_opts:
            cmd.extend(spec.ssh_opts)
        
        # Add tunnel-specific options
        if spec.mode == "local":
            # Local port forwarding: -L local_port:remote_host:remote_port
            forward_spec = f"{spec.advertise_port}:{spec.remote_host}:{spec.remote_port}"
            cmd.extend(["-L", forward_spec])
            
            # SSH to the remote host (compute node)
            if spec.ssh_user:
                target = f"{spec.ssh_user}@{spec.remote_host}"
            else:
                target = spec.remote_host
                
        elif spec.mode == "reverse":
            # Reverse port forwarding: -R remote_port:local_host:local_port
            forward_spec = f"{spec.advertise_port}:localhost:{spec.remote_port}"
            cmd.extend(["-R", forward_spec])
            
            # SSH to orchestrator host (where router runs)
            if spec.ssh_user:
                target = f"{spec.ssh_user}@{spec.orchestrator_host}"
            else:
                target = spec.orchestrator_host
        else:
            raise ValueError(f"Unknown tunnel mode: {spec.mode}")
        
        cmd.append(target)
        return cmd
    
    def _is_tunnel_healthy(self, tunnel: TunnelInfo) -> bool:
        """Check if a tunnel is healthy."""
        # Check if process is alive
        if not self._is_process_alive(tunnel.process):
            return False
        
        # For local tunnels, test the local port
        if tunnel.spec.mode == "local":
            return test_tcp_connection("127.0.0.1", tunnel.spec.advertise_port, timeout=2)
        
        # For reverse tunnels, we trust that the process being alive is sufficient
        # (we can't easily test from this end)
        return True
    
    def _is_process_alive(self, process: subprocess.Popen) -> bool:
        """Check if a process is still running."""
        return process.poll() is None
    
    def _stop_tunnel(self, key: str, tunnel: TunnelInfo) -> None:
        """Stop a tunnel process."""
        logger.debug(f"Stopping tunnel {key}")
        
        try:
            self._cleanup_process(tunnel.process)
        except Exception as e:
            logger.warning(f"Error stopping tunnel {key}: {e}")
    
    def _cleanup_process(self, process: subprocess.Popen) -> None:
        """Clean up a process, trying graceful shutdown first."""
        if process.poll() is not None:
            # Process already dead
            return
        
        try:
            # Try graceful shutdown first (SIGTERM)
            if hasattr(process, 'send_signal'):
                process.send_signal(signal.SIGTERM)
            
            # Wait a bit for graceful shutdown
            try:
                process.wait(timeout=5)
                return
            except subprocess.TimeoutExpired:
                pass
            
            # Force kill if still alive
            process.kill()
            process.wait(timeout=5)
            
        except Exception as e:
            logger.warning(f"Error during process cleanup: {e}")


class SupervisedTunnelManager(TunnelManager):
    """
    TunnelManager with automatic restart and health monitoring.
    """
    
    def __init__(self, check_interval: float = 30.0, max_restart_attempts: int = 5):
        super().__init__()
        self.check_interval = check_interval
        self.max_restart_attempts = max_restart_attempts
        self.restart_backoffs: Dict[str, BackoffManager] = {}
    
    def ensure(self, key: str, spec: TunnelSpec) -> str:
        """Ensure a tunnel with automatic restart capability."""
        result = super().ensure(key, spec)
        
        # Initialize backoff manager for this tunnel
        if key not in self.restart_backoffs:
            self.restart_backoffs[key] = BackoffManager(
                strategy="exponential",
                base_delay=5.0,
                max_delay=300.0,
                max_attempts=self.max_restart_attempts
            )
        
        return result
    
    def monitor_and_restart(self) -> Dict[str, str]:
        """
        Monitor all tunnels and restart failed ones.
        
        Returns:
            Dict of tunnel_key -> status
        """
        if self._shutdown:
            return {}
        
        results = {}
        
        with self.lock:
            for key, tunnel in list(self.tunnels.items()):
                try:
                    if self._is_tunnel_healthy(tunnel):
                        # Tunnel is healthy
                        tunnel.last_check = time.time()
                        tunnel.check_failures = 0
                        results[key] = "healthy"
                        
                        # Reset backoff on success
                        if key in self.restart_backoffs:
                            self.restart_backoffs[key].reset()
                    else:
                        # Tunnel is unhealthy
                        tunnel.check_failures += 1
                        
                        logger.warning(
                            f"Tunnel {key} unhealthy "
                            f"(failure #{tunnel.check_failures})"
                        )
                        
                        # Attempt restart
                        success = self._attempt_restart(key, tunnel)
                        results[key] = "restarted" if success else "failed"
                        
                except Exception as e:
                    logger.error(f"Error monitoring tunnel {key}: {e}")
                    results[key] = "error"
        
        return results
    
    def _attempt_restart(self, key: str, tunnel: TunnelInfo) -> bool:
        """Attempt to restart a failed tunnel."""
        backoff = self.restart_backoffs.get(key)
        if not backoff:
            return False
        
        if not backoff.should_retry():
            logger.error(f"Tunnel {key} exceeded max restart attempts")
            return False
        
        # Get delay for this attempt
        delay = backoff.next_delay()
        if delay is None:
            return False
        
        # Wait if needed
        if delay > 0:
            logger.info(f"Waiting {delay:.1f}s before restarting tunnel {key}")
            time.sleep(delay)
        
        try:
            # Stop old tunnel
            self._stop_tunnel(key, tunnel)
            
            # Create new tunnel
            new_tunnel = self._create_tunnel(tunnel.spec)
            self.tunnels[key] = new_tunnel
            
            logger.info(f"Successfully restarted tunnel {key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restart tunnel {key}: {e}")
            return False
    
    def drop(self, key: str) -> None:
        """Drop a tunnel and clean up its restart tracking."""
        super().drop(key)
        
        # Clean up restart tracking
        if key in self.restart_backoffs:
            del self.restart_backoffs[key]