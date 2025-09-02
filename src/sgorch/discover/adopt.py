import re
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass

from ..logging_setup import get_logger
from ..config import DeploymentConfig
from ..slurm.base import ISlurm, JobInfo
from ..router.client import RouterClient
from ..net.hostaddr import resolve_slurm_node_ip
from ..net.ports import PortAllocator


logger = get_logger(__name__)


@dataclass
class DiscoveredWorker:
    """Information about a discovered worker."""
    job_id: str
    instance_uuid: str
    worker_url: str
    # Stable replica index if known (parsed from logs/job name)
    instance_idx: int | None = None
    node: Optional[str] = None
    remote_port: Optional[int] = None
    advertise_port: Optional[int] = None
    log_file: Optional[str] = None


class WorkerDiscovery:
    """Discovers existing workers for graceful resumption."""
    
    def __init__(self, deployment_config: DeploymentConfig, slurm: ISlurm):
        self.config = deployment_config
        self.slurm = slurm
        self.logger = logger.bind(deployment=deployment_config.name)
    
    def discover_workers(self) -> List[DiscoveredWorker]:
        """
        Discover existing workers for this deployment.
        
        Returns:
            List of discovered workers
        """
        self.logger.info("Starting worker discovery")
        
        # Step 1: Get SLURM jobs for this deployment
        job_prefix = f"sgl-{self.config.name}-"
        jobs = self.slurm.list_jobs(job_prefix)
        
        self.logger.info(f"Found {len(jobs)} SLURM jobs with prefix {job_prefix}")
        
        # Step 2: Process each running job
        discovered = []
        for job in jobs:
            if job.state == "RUNNING":
                worker = self._discover_worker_from_job(job)
                if worker:
                    discovered.append(worker)
        
        self.logger.info(f"Discovered {len(discovered)} workers")
        return discovered
    
    def _discover_worker_from_job(self, job: JobInfo) -> Optional[DiscoveredWorker]:
        """Discover worker information from a SLURM job."""
        try:
            # Try to extract information from log files
            worker = self._extract_from_logs(job)
            if worker:
                return worker
            
            # Fallback: compute from job info and config
            return self._compute_from_job_info(job)
            
        except Exception as e:
            self.logger.warning(f"Failed to discover worker from job {job.job_id}: {e}")
            return None
    
    def _extract_from_logs(self, job: JobInfo) -> Optional[DiscoveredWorker]:
        """Extract worker info from SLURM job logs."""
        log_dir = Path(self.config.slurm.log_dir)
        
        # Look for job output files
        log_patterns = [
            f"sgl-{self.config.name}-*_{job.job_id}.out",  # SLURM output
            f"server_{job.job_id}.log"  # SGLang server log
        ]
        
        for pattern in log_patterns:
            log_files = list(log_dir.glob(pattern))
            
            for log_file in log_files:
                worker = self._parse_log_file(log_file, job)
                if worker:
                    # Try to parse instance index from filename: sgl-<name>-<idx>_<jobid>.out
                    try:
                        import re
                        m = re.match(rf"sgl-{re.escape(self.config.name)}-(\d+)_\d+\.out$", log_file.name)
                        if m:
                            worker.instance_idx = int(m.group(1))
                    except Exception:
                        pass
                    return worker
        
        return None
    
    def _parse_log_file(self, log_file: Path, job: JobInfo) -> Optional[DiscoveredWorker]:
        """Parse a log file to extract worker information."""
        try:
            # Look for the READY marker line
            # Format: "READY URL=http://IP:PORT JOB=JOBID INSTANCE=UUID"
            ready_pattern = re.compile(
                r'READY URL=(\S+) JOB=(\S+) INSTANCE=(\S+)'
            )
            
            with log_file.open('r') as f:
                # Read last 1000 lines (worker might have been running for a while)
                lines = f.readlines()
                for line in reversed(lines[-1000:]):
                    match = ready_pattern.search(line)
                    if match:
                        worker_url, log_job_id, instance_uuid = match.groups()
                        
                        # Verify job ID matches
                        if log_job_id == job.job_id:
                            # Extract port from URL
                            url_parts = worker_url.split(':')
                            remote_port = int(url_parts[-1]) if len(url_parts) >= 3 else None
                            
                            return DiscoveredWorker(
                                job_id=job.job_id,
                                instance_uuid=instance_uuid,
                                worker_url=worker_url,
                                instance_idx=None,
                                node=job.node,
                                remote_port=remote_port,
                                log_file=str(log_file)
                            )
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error parsing log file {log_file}: {e}")
            return None
    
    def _compute_from_job_info(self, job: JobInfo) -> Optional[DiscoveredWorker]:
        """Compute worker info from job information and configuration."""
        if not job.node:
            self.logger.warning(f"Job {job.job_id} has no node information")
            return None
        
        # Resolve node IP
        node_ip = resolve_slurm_node_ip(job.node)
        if not node_ip:
            self.logger.warning(f"Could not resolve IP for node {job.node}")
            return None
        
        # For recovery, we need to guess the port
        # This is challenging without log parsing
        # We'll use a heuristic based on typical SGLang port ranges
        remote_port = self._guess_worker_port(job, node_ip)
        if not remote_port:
            return None
        
        worker_url = f"http://{node_ip}:{remote_port}"
        
        # Generate a placeholder UUID since we don't have the real one
        instance_uuid = f"recovered-{job.job_id}"
        
        return DiscoveredWorker(
            job_id=job.job_id,
            instance_uuid=instance_uuid,
            worker_url=worker_url,
            node=job.node,
            remote_port=remote_port
        )
    
    def _guess_worker_port(self, job: JobInfo, node_ip: str) -> Optional[int]:
        """Guess the worker port (fallback when logs are unavailable)."""
        # Try common SGLang ports
        common_ports = [8000, 8001, 8080, 8888, 30000, 30001]
        
        from ..net.hostaddr import test_tcp_connection
        
        for port in common_ports:
            if test_tcp_connection(node_ip, port, timeout=2):
                self.logger.debug(f"Found open port {port} on {node_ip}")
                return port
        
        # Try scanning a range (be conservative to avoid being intrusive)
        for port in range(8000, 8010):
            if test_tcp_connection(node_ip, port, timeout=1):
                return port
        
        return None


class AdoptionManager:
    """Manages adoption of existing workers during startup."""
    
    def __init__(
        self,
        deployment_config: DeploymentConfig,
        slurm: ISlurm,
        router_client: RouterClient,
        port_allocator: PortAllocator
    ):
        self.config = deployment_config
        self.slurm = slurm
        self.router_client = router_client
        self.port_allocator = port_allocator
        self.logger = logger.bind(deployment=deployment_config.name)
    
    def adopt_existing_workers(self) -> Dict[str, DiscoveredWorker]:
        """
        Adopt existing workers and reconstruct state.
        
        Returns:
            Dict mapping job_id -> DiscoveredWorker for adopted workers
        """
        self.logger.info("Starting worker adoption")
        
        # Discover workers
        discovery = WorkerDiscovery(self.config, self.slurm)
        discovered_workers = discovery.discover_workers()
        
        if not discovered_workers:
            self.logger.info("No existing workers found to adopt")
            return {}
        
        # Process each discovered worker
        adopted = {}
        router_workers = self._get_router_workers()
        
        for worker in discovered_workers:
            if self._adopt_worker(worker, router_workers):
                adopted[worker.job_id] = worker
        
        self.logger.info(f"Successfully adopted {len(adopted)} workers")
        return adopted
    
    def _adopt_worker(
        self,
        worker: DiscoveredWorker,
        router_workers: Set[str]
    ) -> bool:
        """Adopt a single worker."""
        try:
            self.logger.info(f"Adopting worker {worker.job_id} at {worker.worker_url}")
            
            # Mark ports as in use to prevent collisions
            if worker.remote_port:
                self.port_allocator.mark_in_use(worker.remote_port)
            
            if worker.advertise_port:
                self.port_allocator.mark_in_use(worker.advertise_port)
            
            # Check if worker is registered with router
            if worker.worker_url not in router_workers:
                self.logger.info(
                    f"Worker {worker.job_id} not in router, will be handled by reconciler"
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to adopt worker {worker.job_id}: {e}")
            return False
    
    def _get_router_workers(self) -> Set[str]:
        """Get current workers registered with the router."""
        try:
            return self.router_client.list()
        except Exception as e:
            self.logger.warning(f"Failed to get router workers: {e}")
            return set()
    
    def reconcile_router_state(self, adopted_workers: Dict[str, DiscoveredWorker]) -> None:
        """Reconcile router state with adopted workers."""
        try:
            router_workers = self._get_router_workers()
            adopted_urls = {w.worker_url for w in adopted_workers.values()}
            
            # Find router workers that don't correspond to adopted workers
            stale_workers = router_workers - adopted_urls

            # Be conservative: only touch URLs that look like ours (advertise host + port range)
            safe_to_remove: Set[str] = set()
            try:
                from urllib.parse import urlparse
                adv_host = self.config.connectivity.advertise_host
                pr_min, pr_max = self.config.connectivity.local_port_range
                for worker_url in stale_workers:
                    try:
                        parsed = urlparse(worker_url)
                        host = parsed.hostname
                        port = parsed.port
                    except Exception:
                        host, port = None, None
                    if host in {adv_host, "127.0.0.1", "localhost"} and port and pr_min <= port <= pr_max:
                        safe_to_remove.add(worker_url)
            except Exception:
                # If parsing fails, err on the side of not removing anything
                safe_to_remove = set()

            if safe_to_remove:
                self.logger.warning(
                    f"Found {len(safe_to_remove)} stale workers in router scoped to this deployment"
                )

                # Remove stale workers considered ours
                for worker_url in safe_to_remove:
                    try:
                        self.router_client.remove(worker_url)
                        self.logger.info(f"Removed stale worker from router: {worker_url}")
                    except Exception as e:
                        self.logger.warning(f"Failed to remove stale worker: {e}")
            
            # Check if any adopted workers need to be registered
            for worker in adopted_workers.values():
                if worker.worker_url not in router_workers:
                    self.logger.info(
                        f"Worker {worker.job_id} needs router registration"
                    )
        
        except Exception as e:
            self.logger.error(f"Failed to reconcile router state: {e}")


def parse_job_comment(comment: str) -> Optional[Tuple[str, str]]:
    """
    Parse SLURM job comment to extract deployment and instance UUID.
    
    Comment format: "sgorch:DEPLOYMENT:UUID"
    
    Returns:
        (deployment_name, instance_uuid) or None if parsing failed
    """
    if not comment:
        return None
    
    parts = comment.split(':')
    if len(parts) == 3 and parts[0] == 'sgorch':
        return parts[1], parts[2]
    
    return None
