import json
import re
import subprocess
import time
import threading
from typing import Optional, Dict, List, NamedTuple
from dataclasses import dataclass

from ..logging_setup import get_logger
from ..config import DeploymentConfig, GPUMonitorConfig
from ..metrics.prometheus import get_metrics


logger = get_logger(__name__)


@dataclass
class GPUInfo:
    """Information about a single GPU."""
    gpu_id: str
    utilization: Optional[float] = None
    memory_used: Optional[int] = None  
    memory_total: Optional[int] = None
    temperature: Optional[float] = None
    power_draw: Optional[float] = None
    fan_speed: Optional[float] = None


class GPUMonitorWorker(threading.Thread):
    """Background worker that monitors GPU metrics on SLURM compute nodes."""

    def __init__(self, deployment_config: DeploymentConfig, gpu_config: GPUMonitorConfig):
        super().__init__(name=f"gpu-monitor-{deployment_config.name}", daemon=True)
        self.deployment_config = deployment_config
        self.gpu_config = gpu_config
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        metrics = get_metrics()
        interval = max(5, int(self.gpu_config.interval_s))

        while not self._stop_event.is_set():
            try:
                # Get nodes and their corresponding job IDs for this deployment
                node_to_job = self._get_deployment_jobs()
                logger.debug(f"GPU monitor found {len(node_to_job)} jobs for {self.deployment_config.name}: {node_to_job}")
                
                for node, job_id in node_to_job.items():
                    success = False
                    try:
                        gpu_infos = self._query_gpu_metrics_by_job(job_id, node)
                        success = len(gpu_infos) > 0
                        
                        if success:
                            for gpu_info in gpu_infos:
                                metrics.record_gpu_metrics(
                                    deployment=self.deployment_config.name,
                                    node=node,
                                    gpu_id=gpu_info.gpu_id,
                                    utilization=gpu_info.utilization,
                                    memory_used=gpu_info.memory_used,
                                    memory_total=gpu_info.memory_total,
                                    temperature=gpu_info.temperature,
                                    power_draw=gpu_info.power_draw,
                                    fan_speed=gpu_info.fan_speed
                                )
                        
                        metrics.record_gpu_monitor_attempt(
                            self.deployment_config.name, 
                            node, 
                            success
                        )
                        
                    except Exception as e:
                        logger.debug(f"GPU monitoring failed for {self.deployment_config.name} on {node} (job {job_id}): {e}")
                        metrics.record_gpu_monitor_attempt(
                            self.deployment_config.name, 
                            node, 
                            False
                        )
                
            except Exception as e:
                logger.error(f"GPU monitor error for {self.deployment_config.name}: {e}")
            
            self._stop_event.wait(interval)

    def _get_deployment_jobs(self) -> Dict[str, str]:
        """Get mapping of nodes to job IDs for this deployment's running jobs."""
        job_prefix = f"sgl-{self.deployment_config.name}-"
        
        try:
            # Use squeue to find running jobs for this deployment with both job ID and node info
            cmd = [
                'squeue', 
                '--format=%i,%N',  # JobID,NodeList
                '--noheader',
                '--states=RUNNING',
                f'--name={job_prefix}*'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.gpu_config.timeout_s
            )
            
            if result.returncode != 0:
                logger.debug(f"squeue failed: {result.stderr}")
                return {}
            
            # Parse job info: JobID,NodeList
            node_to_job = {}
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                    
                parts = line.strip().split(',', 1)
                if len(parts) != 2:
                    continue
                    
                job_id = parts[0].strip()
                node_spec = parts[1].strip()
                
                # Expand node list - SLURM can return ranges like "node[01-02]"
                nodes = self._expand_node_list(node_spec)
                for node in nodes:
                    node_to_job[node] = job_id
            
            return node_to_job
            
        except Exception as e:
            logger.debug(f"Failed to get deployment jobs: {e}")
            return {}

    def _expand_node_list(self, node_spec: str) -> List[str]:
        """Expand SLURM node specification like 'node[01-02]' into individual nodes."""
        # Simple regex to handle basic node range expansion
        # For more complex cases, could use hostlist library if needed
        match = re.match(r'(\w+)\[(\d+)-(\d+)\]', node_spec)
        if match:
            prefix = match.group(1)
            start = int(match.group(2))
            end = int(match.group(3))
            width = len(match.group(2))  # Preserve zero-padding
            return [f"{prefix}{str(i).zfill(width)}" for i in range(start, end + 1)]
        else:
            # Single node or comma-separated list
            return [node.strip() for node in node_spec.split(',')]

    def _query_gpu_metrics_by_job(self, job_id: str, node: str) -> List[GPUInfo]:
        """Query GPU metrics from a specific job using srun + nvidia-smi."""
        # Build srun command to run nvidia-smi on the existing job
        cmd = [
            'srun',
            '--jobid', job_id,
            '--quiet',  # Suppress srun output
            'nvidia-smi',
            '--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,fan.speed',
            '--format=csv,noheader,nounits'
        ]
        
        try:
            logger.debug(f"GPU monitor running command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.gpu_config.timeout_s
            )
            
            if result.returncode != 0:
                logger.debug(f"srun nvidia-smi failed for job {job_id} on {node}: {result.stderr}")
                return []
            
            return self._parse_nvidia_smi_output(result.stdout)
            
        except subprocess.TimeoutExpired:
            logger.debug(f"GPU monitoring timed out for job {job_id} on {node}")
            return []
        except Exception as e:
            logger.debug(f"GPU monitoring failed for job {job_id} on {node}: {e}")
            return []

    def _parse_nvidia_smi_output(self, output: str) -> List[GPUInfo]:
        """Parse nvidia-smi CSV output into GPUInfo objects."""
        gpu_infos = []
        
        for line in output.strip().split('\n'):
            if not line.strip():
                continue
                
            try:
                # Parse CSV line: index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,fan.speed
                parts = [p.strip() for p in line.split(',')]
                if len(parts) < 7:
                    continue
                
                gpu_info = GPUInfo(
                    gpu_id=parts[0],
                    utilization=self._safe_float(parts[1]),
                    memory_used=self._safe_int_mb_to_bytes(parts[2]),
                    memory_total=self._safe_int_mb_to_bytes(parts[3]),
                    temperature=self._safe_float(parts[4]),
                    power_draw=self._safe_float(parts[5]),
                    fan_speed=self._safe_float(parts[6])
                )
                
                gpu_infos.append(gpu_info)
                
            except Exception as e:
                logger.debug(f"Failed to parse nvidia-smi line '{line}': {e}")
                continue
        
        return gpu_infos

    def _safe_float(self, value: str) -> Optional[float]:
        """Safely convert string to float, handling 'N/A' and other invalid values."""
        if not value or value.lower() in ('n/a', 'not supported', '[n/a]'):
            return None
        try:
            return float(value)
        except ValueError:
            return None

    def _safe_int_mb_to_bytes(self, value: str) -> Optional[int]:
        """Safely convert string MB value to bytes, handling 'N/A' and other invalid values."""
        if not value or value.lower() in ('n/a', 'not supported', '[n/a]'):
            return None
        try:
            # nvidia-smi reports memory in MB, convert to bytes
            return int(float(value) * 1024 * 1024)
        except ValueError:
            return None


class GPUMonitorManager:
    """Manages GPU monitoring for all deployments."""

    def __init__(self, gpu_config: GPUMonitorConfig):
        self.gpu_config = gpu_config
        self._workers: Dict[str, GPUMonitorWorker] = {}

    def start_for(self, deployment: DeploymentConfig) -> None:
        """Start GPU monitoring for a deployment."""
        if not self.gpu_config.enabled:
            return
            
        name = deployment.name
        if name in self._workers:
            return
            
        worker = GPUMonitorWorker(deployment, self.gpu_config)
        self._workers[name] = worker
        worker.start()
        
        logger.info(f"GPU monitoring started for deployment: {name}")

    def stop_for(self, deployment_name: str) -> None:
        """Stop GPU monitoring for a deployment."""
        worker = self._workers.pop(deployment_name, None)
        if worker:
            worker.stop()
            worker.join(timeout=5.0)
            logger.info(f"GPU monitoring stopped for deployment: {deployment_name}")

    def stop_all(self) -> None:
        """Stop all GPU monitoring workers."""
        for name in list(self._workers.keys()):
            self.stop_for(name)
