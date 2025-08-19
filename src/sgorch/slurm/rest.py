import json
from typing import Dict, List, Optional, Any

import httpx

from ..logging_setup import get_logger
from .base import ISlurm, SubmitSpec, JobInfo, JobState


logger = get_logger(__name__)


class SlurmRestAdapter(ISlurm):
    """SLURM adapter using slurmrestd REST API."""
    
    def __init__(
        self, 
        base_url: str,
        auth_token: Optional[str] = None,
        timeout: int = 30
    ):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        
        # Set up httpx client with auth
        headers = {"Content-Type": "application/json"}
        if auth_token:
            headers["X-SLURM-USER-TOKEN"] = auth_token
            
        self.client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout
        )
    
    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, 'client'):
            self.client.close()
    
    def submit(self, spec: SubmitSpec) -> str:
        """Submit a job via REST API."""
        logger.info(f"Submitting job via REST API: {spec.name}")
        
        # Build job submission payload
        job_spec = {
            "job": {
                "name": spec.name,
                "account": spec.account,
                "partition": spec.partition,
                "gres": spec.gres,
                "cpus_per_task": spec.cpus_per_task,
                "memory": spec.mem,
                "time_limit": spec.time_limit,
                "standard_output": spec.stdout,
                "standard_error": spec.stderr,
                "environment": spec.env,
                "script": spec.script,
            }
        }
        
        # Add optional fields
        if spec.reservation:
            job_spec["job"]["reservation"] = spec.reservation
        if spec.qos:
            job_spec["job"]["qos"] = spec.qos
        if spec.constraint:
            job_spec["job"]["constraints"] = spec.constraint
        
        try:
            response = self.client.post("/slurm/v0.0.39/job/submit", json=job_spec)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("errors"):
                error_msg = f"Job submission failed: {data['errors']}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            job_id = str(data["job_id"])
            logger.info(f"Job submitted successfully via REST: {job_id}")
            return job_id
            
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error during job submission: {e.response.status_code} {e.response.text}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            logger.error(f"Error submitting job via REST: {e}")
            raise
    
    def status(self, job_id: str) -> JobInfo:
        """Get job status via REST API."""
        logger.debug(f"Getting job status via REST: {job_id}")
        
        try:
            response = self.client.get(f"/slurm/v0.0.39/job/{job_id}")
            
            if response.status_code == 404:
                # Job not found, might be completed/purged
                logger.warning(f"Job {job_id} not found in REST API")
                return JobInfo(
                    job_id=job_id,
                    state='UNKNOWN',
                    node=None,
                    time_left_s=None
                )
            
            response.raise_for_status()
            data = response.json()
            
            if data.get("errors"):
                logger.warning(f"REST API errors for job {job_id}: {data['errors']}")
                return JobInfo(
                    job_id=job_id,
                    state='UNKNOWN',
                    node=None,
                    time_left_s=None
                )
            
            job_data = data.get("jobs", [])
            if not job_data:
                return JobInfo(
                    job_id=job_id,
                    state='UNKNOWN',
                    node=None,
                    time_left_s=None
                )
            
            job = job_data[0]
            return self._parse_job_info(job)
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return JobInfo(
                    job_id=job_id,
                    state='UNKNOWN',
                    node=None,
                    time_left_s=None
                )
            logger.error(f"HTTP error getting job status: {e.response.status_code}")
            raise
        except Exception as e:
            logger.error(f"Error getting job status via REST: {e}")
            raise
    
    def cancel(self, job_id: str) -> None:
        """Cancel a job via REST API."""
        logger.info(f"Cancelling job via REST: {job_id}")
        
        try:
            response = self.client.delete(f"/slurm/v0.0.39/job/{job_id}")
            
            if response.status_code == 404:
                logger.warning(f"Job {job_id} not found for cancellation")
                return
            
            response.raise_for_status()
            
            data = response.json()
            if data.get("errors"):
                error_msg = f"Job cancellation failed: {data['errors']}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            logger.info(f"Job cancelled successfully via REST: {job_id}")
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"Job {job_id} not found for cancellation")
                return
            error_msg = f"HTTP error cancelling job: {e.response.status_code}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            logger.error(f"Error cancelling job via REST: {e}")
            raise
    
    def list_jobs(self, name_prefix: str) -> List[JobInfo]:
        """List jobs with name prefix via REST API."""
        logger.debug(f"Listing jobs via REST with prefix: {name_prefix}")
        
        try:
            # Get all jobs for the current user
            response = self.client.get("/slurm/v0.0.39/jobs")
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("errors"):
                logger.error(f"REST API errors listing jobs: {data['errors']}")
                return []
            
            jobs = []
            for job in data.get("jobs", []):
                job_name = job.get("name", "")
                if job_name.startswith(name_prefix):
                    jobs.append(self._parse_job_info(job))
            
            logger.debug(f"Found {len(jobs)} jobs with prefix {name_prefix}")
            return jobs
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error listing jobs: {e.response.status_code}")
            return []
        except Exception as e:
            logger.error(f"Error listing jobs via REST: {e}")
            return []
    
    def _parse_job_info(self, job_data: Dict[str, Any]) -> JobInfo:
        """Parse job data from REST API into JobInfo."""
        job_id = str(job_data["job_id"])
        
        # Parse job state
        job_state_str = job_data.get("job_state", "UNKNOWN")
        state = self._parse_rest_job_state(job_state_str)
        
        # Extract node information
        node = None
        nodes = job_data.get("nodes", "")
        if nodes and nodes not in ("", "None"):
            # Take first node from node list
            node = nodes.split(",")[0].split("[")[0]
        
        # Calculate time left
        time_left_s = None
        if state == "RUNNING":
            time_limit = job_data.get("time_limit", 0)
            start_time = job_data.get("start_time", 0)
            
            if time_limit and start_time:
                import time
                elapsed = time.time() - start_time
                time_left_s = max(0, time_limit * 60 - elapsed)  # time_limit in minutes
        
        return JobInfo(
            job_id=job_id,
            state=state,
            node=node,
            time_left_s=time_left_s
        )
    
    def _parse_rest_job_state(self, state_str: str) -> JobState:
        """Parse REST API job state to JobState."""
        # Map REST API states to our JobState enum
        state_mapping = {
            "PENDING": "PENDING",
            "RUNNING": "RUNNING",
            "COMPLETED": "COMPLETED",
            "FAILED": "FAILED",
            "CANCELLED": "CANCELLED",
            "TIMEOUT": "TIMEOUT",
            "NODE_FAIL": "NODE_FAIL",
            "BOOT_FAIL": "NODE_FAIL",
            "DEADLINE": "TIMEOUT",
            "OUT_OF_MEMORY": "FAILED",
        }
        
        return state_mapping.get(state_str.upper(), "UNKNOWN")