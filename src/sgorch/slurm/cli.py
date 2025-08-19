import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from ..logging_setup import get_logger
from .base import ISlurm, SubmitSpec, JobInfo
from .parse import parse_squeue_output, parse_scontrol_output


logger = get_logger(__name__)


class SlurmCliAdapter(ISlurm):
    """SLURM adapter using CLI commands (sbatch, squeue, scancel, scontrol)."""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
    
    def submit(self, spec: SubmitSpec) -> str:
        """Submit a job using sbatch command."""
        logger.info(f"Submitting job: {spec.name}")
        
        # Write script to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write(spec.script)
            script_path = f.name
        
        try:
            # Build sbatch command
            cmd = ['sbatch']
            
            # Add sbatch options
            cmd.extend(['--job-name', spec.name])
            cmd.extend(['--account', spec.account])
            cmd.extend(['--partition', spec.partition])
            cmd.extend(['--gres', spec.gres])
            cmd.extend(['--cpus-per-task', str(spec.cpus_per_task)])
            cmd.extend(['--mem', spec.mem])
            cmd.extend(['--time', spec.time_limit])
            cmd.extend(['--output', spec.stdout])
            cmd.extend(['--error', spec.stderr])
            
            if spec.reservation:
                cmd.extend(['--reservation', spec.reservation])
            if spec.qos:
                cmd.extend(['--qos', spec.qos])
            if spec.constraint:
                cmd.extend(['--constraint', spec.constraint])
            
            # Add environment variables
            for key, value in spec.env.items():
                cmd.extend(['--export', f'{key}={value}'])
            
            cmd.append(script_path)
            
            # Execute sbatch
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode != 0:
                error_msg = f"sbatch failed: {result.stderr}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Parse job ID from output (e.g., "Submitted batch job 12345")
            output = result.stdout.strip()
            job_id = output.split()[-1]
            
            logger.info(f"Job submitted successfully: {job_id}")
            return job_id
            
        finally:
            # Clean up temporary script file
            try:
                Path(script_path).unlink()
            except FileNotFoundError:
                pass
    
    def status(self, job_id: str) -> JobInfo:
        """Get job status using scontrol."""
        logger.debug(f"Getting status for job: {job_id}")
        
        cmd = ['scontrol', 'show', 'job', job_id]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode != 0:
                # Job might not exist anymore
                logger.warning(f"scontrol show job failed for {job_id}: {result.stderr}")
                return JobInfo(
                    job_id=job_id,
                    state='UNKNOWN',
                    node=None,
                    time_left_s=None
                )
            
            job_info = parse_scontrol_output(result.stdout)
            if job_info:
                return job_info
            else:
                logger.warning(f"Failed to parse scontrol output for job {job_id}")
                return JobInfo(
                    job_id=job_id,
                    state='UNKNOWN',
                    node=None,
                    time_left_s=None
                )
                
        except subprocess.TimeoutExpired:
            logger.error(f"scontrol command timed out for job {job_id}")
            raise
        except Exception as e:
            logger.error(f"Error getting job status: {e}")
            raise
    
    def cancel(self, job_id: str) -> None:
        """Cancel a job using scancel."""
        logger.info(f"Cancelling job: {job_id}")
        
        cmd = ['scancel', job_id]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode != 0:
                error_msg = f"scancel failed for {job_id}: {result.stderr}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            logger.info(f"Job cancelled successfully: {job_id}")
            
        except subprocess.TimeoutExpired:
            logger.error(f"scancel command timed out for job {job_id}")
            raise
        except Exception as e:
            logger.error(f"Error cancelling job: {e}")
            raise
    
    def list_jobs(self, name_prefix: str) -> List[JobInfo]:
        """List jobs with name prefix using squeue."""
        logger.debug(f"Listing jobs with prefix: {name_prefix}")
        
        # Use squeue to get jobs for current user with name filter
        cmd = [
            'squeue',
            '--user', 'ME',  # Current user only
            '--name', f'{name_prefix}*',
            '--format', '%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R',
            '--noheader'
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode != 0:
                logger.error(f"squeue failed: {result.stderr}")
                return []
            
            jobs = parse_squeue_output(result.stdout)
            logger.debug(f"Found {len(jobs)} jobs with prefix {name_prefix}")
            return jobs
            
        except subprocess.TimeoutExpired:
            logger.error(f"squeue command timed out")
            return []
        except Exception as e:
            logger.error(f"Error listing jobs: {e}")
            return []
    
    def _run_command(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """Run a command and return the result."""
        logger.debug(f"Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            return result
        except subprocess.TimeoutExpired as e:
            logger.error(f"Command timed out: {' '.join(cmd)}")
            raise
        except Exception as e:
            logger.error(f"Command failed: {e}")
            raise