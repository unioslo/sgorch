import subprocess
import tempfile
from pathlib import Path
from typing import List

from ..logging_setup import get_logger
from .base import ISlurm, SubmitSpec, JobInfo, SlurmUnavailableError
from .errors import raise_if_unavailable, SlurmOperation
from .parse import parse_squeue_output, parse_scontrol_output


logger = get_logger(__name__)


class SlurmCliAdapter(ISlurm):
    """SLURM adapter using CLI commands (sbatch, squeue, scancel, scontrol)."""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def _check_slurm_unavailable(self, message: str, *, operation: SlurmOperation) -> None:
        """Raise SlurmUnavailableError when message matches outage patterns."""
        raise_if_unavailable(message, operation=operation)

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
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )
            except subprocess.TimeoutExpired:
                logger.error("sbatch command timed out")
                raise SlurmUnavailableError("sbatch command timed out")
            except Exception as exc:
                logger.error(f"sbatch command failed: {exc}")
                message = str(exc)
                self._check_slurm_unavailable(message, operation=SlurmOperation.SUBMIT)
                raise

            if result.returncode != 0:
                raw_error = (result.stderr or result.stdout or "").strip() or "sbatch failed"
                logger.error(f"sbatch failed: {raw_error}")
                self._check_slurm_unavailable(raw_error, operation=SlurmOperation.SUBMIT)
                raise RuntimeError(f"sbatch failed: {raw_error}")
            
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
        """List jobs with names starting with the given prefix using squeue.

        We filter by name on the client side using the formatted output to avoid
        relying on cluster-specific squeue matching semantics.
        """
        logger.debug(f"Listing jobs with prefix: {name_prefix}")

        # Use squeue to get jobs for current user
        import os
        current_user = os.getenv('USER', os.getenv('USERNAME', 'ME'))

        cmd = [
            'squeue',
            '--user', current_user,
            '--format', '%.18i %.9P %j %.8u %.2t %.10M %.6D %R',
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
                message = (result.stderr or result.stdout or "").strip() or "squeue failed"
                logger.error(f"squeue failed: {message}")
                self._check_slurm_unavailable(message, operation=SlurmOperation.LIST)
                raise RuntimeError(message)

            # Client-side filter by job name (3rd column, index 2)
            filtered_lines: list[str] = []
            for line in (ln for ln in result.stdout.splitlines() if ln.strip()):
                parts = line.split()
                if len(parts) >= 3:
                    job_name = parts[2]
                    if job_name.startswith(name_prefix):
                        filtered_lines.append(line)

            filtered_output = "\n".join(filtered_lines)
            jobs = parse_squeue_output(filtered_output)
            logger.debug(f"Found {len(jobs)} jobs matching prefix {name_prefix}")
            return jobs

        except subprocess.TimeoutExpired:
            logger.error("squeue command timed out")
            raise SlurmUnavailableError("squeue command timed out")
        except Exception as e:
            message = str(e)
            logger.error(f"Error listing jobs: {message}")
            self._check_slurm_unavailable(message, operation=SlurmOperation.LIST)
            raise RuntimeError(message)
    
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
