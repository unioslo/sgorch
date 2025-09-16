import re
from typing import List, Optional, Dict, Any

from .base import JobInfo, JobState


def parse_squeue_output(output: str) -> List[JobInfo]:
    """Parse squeue output into JobInfo objects.

    Supports both header and headerless output, and works with the
    custom --format used by SlurmCliAdapter:
      '%.18i %.9P %j %.8u %.2t %.10M %.6D %R'
    """
    jobs: List[JobInfo] = []
    lines = [ln for ln in output.strip().split('\n') if ln.strip()]

    if not lines:
        return jobs

    # Detect whether the first line is a header: if it starts with non-digits
    # (e.g., "JOBID"), treat it as a header and skip it; otherwise, parse all lines.
    first_parts = lines[0].split()
    first_token = first_parts[0] if first_parts else ''
    has_header = not first_token.isdigit()
    start_idx = 1 if has_header else 0

    for line in lines[start_idx:]:
        parts = line.split()
        if len(parts) < 6:
            continue

        # Expected columns with our --format:
        # 0: JOBID, 1: PARTITION, 2: NAME, 3: USER, 4: ST, 5: TIME, 6: NODES, 7: NODELIST(REASON)
        job_id = parts[0]
        state_str = parts[4]
        state = _parse_job_state(state_str)

        node = None
        if len(parts) >= 8:
            nodelist = parts[7]
            if not nodelist.startswith('('):  # Not a reason in parentheses
                node = _extract_first_node(nodelist)

        time_left_s = _parse_time_remaining(parts[5])

        jobs.append(JobInfo(
            job_id=job_id,
            state=state,
            node=node,
            time_left_s=time_left_s
        ))

    return jobs


def parse_sacct_output(output: str) -> List[JobInfo]:
    """Parse sacct output into JobInfo objects."""
    jobs = []
    lines = output.strip().split('\n')
    
    if len(lines) < 2:  # No header or data
        return jobs
    
    # Skip header line
    for line in lines[1:]:
        if not line.strip():
            continue
        
        # Sacct format can be customized, assume: JobID|State|NodeList|TimeLeft
        parts = line.split('|')
        
        if len(parts) < 2:
            continue
        
        job_id = parts[0].split('.')[0]  # Remove step information
        
        state_str = parts[1]
        state = _parse_job_state(state_str)
        
        node = None
        if len(parts) > 2 and parts[2].strip():
            node = _extract_first_node(parts[2])
        
        time_left_s = None
        if len(parts) > 3 and parts[3].strip():
            time_left_s = _parse_time_remaining(parts[3])
        
        jobs.append(JobInfo(
            job_id=job_id,
            state=state,
            node=node,
            time_left_s=time_left_s
        ))
    
    return jobs


def parse_scontrol_output(output: str) -> Optional[JobInfo]:
    """Parse scontrol show job output for a single job."""
    # Parse key=value pairs from scontrol output
    data = {}
    
    # scontrol output is key=value pairs separated by spaces/newlines
    # Handle multi-line values and embedded spaces
    current_key = None
    current_value = []
    
    for line in output.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # Look for key=value patterns
        key_value_pairs = re.findall(r'(\w+)=([^\s]*(?:\s[^\w=]*)*)', line)
        
        for key, value in key_value_pairs:
            data[key] = value.strip()
    
    if 'JobId' not in data:
        return None
    
    job_id = data['JobId']
    state_str = data.get('JobState', 'UNKNOWN')
    state = _parse_job_state(state_str)
    
    # Extract node from NodeList
    node = None
    if 'NodeList' in data and data['NodeList'] not in ('(null)', ''):
        node = _extract_first_node(data['NodeList'])
    
    # Parse time left from TimeLeft
    time_left_s = None
    if 'TimeLeft' in data:
        time_left_s = _parse_time_remaining(data['TimeLeft'])
    
    return JobInfo(
        job_id=job_id,
        state=state,
        node=node,
        time_left_s=time_left_s
    )


def _parse_job_state(state_str: str) -> JobState:
    """Parse SLURM job state string to JobState enum."""
    state_mapping = {
        'PD': 'PENDING',
        'R': 'RUNNING',
        'CD': 'COMPLETED',
        'F': 'FAILED',
        'CA': 'CANCELLED',
        'TO': 'TIMEOUT',
        'NF': 'NODE_FAIL',
        'PENDING': 'PENDING',
        'RUNNING': 'RUNNING',
        'COMPLETED': 'COMPLETED',
        'FAILED': 'FAILED',
        'CANCELLED': 'CANCELLED',
        'TIMEOUT': 'TIMEOUT',
        'NODE_FAIL': 'NODE_FAIL',
    }
    
    return state_mapping.get(state_str.upper(), 'UNKNOWN')


def _extract_first_node(nodelist: str) -> Optional[str]:
    """Extract the first node from a SLURM nodelist."""
    if not nodelist or nodelist in ('(null)', ''):
        return None
    
    # Handle node lists like: cn[001-003] -> cn001
    if '[' in nodelist:
        base = nodelist.split('[')[0]
        range_part = nodelist.split('[')[1].split(']')[0]
        
        if '-' in range_part:
            first_num = range_part.split('-')[0]
            return base + first_num
        else:
            return base + range_part.split(',')[0]
    
    # Handle comma-separated lists: node1,node2 -> node1
    return nodelist.split(',')[0]


def _parse_time_remaining(time_str: str) -> Optional[int]:
    """Parse time remaining string to seconds."""
    if not time_str or time_str in ('UNLIMITED', 'INFINITE', ''):
        return None
    
    # Handle different time formats:
    # MM:SS
    # HH:MM:SS
    # DD-HH:MM:SS
    # UNLIMITED
    
    if time_str == 'UNLIMITED' or time_str == 'INFINITE':
        return None
    
    try:
        # Split by '-' for days
        if '-' in time_str:
            days_part, time_part = time_str.split('-', 1)
            days = int(days_part)
        else:
            days = 0
            time_part = time_str
        
        # Split time part
        time_parts = time_part.split(':')
        
        if len(time_parts) == 2:  # MM:SS
            minutes, seconds = map(int, time_parts)
            hours = 0
        elif len(time_parts) == 3:  # HH:MM:SS
            hours, minutes, seconds = map(int, time_parts)
        else:
            return None
        
        total_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds
        return total_seconds
        
    except (ValueError, IndexError):
        return None
