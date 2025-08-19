from sgorch.slurm.parse import parse_squeue_output, parse_sacct_output, parse_scontrol_output, _parse_time_remaining


def test_parse_squeue_output_header_and_headerless():
    header = "JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)\n"
    row = "12345 p         sgl-d1-0 u     R        01:23      1 cn001\n"
    out = header + row
    jobs = parse_squeue_output(out)
    assert jobs and jobs[0].job_id == "12345" and jobs[0].state == "RUNNING" and jobs[0].node == "cn001"
    jobs2 = parse_squeue_output(row)
    assert jobs2 and jobs2[0].job_id == "12345"


def test_parse_sacct_output_various_rows():
    out = "JobID|State|NodeList|TimeLeft\n123|COMPLETED|cn001|00:00\n124|FAILED||\n"
    jobs = parse_sacct_output(out)
    assert len(jobs) == 2
    assert jobs[0].state == "COMPLETED" and jobs[0].node == "cn001"
    assert jobs[1].state == "FAILED" and jobs[1].node is None


def test_parse_scontrol_output_extracts_fields():
    out = "JobId=777 JobState=RUNNING NodeList=cn[001-003] TimeLeft=1-02:03:04"
    info = parse_scontrol_output(out)
    assert info.job_id == "777" and info.state == "RUNNING" and info.node == "cn001" and info.time_left_s is not None


def test_time_formats_to_seconds():
    assert _parse_time_remaining("05:06") == 5*60 + 6
    assert _parse_time_remaining("01:02:03") == 1*3600 + 2*60 + 3
    assert _parse_time_remaining("2-01:02:03") == 2*86400 + 1*3600 + 2*60 + 3

