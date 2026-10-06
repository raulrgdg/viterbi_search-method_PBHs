"""Splitting work across the jobs of a Condor cluster or a Slurm array."""


def split_for_job(targets, n_jobs, job_id):
    """Return the contiguous share of ``targets`` processed by job ``job_id``.

    The targets are split into ``n_jobs`` nearly equal contiguous chunks; the
    first ``len(targets) % n_jobs`` jobs receive one extra target. Jobs beyond
    the number of targets receive an empty list.
    """
    if n_jobs <= 0:
        raise ValueError("n_jobs must be > 0")
    if not 0 <= job_id < n_jobs:
        raise ValueError(f"job_id must be in [0, {n_jobs - 1}]")

    per_job, remainder = divmod(len(targets), n_jobs)
    start = job_id * per_job + min(job_id, remainder)
    end = start + per_job + (1 if job_id < remainder else 0)
    return list(targets[start:end])
