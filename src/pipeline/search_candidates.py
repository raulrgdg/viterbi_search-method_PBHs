import argparse
import csv
import os
from pathlib import Path

import numpy as np

from pipeline.common.paths import OUTPUTS_DIR, OUTPUTS_REPORTS_DIR, ensure_dir
from pipeline.search_metrics import search_candidates

DEFAULT_N_JOBS = 1
DEFAULT_JOB_ID = 0
MCHIRP_GRID = [1e-04, 5e-04, 8e-04, 1e-03, 2e-03, 3e-03, 4e-03, 5e-03, 6e-03, 7e-03, 8e-03, 9e-03, 1e-02, 2e-02, 3e-02, 4e-02, 5e-02, 6e-02, 7e-02, 8e-02, 9e-02, 1e-01]
DISTANCE_GRID = np.concatenate([np.array([0.001]), np.arange(0.005, 0.155, 0.005)])
TSFT_LIST = [2, 3, 4, 5, 6, 9, 12, 15, 21, 29, 39, 54, 74, 101, 132, 181, 248, 340]
PACKS_LIST = list(range(1, 109))
NOISE_MODE = True
SIGNAL_PACK = 3
REPORTS_DIR = ensure_dir(OUTPUTS_REPORTS_DIR)
TEMP_SHARDS_DIR = ensure_dir(OUTPUTS_DIR / "tmp" / "search_shards")
NOISE_CSV_NAME = "search_results_noise-01_04.csv"
SIGNAL_CSV_NAME = "search_results_signal.csv"
SEARCH_RESULT_FIELDNAMES = ["pack", "mchirp", "distance", "candidate", "nmse", "nsigma", "mass", "injected", "status"]


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone candidate search over precomputed products.")
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--job-id", type=int, default=DEFAULT_JOB_ID)
    return parser.parse_args()


def _output_csv_path(base_name, n_jobs, job_id):
    if n_jobs == 1:
        return REPORTS_DIR / base_name
    stem, suffix = base_name.rsplit(".", 1)
    return TEMP_SHARDS_DIR / f"{stem}.job-{job_id:03d}-of-{n_jobs:03d}.{suffix}"


def csv_shard_output_path(base_name, n_jobs=1, job_id=0):
    """Return the output path for a CSV shard or the final CSV when n_jobs == 1."""
    return _output_csv_path(base_name, n_jobs, job_id)


def search_results_csv_name(injected, pack=None):
    """Return the standard CSV filename for noise or injected search results."""
    if not injected:
        return NOISE_CSV_NAME
    if pack is None:
        return SIGNAL_CSV_NAME

    stem, suffix = SIGNAL_CSV_NAME.rsplit(".", 1)
    return f"{stem}_pack-{pack}.{suffix}"


def search_results_output_path(injected, n_jobs=1, job_id=0, pack=None):
    """Return the standard output path for one search-results writer."""
    return _output_csv_path(search_results_csv_name(injected, pack=pack), n_jobs, job_id)


def append_search_result_row(result, output_csv, *, write_header_if_missing=True):
    """Append one normalized search-result dict to a CSV file."""
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_csv.exists() and output_csv.stat().st_size > 0

    with output_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SEARCH_RESULT_FIELDNAMES)
        if write_header_if_missing and not file_exists:
            writer.writeheader()
        writer.writerow({field: result.get(field, "") for field in SEARCH_RESULT_FIELDNAMES})


def append_csv_rows(rows, output_csv, fieldnames):
    """Append normalized rows to a CSV, writing the header once if needed."""
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_csv.exists() and output_csv.stat().st_size > 0

    with output_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_search_results_rows(rows, output_csv):
    """Write a full collection of normalized search-result dicts to a CSV file."""
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SEARCH_RESULT_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SEARCH_RESULT_FIELDNAMES})


def write_csv_rows(rows, output_csv, fieldnames):
    """Write normalized rows to a CSV, replacing any existing file."""
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def mark_search_results_job_done(injected, n_jobs, job_id, pack=None):
    """Create the shard done-marker used for later merge."""
    mark_csv_shard_done(search_results_csv_name(injected, pack=pack), n_jobs, job_id)


def merge_search_results_if_ready(injected, total_jobs, pack=None):
    """Merge search-result shards when all jobs have completed."""
    merge_csv_shards_if_ready(search_results_csv_name(injected, pack=pack), SEARCH_RESULT_FIELDNAMES, total_jobs)


def _expected_shard_paths(base_name, total_jobs):
    return [_output_csv_path(base_name, total_jobs, job_id) for job_id in range(total_jobs)]


def _done_marker_path(base_name, n_jobs, job_id):
    shard_path = _output_csv_path(base_name, n_jobs, job_id)
    return shard_path.with_suffix(f"{shard_path.suffix}.done")


def csv_shard_done_marker_path(base_name, n_jobs, job_id):
    """Return the done-marker path for one CSV shard."""
    return _done_marker_path(base_name, n_jobs, job_id)


def _expected_done_markers(base_name, total_jobs):
    return [_done_marker_path(base_name, total_jobs, job_id) for job_id in range(total_jobs)]


def _merge_lock_path(base_name):
    return TEMP_SHARDS_DIR / f"{base_name}.merge.lock"


def _try_acquire_merge_lock(lock_path):
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None

    os.close(fd)
    return lock_path


def _merge_shards_if_ready(base_name, fieldnames, total_jobs):
    if total_jobs == 1:
        return

    shard_paths = _expected_shard_paths(base_name, total_jobs)
    done_markers = _expected_done_markers(base_name, total_jobs)
    missing_done = [path for path in done_markers if not path.exists()]
    if missing_done:
        print(
            f"Merge pendiente para {base_name}: faltan {len(missing_done)} job(s) por terminar.",
            flush=True,
        )
        return

    lock_path = _merge_lock_path(base_name)
    if _try_acquire_merge_lock(lock_path) is None:
        print(f"Merge ya en curso para {base_name}.", flush=True)
        return

    temp_output_csv = None
    try:
        output_csv = REPORTS_DIR / base_name
        temp_output_csv = output_csv.with_suffix(f"{output_csv.suffix}.tmp")
        with temp_output_csv.open("w", newline="", encoding="utf-8") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

            for shard_path in shard_paths:
                with shard_path.open(newline="", encoding="utf-8") as infile:
                    reader = csv.DictReader(infile)
                    for row in reader:
                        writer.writerow(row)

        temp_output_csv.replace(output_csv)

        for shard_path in shard_paths:
            shard_path.unlink(missing_ok=True)
        for done_marker in done_markers:
            done_marker.unlink(missing_ok=True)

        print(f"CSV unificado actualizado en: {output_csv}", flush=True)
    finally:
        if temp_output_csv is not None:
            temp_output_csv.unlink(missing_ok=True)
        lock_path.unlink(missing_ok=True)


def mark_csv_shard_done(base_name, n_jobs, job_id):
    """Create the done-marker for one generic CSV shard."""
    csv_shard_done_marker_path(base_name, n_jobs, job_id).write_text("", encoding="utf-8")


def merge_csv_shards_if_ready(base_name, fieldnames, total_jobs):
    """Merge generic CSV shards into the final reports directory output."""
    _merge_shards_if_ready(base_name, fieldnames, total_jobs)


def split_targets_for_job(targets, base_jobs, job_id):
    """
    Reparto sencillo y general usando exactamente `base_jobs` jobs.
    El resto se reparte entre los primeros jobs.
    """
    if base_jobs <= 0:
        raise ValueError("base_jobs debe ser > 0")

    total = len(targets)
    targets_per_job = total // base_jobs
    remainder = total % base_jobs
    total_jobs = base_jobs

    if job_id < 0 or job_id >= total_jobs:
        raise ValueError(f"job_id debe estar en [0, {total_jobs - 1}]")

    start = job_id * targets_per_job + min(job_id, remainder)
    end = start + targets_per_job + (1 if job_id < remainder else 0)
    return targets[start:end], targets_per_job, remainder, total_jobs


def _build_noise_targets():
    return [{"pack": pack} for pack in PACKS_LIST]


def _build_signal_targets():
    return [
        {"mchirp": mchirp, "distance": distance}
        for mchirp in MCHIRP_GRID
        for distance in DISTANCE_GRID
    ]


def _run_noise_search(assigned_targets):
    """Run the standalone search over the assigned noise packs."""
    rows = []
    skipped = 0

    for target in assigned_targets:
        pack = target["pack"]
        result = search_candidates(
            mchirp=None,
            distance=None,
            tsft_list=TSFT_LIST,
            pack=pack,
            noise=True,
        )
        if not result["candidate"]:
            print(f"[SKIP] pack={pack}, noise")
            skipped += 1

        rows.append(result)

    return rows, skipped


def _run_signal_search(assigned_targets):
    """Run the standalone search over the assigned injected-signal targets."""
    rows = []
    skipped = 0

    for target in assigned_targets:
        mchirp = target["mchirp"]
        distance = target["distance"]
        result = search_candidates(
            mchirp,
            distance,
            tsft_list=TSFT_LIST,
            pack=SIGNAL_PACK,
            noise=False,
        )
        if not result["candidate"]:
            print(f"[SKIP] mchirp={mchirp:.0e}, distance={distance}")
            skipped += 1

        rows.append(result)

    return rows, skipped


def main():
    """Run the legacy standalone search workflow and write the summary CSV."""
    args = parse_args()

    if NOISE_MODE:
        targets = _build_noise_targets()
        assigned_targets, targets_per_job, remainder, total_jobs = split_targets_for_job(
            targets, args.n_jobs, args.job_id
        )
        output_csv = _output_csv_path(NOISE_CSV_NAME, args.n_jobs, args.job_id)
        rows, skipped = _run_noise_search(assigned_targets)
    else:
        targets = _build_signal_targets()
        assigned_targets, targets_per_job, remainder, total_jobs = split_targets_for_job(
            targets, args.n_jobs, args.job_id
        )
        output_csv = _output_csv_path(SIGNAL_CSV_NAME, args.n_jobs, args.job_id)
        rows, skipped = _run_signal_search(assigned_targets)

    print(
        f"Reparto search: total_targets={len(targets)}, njobs={args.n_jobs}, "
        f"targets_por_job={targets_per_job}, resto={remainder}, total_jobs_reales={total_jobs}",
        flush=True,
    )

    print(
        f"Job actual: job_id={args.job_id}/{total_jobs - 1}, targets_asignados={len(assigned_targets)}",
        flush=True,
    )

    write_search_results_rows(rows, output_csv)

    if args.n_jobs > 1:
        _done_marker_path(
            NOISE_CSV_NAME if NOISE_MODE else SIGNAL_CSV_NAME,
            total_jobs,
            args.job_id,
        ).write_text("", encoding="utf-8")

    if not assigned_targets:
        print(
            f"Job sin targets asignados: job_id={args.job_id}, n_jobs={args.n_jobs}. Shard vacio generado.",
            flush=True,
        )

    if NOISE_MODE:
        _merge_shards_if_ready(NOISE_CSV_NAME, SEARCH_RESULT_FIELDNAMES, total_jobs)
    else:
        _merge_shards_if_ready(SIGNAL_CSV_NAME, SEARCH_RESULT_FIELDNAMES, total_jobs)

    print(f"Total procesados por este job: {len(rows)}")
    print(f"Total skipped candidates: {skipped}")
    print(f"CSV guardado en: {output_csv}")


if __name__ == "__main__":
    main()
