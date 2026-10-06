"""Search-result CSV files.

Each job appends one row per analysed chunk to its own shard
(``results/tmp/shards/<name>.job-XXX-of-YYY.csv``) and leaves a ``.done``
marker when it finishes. The last job to finish merges all shards into
``results/search/<name>.csv``.
"""

import csv
import logging
import math
import os
from pathlib import Path

from pbh_viterbi.paths import SEARCH_RESULTS_DIR, SHARDS_DIR, ensure_dir

log = logging.getLogger(__name__)

FIELDNAMES = [
    "pack",       # O3b pack id
    "mchirp",     # injected chirp mass (Msun); empty for noise
    "distance",   # injected luminosity distance (Mpc); empty for noise
    "ra", "dec", "pol",  # injected sky location and polarisation (rad); empty for noise
    "candidate",  # passes the provisional linear threshold
    "nmse",       # NMSE of the isolated candidate (Eq. 22-23)
    "nsigma",     # nsigma of the selected Tsft (Eq. 17)
    "mass",       # chirp mass of the best fit, Eq. (25) (Msun)
    "tsft",       # Tsft selected by nsigma (s)
    "injected",   # 1 for injected searches, 0 for noise
    "status",     # outcome of the candidate isolation
]


def _fmt(value):
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def result_row(pack, search_result, injected, mchirp=None, distance=None, sky=None):
    """Assemble one CSV row from the output of search.candidates.search_candidate."""
    sky = sky or {}
    row = {
        "pack": int(pack),
        "mchirp": float(mchirp) if mchirp is not None else None,
        "distance": float(distance) if distance is not None else None,
        "ra": sky.get("ra"),
        "dec": sky.get("dec"),
        "pol": sky.get("pol"),
        "candidate": bool(search_result["candidate"]),
        "nmse": search_result["nmse"],
        "nsigma": search_result["nsigma"],
        "mass": search_result["mass"],
        "tsft": search_result["tsft"],
        "injected": 1 if injected else 0,
        "status": search_result["status"],
    }
    return {key: _fmt(value) for key, value in row.items()}


def csv_name(injected, pack=None):
    if not injected:
        return "search_results_noise.csv"
    return "search_results_injected.csv" if pack is None else f"search_results_injected_pack-{pack}.csv"


def shard_path(name, n_jobs, job_id):
    """Final CSV for single-job runs, otherwise the shard of job ``job_id``."""
    if n_jobs == 1:
        return SEARCH_RESULTS_DIR / name
    stem, suffix = name.rsplit(".", 1)
    return SHARDS_DIR / f"{stem}.job-{job_id:03d}-of-{n_jobs:03d}.{suffix}"


def append_rows(rows, path, fieldnames=FIELDNAMES):
    """Append rows to a CSV, writing the header if the file is new or empty."""
    path = Path(path)
    ensure_dir(path.parent)
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_rows(rows, path, fieldnames=FIELDNAMES):
    """Write rows to a CSV, replacing any existing file."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def finish_shard(name, n_jobs, job_id, fieldnames=FIELDNAMES):
    """Mark job ``job_id`` as done and merge all shards if every job has finished."""
    if n_jobs == 1:
        return
    shard = shard_path(name, n_jobs, job_id)
    if not shard.exists():
        write_rows([], shard, fieldnames)
    Path(f"{shard}.done").write_text("", encoding="utf-8")
    merge_shards_if_ready(name, n_jobs, fieldnames)


def merge_shards_if_ready(name, n_jobs, fieldnames=FIELDNAMES):
    shards = [shard_path(name, n_jobs, job_id) for job_id in range(n_jobs)]
    markers = [Path(f"{shard}.done") for shard in shards]
    missing = sum(not marker.exists() for marker in markers)
    if missing:
        log.info("Merge of %s pending: %d job(s) still running.", name, missing)
        return

    lock = SHARDS_DIR / f"{name}.merge.lock"
    try:
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        log.info("Merge of %s already in progress.", name)
        return

    output = ensure_dir(SEARCH_RESULTS_DIR) / name
    tmp_output = output.with_suffix(output.suffix + ".tmp")
    try:
        with tmp_output.open("w", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=fieldnames)
            writer.writeheader()
            for shard in shards:
                with shard.open(newline="", encoding="utf-8") as handle:
                    writer.writerows(csv.DictReader(handle))
        tmp_output.replace(output)
        for path in shards + markers:
            path.unlink(missing_ok=True)
        log.info("Merged search results written to %s", output)
    finally:
        tmp_output.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)
