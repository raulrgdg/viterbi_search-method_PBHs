import argparse
import csv
from pathlib import Path
import time

import numpy as np

from pipeline_paths import OUTPUTS_DIR, OUTPUTS_REPORTS_DIR, ensure_dir
from power_metric_prueba import search_candidates

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
NOISE_CSV_NAME = "search_results_noise_final-big-search-flag_perc-90-new-recorte-total-window-v4.csv"
SIGNAL_CSV_NAME = "search_results_signal_final-big-search-flag_perc-90-new-recorte-total-window-v4.csv"
MERGE_WAIT_SECONDS = 120
MERGE_POLL_SECONDS = 5


def parse_args():
    parser = argparse.ArgumentParser(description="Candidate search repartido en jobs Condor.")
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--job-id", type=int, default=DEFAULT_JOB_ID)
    return parser.parse_args()


def _resolve_result(result):
    """Normalize the variable-length search result tuple."""
    if result[0] is None:
        if len(result) == 2:
            res, opt_nsigma = result
            return res, None, opt_nsigma, None
        res, nmse_eval, opt_nsigma, extended_optimal_block = result
        return res, nmse_eval, opt_nsigma, extended_optimal_block

    res, nmse_eval, opt_nsigma, extended_optimal_block = result
    return res, nmse_eval, opt_nsigma, extended_optimal_block


def _output_csv_path(base_name, n_jobs, job_id):
    if n_jobs == 1:
        return REPORTS_DIR / base_name
    stem, suffix = base_name.rsplit(".", 1)
    return TEMP_SHARDS_DIR / f"{stem}.job-{job_id:03d}-of-{n_jobs:03d}.{suffix}"


def _expected_shard_paths(base_name, total_jobs):
    return [_output_csv_path(base_name, total_jobs, job_id) for job_id in range(total_jobs)]


def _merge_shards_if_ready(base_name, fieldnames, total_jobs, job_id):
    if total_jobs == 1:
        return

    # Solo un job intenta cerrar el merge para evitar carreras y borrados cruzados.
    if job_id != total_jobs - 1:
        print(
            f"Merge delegado al ultimo job: actual={job_id}, merger={total_jobs - 1}.",
            flush=True,
        )
        return

    output_csv = REPORTS_DIR / base_name
    shard_paths = _expected_shard_paths(base_name, total_jobs)
    deadline = time.time() + MERGE_WAIT_SECONDS
    while True:
        missing = [path for path in shard_paths if not path.exists()]
        if not missing:
            break
        if time.time() >= deadline:
            print(
                f"Merge pendiente para {base_name}: faltan {len(missing)} shard(s) tras esperar {MERGE_WAIT_SECONDS}s.",
                flush=True,
            )
            return
        print(
            f"Esperando shards para {base_name}: faltan {len(missing)} shard(s).",
            flush=True,
        )
        time.sleep(MERGE_POLL_SECONDS)

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

    print(f"CSV unificado actualizado en: {output_csv}", flush=True)


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
    """Run the search over the assigned noise packs and collect CSV rows."""
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
        res, nmse_eval, opt_nsigma, _extended_optimal_block = _resolve_result(result)
        candidate = bool(res)
        if not candidate:
            print(f"[SKIP] pack={pack}, noise")
            skipped += 1

        rows.append(
            {
                "candidate": candidate,
                "final_nmse": nmse_eval if nmse_eval is not None else "",
                "opt_nsigma": opt_nsigma if opt_nsigma is not None else "",
            }
        )

    return rows, skipped


def _run_signal_search(assigned_targets):
    """Run the search over the assigned injected-signal targets and collect CSV rows."""
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
        res, nmse_eval, opt_nsigma, extended_optimal_block = _resolve_result(result)
        candidate = bool(res)
        if not candidate:
            print(f"[SKIP] mchirp={mchirp:.0e}, distance={distance}")
            skipped += 1

        final_mass = ""
        final_slope = ""
        if extended_optimal_block is not None:
            final_mass_value = extended_optimal_block.get("final_mass")
            final_slope_value = extended_optimal_block.get("final_slope")
            if final_mass_value is not None:
                final_mass = float(final_mass_value)
            if final_slope_value is not None:
                final_slope = float(final_slope_value)

        rows.append(
            {
                "real_mchirp": mchirp,
                "real_distance": distance,
                "candidate": candidate,
                "final_nmse": nmse_eval if nmse_eval is not None else "",
                "opt_nsigma": opt_nsigma if opt_nsigma is not None else "",
                "final_mass": final_mass,
                "final_slope": final_slope,
            }
        )

    return rows, skipped


def main():
    """Run the candidate search workflow and write the summary CSV."""
    args = parse_args()

    if NOISE_MODE:
        targets = _build_noise_targets()
        assigned_targets, targets_per_job, remainder, total_jobs = split_targets_for_job(
            targets, args.n_jobs, args.job_id
        )
        output_csv = _output_csv_path(NOISE_CSV_NAME, args.n_jobs, args.job_id)
        fieldnames = ["candidate", "final_nmse", "opt_nsigma"]
        rows, skipped = _run_noise_search(assigned_targets)
    else:
        targets = _build_signal_targets()
        assigned_targets, targets_per_job, remainder, total_jobs = split_targets_for_job(
            targets, args.n_jobs, args.job_id
        )
        output_csv = _output_csv_path(SIGNAL_CSV_NAME, args.n_jobs, args.job_id)
        fieldnames = [
            "real_mchirp",
            "real_distance",
            "candidate",
            "final_nmse",
            "opt_nsigma",
            "final_mass",
            "final_slope",
        ]
        rows, skipped = _run_signal_search(assigned_targets)

    print(
        f"Reparto search: total_targets={len(targets)}, njobs={args.n_jobs}, "
        f"targets_por_job={targets_per_job}, resto={remainder}, total_jobs_reales={total_jobs}",
        flush=True,
    )

    if not assigned_targets:
        print(
            f"Job sin targets asignados: job_id={args.job_id}, n_jobs={args.n_jobs}. No hay trabajo.",
            flush=True,
        )
        return

    print(
        f"Job actual: job_id={args.job_id}/{total_jobs - 1}, targets_asignados={len(assigned_targets)}",
        flush=True,
    )

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if NOISE_MODE:
        _merge_shards_if_ready(NOISE_CSV_NAME, fieldnames, total_jobs, args.job_id)
    else:
        _merge_shards_if_ready(SIGNAL_CSV_NAME, fieldnames, total_jobs, args.job_id)

    print(f"Total procesados por este job: {len(rows)}")
    print(f"Total skipped candidates: {skipped}")
    print(f"CSV guardado en: {output_csv}")


if __name__ == "__main__":
    main()
