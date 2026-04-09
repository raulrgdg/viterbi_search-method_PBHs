import argparse
import os
import tempfile
import time

import numpy as np
import soapcw as soap

from pipeline.common.paths import BIN_DIR
from pipeline.download.download_o3 import DEFAULT_OUTPUT_ROOT, O3_WINDOWS
from pipeline.search_candidates import (
    append_csv_rows,
    append_search_result_row,
    csv_shard_output_path,
    mark_csv_shard_done,
    mark_search_results_job_done,
    merge_csv_shards_if_ready,
    merge_search_results_if_ready,
    search_results_output_path,
    split_targets_for_job,
    write_csv_rows,
    write_search_results_rows,
)
from pipeline.search_metrics import (
    POWER_NOISE_CSV_NAME,
    compute_noise_total_power_rows_in_memory,
    search_candidates_in_memory,
)
from pipeline.sft.make_sfts import run_make_sfts_script
from pipeline.sft.tracking import (
    VITERBI_TRANSITION_LOG_PROBS,
    build_remap_geometry,
    load_sfts,
    preprocess_data,
    remap_CShuster_to_fm83,
)
from pipeline.utils.framecache import generate_framecache_raw_strain

BASH_SCRIPT_PATH = BIN_DIR / "make_SFT-final-v2.sh"

SEARCH = True

PACKS = list(range(1, 109))
TSFT_VALUES = [2, 3, 4, 5, 7, 10, 13, 18, 25, 35, 47, 63, 88]
REQUEST_CPUS_PER_JOB = 8
DEFAULT_N_JOBS = 108
DEFAULT_JOB_ID = 0

MAKE_SFT_THREADS = 256

FRAME_LENGTH = 4096
NUM_FRAMES = 8
FMIN = 61.1
FMAX = 127.2
FBAND = FMAX - FMIN
CHANNEL_NAME = "H1:GWOSC-4KHZ_R1_STRAIN"
WINDOWTYPE = "rectangular"
IFO = "H1"
FS_TARGET = 512
SFT_VERBOSE = True
PIPELINE_VERBOSE = True


def get_local_o3_pack_dir(pack):
    raw_data_dir = os.path.join(DEFAULT_OUTPUT_ROOT, f"O3b-pack{pack}")
    if not os.path.isdir(raw_data_dir):
        raise FileNotFoundError(
            f"No existe el pack O3 descargado en disco: {raw_data_dir}. "
            "Primero ejecuta pipeline.download.download_o3."
        )
    return raw_data_dir


def update_pack_progress(completed, total, status="ok", detail=""):
    message = f"Packs procesados {completed}/{total}"
    if status != "ok":
        message = f"{message} | estado={status}"
    if detail:
        message = f"{message} | {detail}"
    end = "\n" if status in {"error", "done"} else "\r"
    print(message, end=end, flush=True)


def log_verbose(message):
    if PIPELINE_VERBOSE:
        print(message, flush=True)


def normalize_injected_field(result):
    normalized = dict(result)
    normalized["injected"] = 1 if bool(result.get("injected")) else 0
    return normalized


def build_tsft_configs(tsft_values):
    total_duration = NUM_FRAMES * FRAME_LENGTH
    return [
        {
            "tsft": tsft,
            "nbins": round(FBAND / (1 / tsft)),
            "nsft": int(total_duration / tsft),
            "remap_geometry": build_remap_geometry(
                tsft,
                FMIN,
                round(FBAND / (1 / tsft)),
            ),
        }
        for tsft in tsft_values
    ]


def power_output_path(n_jobs=1, job_id=0):
    return csv_shard_output_path(POWER_NOISE_CSV_NAME, n_jobs=n_jobs, job_id=job_id)


def append_power_rows(rows, output_csv):
    append_csv_rows(rows, output_csv, ["pack", "tsft", "total_power"])


def write_power_rows(rows, output_csv):
    write_csv_rows(rows, output_csv, ["pack", "tsft", "total_power"])


def mark_power_job_done(n_jobs, job_id):
    mark_csv_shard_done(POWER_NOISE_CSV_NAME, n_jobs, job_id)


def merge_power_results_if_ready(total_jobs):
    merge_csv_shards_if_ready(POWER_NOISE_CSV_NAME, ["pack", "tsft", "total_power"], total_jobs)


def process_single_tsft(tsft_config, t_start, t_end, framecache_path_raw_strain, num_threads_per_worker):
    tsft = tsft_config["tsft"]

    with tempfile.TemporaryDirectory(prefix=f"sft-tsft{tsft}-") as sft_output_path:
        sft_start = time.perf_counter()
        run_make_sfts_script(
            bash_script_path=BASH_SCRIPT_PATH,
            t_start=t_start,
            t_end=t_end,
            sft_output_path=sft_output_path,
            framecache_path=framecache_path_raw_strain,
            num_threads=num_threads_per_worker,
            Tseg=tsft,
            fmin=FMIN,
            Band=FBAND,
            windowtype=WINDOWTYPE,
            channel=CHANNEL_NAME,
            verbose=SFT_VERBOSE,
        )
        sft_elapsed = time.perf_counter() - sft_start
        print(f"[TIMING] MakeSFTs tsft={tsft}: {sft_elapsed:.2f} s", flush=True)

        data = load_sfts(
            sft_output_path,
            t_start,
            tsft,
            tsft_config["nbins"],
            tsft_config["nsft"],
        )
        remap_geometry = tsft_config["remap_geometry"]
        cshuster, freqs = preprocess_data(
            data,
            tsft,
            FMIN,
            FMAX,
            freqs=remap_geometry["freqs"],
        )
        x_remap, cshuster_remap = remap_CShuster_to_fm83(
            cshuster,
            freqs,
            x_new=remap_geometry["x_new"],
            fill_value=np.nan,
            x_inc=remap_geometry["x_inc"],
        )
        one_tracks_ng = soap.single_detector(VITERBI_TRANSITION_LOG_PROBS, cshuster_remap, lookup_table=None)
        track_index = np.asarray(one_tracks_ng.vit_track, dtype=int)
        track_freq = np.asarray(x_remap[track_index], dtype=float)
    return {
        "tsft": tsft,
        "track_index": track_index,
        "track_freq": track_freq,
        "power": np.asarray(cshuster_remap, dtype=float),
    }


def process_single_pack(pack, tsft_values, num_threads_per_worker):
    t_start, t_end = O3_WINDOWS[f"O3b-pack{pack}"]
    tsft_configs = build_tsft_configs(tsft_values)
    raw_data_dir = get_local_o3_pack_dir(pack)
    log_verbose(f"Usando datos O3 ya descargados para pack {pack}: {raw_data_dir}")

    framecache_path_raw_strain = generate_framecache_raw_strain(
        input_dir=raw_data_dir,
        det=IFO[0],
        num_frames=NUM_FRAMES,
        frame_length=FRAME_LENGTH,
        t_start=t_start,
        verbose=PIPELINE_VERBOSE,
    )

    tsft_results = []
    for tsft_config in tsft_configs:
        tsft = tsft_config["tsft"]
        log_verbose(f"[pack {pack}] Iniciando Tsft={tsft}")
        tsft_results.append(
            process_single_tsft(
                tsft_config=tsft_config,
                t_start=t_start,
                t_end=t_end,
                framecache_path_raw_strain=framecache_path_raw_strain,
                num_threads_per_worker=num_threads_per_worker,
            )
        )
        log_verbose(f"[pack {pack}] Tsft completado: {tsft}")
    if SEARCH:
        return search_candidates_in_memory(mchirp=None, distance=None, pack=pack, noise=True, tsft_results=tsft_results)
    return compute_noise_total_power_rows_in_memory(pack=pack, tsft_results=tsft_results, n_windows=1)


def split_packs_for_job(packs, base_jobs, job_id):
    return split_targets_for_job(packs, base_jobs, job_id)


def parse_args():
    parser = argparse.ArgumentParser(description="Noise data generation repartido en jobs Condor.")
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--job-id", type=int, default=DEFAULT_JOB_ID)
    return parser.parse_args()


def initialize_output_csv(search_enabled, n_jobs, job_id):
    if search_enabled:
        return search_results_output_path(injected=False, n_jobs=n_jobs, job_id=job_id)
    return power_output_path(n_jobs=n_jobs, job_id=job_id)


def write_empty_output(search_enabled, output_csv):
    if search_enabled:
        write_search_results_rows([], output_csv)
    else:
        write_power_rows([], output_csv)


def append_result_rows(search_enabled, result, output_csv):
    if search_enabled:
        append_search_result_row(normalize_injected_field(result), output_csv)
    else:
        append_power_rows(result, output_csv)


def finalize_noise_shard(search_enabled, total_jobs, job_id):
    if search_enabled:
        mark_search_results_job_done(injected=False, n_jobs=total_jobs, job_id=job_id)
        merge_search_results_if_ready(injected=False, total_jobs=total_jobs)
    else:
        mark_power_job_done(n_jobs=total_jobs, job_id=job_id)
        merge_power_results_if_ready(total_jobs=total_jobs)


def print_job_distribution(total_packs, n_jobs, packs_per_job, remainder, total_jobs):
    print(
        f"Reparto jobs: npacks={total_packs}, njobs={n_jobs}, "
        f"packs_por_job={packs_per_job}, resto={remainder}, total_jobs_reales={total_jobs}",
        flush=True,
    )


def print_current_job(job_id, total_jobs, packs, num_threads_per_worker):
    print(
        f"Job actual: job_id={job_id}/{total_jobs - 1}, packs_asignados={len(packs)}, "
        f"primer_pack={packs[0]}, ultimo_pack={packs[-1]}, request_cpus={REQUEST_CPUS_PER_JOB}, "
        f"make_sft_threads={num_threads_per_worker}",
        flush=True,
    )


def main():
    args = parse_args()
    packs = PACKS
    tsft_values = TSFT_VALUES
    num_threads_per_worker = MAKE_SFT_THREADS

    packs, packs_per_job, remainder, total_jobs = split_packs_for_job(packs, args.n_jobs, args.job_id)
    output_csv = initialize_output_csv(SEARCH, args.n_jobs, args.job_id)
    if not packs:
        write_empty_output(SEARCH, output_csv)

        print(
            f"Job sin packs asignados: job_id={args.job_id}, n_jobs={args.n_jobs}. "
            "Shard vacio generado."
        )
        if args.n_jobs > 1:
            finalize_noise_shard(SEARCH, total_jobs, args.job_id)
        return

    print_job_distribution(len(PACKS), args.n_jobs, packs_per_job, remainder, total_jobs)
    print_current_job(args.job_id, total_jobs, packs, num_threads_per_worker)

    completed_packs = 0
    total_assigned_packs = len(packs)
    for idx, pack in enumerate(packs, start=1):
        update_pack_progress(
            completed_packs,
            total_assigned_packs,
            status="running",
            detail=f"procesando pack {idx}/{total_assigned_packs} (pack={pack})",
        )
        try:
            result = process_single_pack(pack, tsft_values, num_threads_per_worker)
            append_result_rows(SEARCH, result, output_csv)
        except Exception as exc:
            update_pack_progress(
                completed_packs,
                total_assigned_packs,
                status="error",
                detail=f"error en pack {idx}/{total_assigned_packs} (pack={pack}): {exc}",
            )
            raise

        completed_packs += 1
        final_status = "done" if completed_packs == total_assigned_packs else "ok"
        update_pack_progress(completed_packs, total_assigned_packs, status=final_status)
    if args.n_jobs > 1:
        finalize_noise_shard(SEARCH, total_jobs, args.job_id)


if __name__ == "__main__":
    main()
