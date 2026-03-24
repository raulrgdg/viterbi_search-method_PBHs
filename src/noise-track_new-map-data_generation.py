import os
import tempfile
import argparse
import time
import numpy as np
from make_framecache_final import generate_framecache_raw_strain
from make_SFT_function import run_make_sfts_script
from download_O3_data import DEFAULT_OUTPUT_ROOT, O3_WINDOWS
from algortihm_final import VITERBI_TRANSITION_LOG_PROBS, build_remap_geometry, load_sfts, preprocess_data, remap_CShuster_to_fm83
from pipeline_paths import BIN_DIR
from power_metric_prueba import search_candidates_in_memory
from search_candidate import append_search_result_row, mark_search_results_job_done, merge_search_results_if_ready, search_results_output_path
import soapcw as soap

BASH_SCRIPT_PATH = BIN_DIR / "make_SFT-final-v2.sh"

# Case: N=noise

DEFAULT_PACKS = list(range(1, 109))
DEFAULT_TSFT_VALUES = [2, 3, 4, 5, 6, 9, 12, 15, 21, 29, 39, 54, 74, 101, 132, 181, 248, 340] #quite 1 !
REQUEST_CPUS_PER_JOB = 16
DEFAULT_N_JOBS = 5
DEFAULT_JOB_ID = 0

# Depend on the number of CPUs requested. See strong_scale_test
MAKE_SFT_THREADS = 256

# Data parameters
FRAME_LENGTH = 4096
NUM_FRAMES = 8
FMIN = 61.1
FMAX = 126.8
FBAND = FMAX - FMIN
CHANNEL_NAME = "H1:GWOSC-4KHZ_R1_STRAIN"
WINDOWTYPE = "rectangular"
IFO = "H1"
FS_TARGET = 512
SFT_VERBOSE = False
PIPELINE_VERBOSE = False


def get_local_o3_pack_dir(pack):
    """Return the local directory containing a pre-downloaded O3 pack."""
    raw_data_dir = os.path.join(DEFAULT_OUTPUT_ROOT, f"O3b-pack{pack}-{FS_TARGET}HZ")
    if not os.path.isdir(raw_data_dir):
        raise FileNotFoundError(
            f"No existe el pack O3 descargado en disco: {raw_data_dir}. "
            "Primero ejecuta src/download_O3_data.py."
        )
    return raw_data_dir


def update_pack_progress(completed, total, status="ok", detail=""):
    """Print a compact progress line for noise-pack generation."""
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


def build_tsft_configs(tsft_values):
    """Precompute per-tsft metadata reused for every pack."""
    total_duration = NUM_FRAMES * FRAME_LENGTH
    return [
        {
            "tsft": tsft,
            "nbins": round(FBAND / (1 / tsft)),
            "nsft": int(total_duration / tsft),
            "remap_geometry": build_remap_geometry(tsft, FMIN, round(FBAND / (1 / tsft))),
        }
        for tsft in tsft_values
    ]


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
    return {"tsft": tsft, "track_index": track_index, "track_freq": track_freq, "power": np.asarray(cshuster_remap, dtype=float)}


def process_single_pack(pack, tsft_values, num_threads_per_worker):
    run_label = O3_WINDOWS[f"O3b-pack{pack}"]
    t_start, t_end = run_label
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
        tsft_results.append(process_single_tsft(tsft_config=tsft_config, t_start=t_start, t_end=t_end, framecache_path_raw_strain=framecache_path_raw_strain, num_threads_per_worker=num_threads_per_worker))
        log_verbose(f"[pack {pack}] Tsft completado: {tsft}")
    return search_candidates_in_memory(mchirp=None, distance=None, pack=pack, noise=True, tsft_results=tsft_results)


def split_packs_for_job(packs, base_jobs, job_id):
    """
    Reparte packs entre exactamente `base_jobs` jobs.
    Los primeros `remainder` jobs reciben un pack extra.
    """
    if base_jobs <= 0:
        raise ValueError("base_jobs debe ser > 0")

    npacks = len(packs)
    packs_per_job, remainder = divmod(npacks, base_jobs)
    total_jobs = base_jobs

    if job_id < 0 or job_id >= total_jobs:
        raise ValueError(f"job_id debe estar en [0, {total_jobs - 1}]")

    start = job_id * packs_per_job + min(job_id, remainder)
    end = start + packs_per_job + (1 if job_id < remainder else 0)
    return packs[start:end], packs_per_job, remainder, total_jobs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Noise data generation repartido en jobs Condor."
    )
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--job-id", type=int, default=DEFAULT_JOB_ID)
    return parser.parse_args()


def main():
    # 1) Configuracion base del job actual.
    args = parse_args()
    packs = DEFAULT_PACKS
    tsft_values = DEFAULT_TSFT_VALUES
    num_threads_per_worker = MAKE_SFT_THREADS

    # 2) Primera division: packs entre jobs (floor + resto en el ultimo job).
    packs, packs_per_job, remainder, total_jobs = split_packs_for_job(packs, args.n_jobs, args.job_id)
    if not packs:
        print(
            f"Job sin packs asignados: job_id={args.job_id}, n_jobs={args.n_jobs}. "
            "No hay trabajo que hacer."
        )
        return

    print(
        f"Reparto jobs: npacks={len(DEFAULT_PACKS)}, njobs={args.n_jobs}, "
        f"packs_por_job={packs_per_job}, resto={remainder}, total_jobs_reales={total_jobs}"
        , flush=True
    )
    print(
        f"Job actual: job_id={args.job_id}/{total_jobs - 1}, packs_asignados={len(packs)}, "
        f"primer_pack={packs[0]}, ultimo_pack={packs[-1]}, request_cpus={REQUEST_CPUS_PER_JOB}, "
        f"make_sft_threads={num_threads_per_worker}"
        , flush=True
    )

    output_csv = search_results_output_path(injected=False, n_jobs=args.n_jobs, job_id=args.job_id)
    # 3) Procesado secuencial de packs por job.
    # IMPORTANTE: cada make_SFT ya paraleliza internamente (hasta los CPUs del nodo).
    # Evitamos lanzar varios packs a la vez para no sobrecargar el job.
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
            append_search_result_row(result, output_csv)
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
        mark_search_results_job_done(injected=False, n_jobs=total_jobs, job_id=args.job_id)
        merge_search_results_if_ready(injected=False, total_jobs=total_jobs)


if __name__ == "__main__":
    main()
