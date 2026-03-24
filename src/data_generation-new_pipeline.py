import argparse
import os
import tempfile
import time

import numpy as np
from pycbc import frame as pycbc_frame
from pycbc.pnutils import mchirp_q_to_mass1_mass2
from pycbc.conversions import mchirp_from_mass1_mass2

from algortihm_final import VITERBI_TRANSITION_LOG_PROBS, build_remap_geometry, load_sfts, preprocess_data, remap_CShuster_to_fm83
from download_O3_data import DEFAULT_OUTPUT_ROOT, O3_WINDOWS
from injection_final import inject_signal_into_real_data
from make_framecache_final import generate_framecache
from make_SFT_function import run_make_sfts_script
from pipeline_paths import BIN_DIR
from power_metric_prueba import search_candidates_in_memory
from search_candidate import append_search_result_row, mark_search_results_job_done, merge_search_results_if_ready, search_results_output_path
import soapcw as soap

BASH_SCRIPT_PATH = BIN_DIR / "make_SFT-final-v2.sh"

# ----------------------------- Configuracion de jobs -----------------------------
DEFAULT_N_JOBS = 300
DEFAULT_JOB_ID = 0

# ----------------------------- Configuracion de datos ----------------------------
DEFAULT_PACK = 3

# Signal parameters
MCHIRP_GRID = [1e-04, 5e-04, 8e-04, 1e-03, 2e-03, 3e-03, 4e-03, 5e-03, 6e-03, 7e-03, 8e-03, 9e-03, 1e-02, 2e-02, 3e-02, 4e-02, 5e-02, 6e-02, 7e-02, 8e-02, 9e-02, 1e-01]
DISTANCE_GRID = np.concatenate([np.array([0.001]), np.arange(0.005, 0.155, 0.005)])

# Rango de masas cubierto por tiempos a coalescencia mas largos para que la señal atraviese una fraccion util de la banda 61.1-126.8 Hz.
T_TO_MERG_BY_MCHIRP = [
    (1.0000000000000000e-04, 2.2053231801153470e-04, 1.3939520546672462e07), # mchirp_min, mchirp_max, t_to_merger
    (2.2053234006476653e-04, 4.8634508413599280e-04, 3.7307317803086136e06),
    (4.8634513277050120e-04, 1.0725482012884173e-03, 9.9848193808004100e05),
    (1.0725483085432376e-03, 2.3653157626732702e-03, 2.6723073068630840e05),
    (2.3653159992048467e-03, 5.2162864185682700e-03, 7.1520831350329030e04),
    (5.2162869401969120e-03, 8.3302027440089410e-03, 3.2780000000000000e04),
]
DEFAULT_T_TO_MERG = 32780
RA = 1.7
DEC = 1.7
POL = 0.2
INC = 0
MAKE_SFT_THREADS = 256
TSFT_VALUES = [2, 3, 4, 5, 6, 9, 12, 15, 21, 29, 39, 54, 74, 101, 132, 181, 248, 340]  # primera prubea: [2, 4, 7, 12, 25, 52, 106, 189, 272] 

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


def update_signal_progress(completed, total, status="ok", detail=""): # verbose
    """Print a compact progress line for signal generation."""
    message = f"Se han generado correctamente {completed}/{total} señales"
    if status != "ok":
        message = f"{message} | estado={status}"
    if detail:
        message = f"{message} | {detail}"
    end = "\n" if status in {"error", "done"} else "\r"
    print(message, end=end, flush=True)


def log_verbose(message):
    if PIPELINE_VERBOSE:
        print(message, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Signal data generation repartido en jobs Condor."
    )
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--job-id", type=int, default=DEFAULT_JOB_ID)
    parser.add_argument("--pack", type=int, default=DEFAULT_PACK)
    return parser.parse_args()


def build_signal_grid(mchirp_grid, distance_grid):
    """Orden: masa1, masa2 para todas las distancias."""
    grid = []
    for mchirp in mchirp_grid:
        m1, m2 = mchirp_q_to_mass1_mass2(mchirp, q=1.0)
        if m1 != m2:
            m2 = m1
        for distance in distance_grid:
            grid.append((m1, m2, distance))
    return grid


def get_t_to_merger_for_mchirp(mchirp):
    for mchirp_min, mchirp_max, t_to_merger in T_TO_MERG_BY_MCHIRP:
        if mchirp_min <= mchirp <= mchirp_max:
            return t_to_merger
    return DEFAULT_T_TO_MERG


def load_raw_existing_data(raw_data_dir, t_start):
    """Load the downloaded raw strain once so all injections reuse it."""
    raw_existing_data = []
    for i in range(NUM_FRAMES):
        start_time = t_start + i * FRAME_LENGTH
        end_time = start_time + FRAME_LENGTH
        raw_file = os.path.join(
            raw_data_dir,
            f"{IFO[0]}-{IFO}_GWOSC_O3b_4KHZ_R1-{start_time}-{FRAME_LENGTH}_resampled_512HZ.gwf",
        )
        raw_existing_data.append(
            pycbc_frame.read_frame(raw_file, CHANNEL_NAME, start_time=start_time, end_time=end_time)
        )
    return raw_existing_data


def get_local_o3_pack_dir(pack):
    """Return the local directory containing a pre-downloaded O3 pack."""
    raw_data_dir = os.path.join(DEFAULT_OUTPUT_ROOT, f"O3b-pack{pack}-{FS_TARGET}HZ")
    if not os.path.isdir(raw_data_dir):
        raise FileNotFoundError(
            f"No existe el pack O3 descargado en disco: {raw_data_dir}. "
            "Primero ejecuta src/download_O3_data.py."
        )
    return raw_data_dir


def split_signals_for_job(signal_grid, base_jobs, job_id):
    """
    Reparte senales entre exactamente `base_jobs` jobs.
    Las primeras `remainder` particiones reciben una senal extra.
    """
    if base_jobs <= 0:
        raise ValueError("base_jobs debe ser > 0")

    total = len(signal_grid)
    signals_per_job, remainder = divmod(total, base_jobs)
    total_jobs = base_jobs

    if job_id < 0 or job_id >= total_jobs:
        raise ValueError(f"job_id debe estar en [0, {total_jobs - 1}]")

    start = job_id * signals_per_job + min(job_id, remainder)
    end = start + signals_per_job + (1 if job_id < remainder else 0)
    return signal_grid[start:end], signals_per_job, remainder, total_jobs


def build_tsft_configs(tsft_values):
    """Precompute per-tsft metadata reused for every signal."""
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


TSFT_CONFIGS = build_tsft_configs(TSFT_VALUES)


def process_single_tsft(tsft_config, t_start, t_end, framecache_path):
    tsft = tsft_config["tsft"]

    with tempfile.TemporaryDirectory(prefix=f"sft-tsft{tsft}-") as sft_output_path:
        sft_start = time.perf_counter()
        run_make_sfts_script(
            bash_script_path=BASH_SCRIPT_PATH,
            t_start=t_start,
            t_end=t_end,
            sft_output_path=sft_output_path,
            framecache_path=framecache_path,
            num_threads=MAKE_SFT_THREADS,
            Tseg=tsft,
            fmin=FMIN,
            Band=FBAND,
            windowtype=WINDOWTYPE,
            channel=CHANNEL_NAME,
            verbose=SFT_VERBOSE,
        )
        sft_elapsed = time.perf_counter() - sft_start
        print(f"[TIMING] MakeSFTs tsft={tsft}: {sft_elapsed:.2f} s", flush=True)

        data_inj = load_sfts(
            sft_output_path,
            t_start,
            tsft,
            tsft_config["nbins"],
            tsft_config["nsft"],
        )
        remap_geometry = tsft_config["remap_geometry"]
        C_inj, freqs = preprocess_data(
            data_inj,
            tsft,
            FMIN,
            FMAX,
            freqs=remap_geometry["freqs"],
        )
        x_remap, C_remap = remap_CShuster_to_fm83(
            C_inj,
            freqs,
            x_new=remap_geometry["x_new"],
            fill_value=np.nan,
            x_inc=remap_geometry["x_inc"],
        )
        one_tracks_ng = soap.single_detector(VITERBI_TRANSITION_LOG_PROBS, C_remap, lookup_table=None)
        track_index = np.asarray(one_tracks_ng.vit_track, dtype=int)
        track_freq = np.asarray(x_remap[track_index], dtype=float)
        return {"tsft": tsft, "track_index": track_index, "track_freq": track_freq, "power": np.asarray(C_remap, dtype=float)}


def process_single_signal(pack, t_start, t_end, raw_data_dir, raw_existing_data, injected_dir, signal_params):
    m1, m2, distance = signal_params
    mchirp = mchirp_from_mass1_mass2(m1, m2)
    distance_str = f"{distance:.3f}".replace(".", "_")
    t_to_merger = get_t_to_merger_for_mchirp(mchirp)
    log_verbose(
        f"Iniciando senal: m1={m1:.6g}, m2={m2:.6g}, dL={distance:.3f} Mpc, "
        f"mchirp={mchirp:.6g}, t_to_merger={t_to_merger:.6g}"
    )

    coal_time = inject_signal_into_real_data(
        m1=m1,
        m2=m2,
        distance=distance,
        t_to_merg=t_to_merger,
        ra=RA,
        dec=DEC,
        pol=POL,
        inc=INC,
        ifo=IFO,
        t_start=t_start,
        num_frames=NUM_FRAMES,
        frame_length=FRAME_LENGTH,
        input_dir=raw_data_dir,
        data_dir=injected_dir,
        channel_name=CHANNEL_NAME,
        existing_data=raw_existing_data,
        verbose=PIPELINE_VERBOSE,
    )

    injected_dir_cache = os.path.join(injected_dir, f"{IFO}_inject_mc-{mchirp:.0e}_dl-{distance_str}")
    framecache_path = generate_framecache(
        input_dir=injected_dir_cache,
        mchirp=mchirp,
        distance=distance,
        det=IFO[0],
        num_frames=NUM_FRAMES,
        frame_length=FRAME_LENGTH,
        t_start=t_start,
        coal_time=coal_time,
        verbose=PIPELINE_VERBOSE,
    )

    tsft_results = []
    for tsft_config in TSFT_CONFIGS:
        tsft = tsft_config["tsft"]
        log_verbose(f"[mc={mchirp:.0e}, dl={distance_str}] Iniciando Tsft={tsft}")
        tsft_results.append(process_single_tsft(tsft_config=tsft_config, t_start=t_start, t_end=t_end, framecache_path=framecache_path))
        log_verbose(f"[mc={mchirp:.0e}, dl={distance_str}] Tsft completado={tsft}")
    return search_candidates_in_memory(mchirp=mchirp, distance=distance, pack=pack, noise=False, tsft_results=tsft_results)


def main():
    args = parse_args()
    pack = args.pack
    run_key = f"O3b-pack{pack}"
    if run_key not in O3_WINDOWS:
        raise ValueError(f"Pack no soportado: {pack}. Packs disponibles: {sorted(int(key.split('pack')[1]) for key in O3_WINDOWS)}")
    run_label = O3_WINDOWS[run_key]
    t_start, t_end = run_label

    signal_grid = build_signal_grid(MCHIRP_GRID, DISTANCE_GRID)
    assigned_signals, signals_per_job, remainder, total_jobs = split_signals_for_job(
        signal_grid, args.n_jobs, args.job_id
    )

    print(
        f"Reparto senales: total={len(signal_grid)}, njobs={args.n_jobs}, "
        f"senales_por_job={signals_per_job}, resto={remainder}, total_jobs_reales={total_jobs}",
        flush=True,
    )

    if not assigned_signals:
        print(f"Job sin senales asignadas: job_id={args.job_id}. No hay trabajo.", flush=True)
        return

    print(
        f"Job actual: job_id={args.job_id}/{total_jobs - 1}, senales_asignadas={len(assigned_signals)}",
        flush=True,
    )

    output_csv = search_results_output_path(injected=True, n_jobs=args.n_jobs, job_id=args.job_id)
    raw_data_dir = get_local_o3_pack_dir(pack)
    with tempfile.TemporaryDirectory(prefix=f"o3-gwf-pack{pack}-") as temp_gwf_root:
        injected_dir = os.path.join(temp_gwf_root, f"O3b-pack{pack}-512HZ-injected")
        log_verbose(f"Usando datos O3 ya descargados en: {raw_data_dir}")
        log_verbose(f"Usando carpeta temporal para inyecciones: {temp_gwf_root}")

        raw_existing_data = load_raw_existing_data(raw_data_dir, t_start)

        # Procesa solo las senales asignadas a este job.
        completed_signals = 0
        total_assigned_signals = len(assigned_signals)
        for idx, signal_params in enumerate(assigned_signals, start=1):
            m1, m2, distance = signal_params
            progress_detail = f"procesando senal {idx}/{total_assigned_signals} (m1={m1:.6g}, m2={m2:.6g}, dL={distance:.3f} Mpc)"
            update_signal_progress(completed_signals, total_assigned_signals, status="running", detail=progress_detail)
            try:
                result = process_single_signal(
                    pack=pack,
                    t_start=t_start,
                    t_end=t_end,
                    raw_data_dir=raw_data_dir,
                    raw_existing_data=raw_existing_data,
                    injected_dir=injected_dir,
                    signal_params=signal_params,
                )
                append_search_result_row(result, output_csv)
            except Exception as exc:
                error_detail = (
                    f"error en senal {idx}/{total_assigned_signals} "
                    f"(m1={m1:.6g}, m2={m2:.6g}, dL={distance:.3f} Mpc): {exc}"
                )
                update_signal_progress(completed_signals, total_assigned_signals, status="error", detail=error_detail)
                raise

            completed_signals += 1
            final_status = "done" if completed_signals == total_assigned_signals else "ok"
            update_signal_progress(completed_signals, total_assigned_signals, status=final_status)
    if args.n_jobs > 1:
        mark_search_results_job_done(injected=True, n_jobs=total_jobs, job_id=args.job_id)
        merge_search_results_if_ready(injected=True, total_jobs=total_jobs)


if __name__ == "__main__":
    main()
