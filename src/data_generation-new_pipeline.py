import argparse
import os
import tempfile

import numpy as np
from pycbc import frame as pycbc_frame
from pycbc.pnutils import mchirp_q_to_mass1_mass2
from pycbc.conversions import mchirp_from_mass1_mass2

from algortihm_final import load_sfts, preprocess_data, remap_CShuster_to_fm83, run_viterbi
from download_O3_data import download_o3_real_data
from injection_final import inject_signal_into_real_data
from make_framecache_final import generate_framecache
from make_SFT_function import run_make_sfts_script
from pipeline_paths import BIN_DIR, OUTPUTS_DATA_DIR, ensure_dir

BASH_SCRIPT_PATH = BIN_DIR / "make_SFT-final-v2.sh"

# ----------------------------- Configuracion de jobs -----------------------------
DEFAULT_N_JOBS = 341
DEFAULT_JOB_ID = 0

# ----------------------------- Configuracion de datos ----------------------------
DEFAULT_PACK = 3
O3_WINDOWS = {
    "O3b-pack3": (1257029632, 1257062400),
}

# Matriz de senales: masa_1 para todas las distancias, masa_2 para todas, etc.
MCHIRP_GRID = [1e-04, 5e-04, 8e-04, 1e-03, 2e-03, 3e-03, 4e-03, 5e-03, 6e-03, 7e-03, 8e-03, 9e-03, 1e-02, 2e-02, 3e-02, 4e-02, 5e-02, 6e-02, 7e-02, 8e-02, 9e-02, 1e-01]
DISTANCE_GRID = np.concatenate([np.array([0.001]), np.arange(0.005, 0.155, 0.005)])

# Parametros fijos de la inyeccion y analisis.
DEFAULT_T_TO_MERG = 32780
# Rango de masas cubierto por tiempos a coalescencia mas largos para que la senal
# atraviese una fraccion util de la banda 61.1-126.8 Hz.
T_TO_MERG_BY_MCHIRP = [
    (1.0000000000000000e-04, 2.2053231801153470e-04, 1.3939520546672462e07),
    (2.2053234006476653e-04, 4.8634508413599280e-04, 3.7307317803086136e06),
    (4.8634513277050120e-04, 1.0725482012884173e-03, 9.9848193808004100e05),
    (1.0725483085432376e-03, 2.3653157626732702e-03, 2.6723073068630840e05),
    (2.3653159992048467e-03, 5.2162864185682700e-03, 7.1520831350329030e04),
    (5.2162869401969120e-03, 8.3302027440089410e-03, 3.2780000000000000e04),
]
RA = 1.7
DEC = 1.7
POL = 0.2
INC = 0
MAKE_SFT_THREADS = 256
TSFT_VALUES = [2, 3, 4, 5, 6, 9, 12, 15, 21, 29, 39, 54, 74, 101, 132, 181, 248, 340]  # primera prubea: [2, 4, 7, 12, 25, 52, 106, 189, 272] 

# Parametros de data.
FRAME_LENGTH = 4096
NUM_FRAMES = 8
FMIN = 61.1
FMAX = 126.8
FBAND = FMAX - FMIN
CHANNEL_NAME = "H1:GWOSC-4KHZ_R1_STRAIN"
WINDOWTYPE = "rectangular"
IFO = "H1"
FS_TARGET = 512

# Salidas.
FREQ_PATH = ensure_dir(OUTPUTS_DATA_DIR / "tracks-frequencies_remote" / "signal")
INDEX_PATH = ensure_dir(OUTPUTS_DATA_DIR / "track-index_remote" / "signal")
POWER_PATH = ensure_dir(OUTPUTS_DATA_DIR / "Chuster-powers_remote" / "signal")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Signal data generation repartido en jobs Condor."
    )
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--job-id", type=int, default=DEFAULT_JOB_ID)
    return parser.parse_args()


def build_signal_grid(mchirp_grid, distance_grid):
    """Orden: masa1 para todas las distancias, masa2 para todas, etc."""
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


def split_signals_for_job(signal_grid, base_jobs, job_id):
    """
    Reparto en 2 fases:
    - Jobs base: `base_jobs`, cada uno con floor(total/base_jobs) senales.
    - Si hay resto, se crea un job extra para ese resto.
    """
    if base_jobs <= 0:
        raise ValueError("base_jobs debe ser > 0")

    total = len(signal_grid)
    signals_per_job = total // base_jobs
    remainder = total - (base_jobs * signals_per_job)
    total_jobs = base_jobs + (1 if remainder > 0 else 0)

    if job_id < 0 or job_id >= total_jobs:
        raise ValueError(f"job_id debe estar en [0, {total_jobs - 1}]")

    # Jobs base.
    if job_id < base_jobs:
        start = job_id * signals_per_job
        end = start + signals_per_job
        return signal_grid[start:end], signals_per_job, remainder, total_jobs

    # Job extra (solo resto).
    start = base_jobs * signals_per_job
    end = total
    return signal_grid[start:end], signals_per_job, remainder, total_jobs


def process_single_tsft(pack, mchirp, distance_str, tsft, t_start, t_end, framecache_path):
    delta_f = 1 / tsft
    nbins = round(FBAND / delta_f)
    nSFT_per_pack = int((NUM_FRAMES * FRAME_LENGTH) / tsft)

    output_txt = FREQ_PATH / f"track-freqs_Tsft-{tsft}_pack-{pack}_mc-{mchirp:.0e}_dl-{distance_str}.txt"
    output_index = INDEX_PATH / f"index-vit_Tsft-{tsft}_pack-{pack}_mc-{mchirp:.0e}_dl-{distance_str}.txt"
    output_power = POWER_PATH / f"power_Tsft-{tsft}_pack-{pack}_mc-{mchirp:.0e}_dl-{distance_str}.npy"
    output_png = None

    with tempfile.TemporaryDirectory(prefix=f"sft-pack{pack}-tsft{tsft}-") as sft_output_path:
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
        )

        data_inj = load_sfts(sft_output_path, t_start, tsft, nbins, nSFT_per_pack)
        C_inj, freqs = preprocess_data(data_inj, tsft, FMIN, FMAX)
        x_remap, C_remap = remap_CShuster_to_fm83(C_inj, freqs, x_new=None, fill_value=np.nan)

        run_viterbi(
            C_remap,
            x_remap,
            tsft,
            FMIN,
            FMAX,
            output_txt=output_txt,
            output_power=output_power,
            output_index=output_index,
            output_png=output_png,
        )


def process_single_signal(pack, t_start, t_end, raw_data_dir, raw_existing_data, injected_dir, signal_params):
    m1, m2, distance = signal_params
    mchirp = mchirp_from_mass1_mass2(m1, m2)
    distance_str = f"{distance:.3f}".replace(".", "_")
    t_to_merger = get_t_to_merger_for_mchirp(mchirp)
    print(
        f"Iniciando senal: m1={m1:.6g}, m2={m2:.6g}, dL={distance:.3f} Mpc, "
        f"mchirp={mchirp:.6g}, t_to_merger={t_to_merger:.6g}",
        flush=True,
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
    )

    for tsft in TSFT_VALUES:
        print(f"[mc={mchirp:.0e}, dl={distance_str}] Iniciando Tsft={tsft}", flush=True)
        process_single_tsft(
            pack=pack,
            mchirp=mchirp,
            distance_str=distance_str,
            tsft=tsft,
            t_start=t_start,
            t_end=t_end,
            framecache_path=framecache_path,
        )
        print(f"[mc={mchirp:.0e}, dl={distance_str}] Tsft completado={tsft}", flush=True)


def main():
    args = parse_args()
    pack = DEFAULT_PACK
    run_label = O3_WINDOWS[f"O3b-pack{pack}"]
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

    with tempfile.TemporaryDirectory(prefix=f"o3-gwf-pack{pack}-") as temp_gwf_root:
        raw_data_dir = os.path.join(temp_gwf_root, f"O3b-pack{pack}-512HZ")
        injected_dir = os.path.join(temp_gwf_root, f"O3b-pack{pack}-512HZ-injected")
        print(f"Usando carpeta temporal para .gwf: {temp_gwf_root}", flush=True)

        print("------------ Downloading real O3 data ------------", flush=True)
        download_o3_real_data(
            input_dir=raw_data_dir,
            ifo=IFO,
            run_label=run_label,
            num_frames=NUM_FRAMES,
            frame_length=FRAME_LENGTH,
            fs_target=FS_TARGET,
            channel_name=CHANNEL_NAME,
            verbose=True,
        )

        raw_existing_data = load_raw_existing_data(raw_data_dir, t_start)

        # Procesa solo las senales asignadas a este job.
        for idx, signal_params in enumerate(assigned_signals, start=1):
            print(f"Senal {idx}/{len(assigned_signals)} del job", flush=True)
            process_single_signal(
                pack=pack,
                t_start=t_start,
                t_end=t_end,
                raw_data_dir=raw_data_dir,
                raw_existing_data=raw_existing_data,
                injected_dir=injected_dir,
                signal_params=signal_params,
            )


if __name__ == "__main__":
    main()
