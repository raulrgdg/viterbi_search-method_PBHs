import os
import tempfile
import argparse
import numpy as np
from make_framecache_final import generate_framecache_raw_strain
from make_SFT_function import run_make_sfts_script
from download_O3_data import download_o3_real_data
from algortihm_final import load_sfts, preprocess_data, run_viterbi, remap_CShuster_to_fm83
from pipeline_paths import BIN_DIR, OUTPUTS_DATA_DIR, ensure_dir

BASH_SCRIPT_PATH = BIN_DIR / "make_SFT-final-v2.sh"

# Case: N=noise

# Real data
O3_WINDOWS = {
    "O3b-pack1": (1256775680, 1256808448),
    "O3b-pack2": (1256841216, 1256873984),
    "O3b-pack3": (1257029632, 1257062400),
    "O3b-pack4": (1257201664, 1257234432),
    "O3b-pack5": (1257328640, 1257361408),
    "O3b-pack6": (1257467904, 1257500672),
    "O3b-pack7": (1257500672, 1257533440),
    "O3b-pack8": (1257574400, 1257607168),
    "O3b-pack9": (1257660416, 1257693184),
    "O3b-pack10": (1257984000, 1258016768),
    "O3b-pack11": (1258344448, 1258377216),
    "O3b-pack12": (1258434560, 1258467328),
    "O3b-pack13": (1258500096, 1258532864),
    "O3b-pack14": (1258631168, 1258663936),
    "O3b-pack15": (1258696704, 1258729472),
    "O3b-pack16": (1258844160, 1258876928),
    "O3b-pack17": (1258967040, 1258999808),
    "O3b-pack18": (1259155456, 1259188224),
    "O3b-pack19": (1259368448, 1259401216),
    "O3b-pack20": (1259483136, 1259515904),
    "O3b-pack21": (1259642880, 1259675648),
    "O3b-pack22": (1259675648, 1259708416),
    "O3b-pack23": (1259708416, 1259741184),
    "O3b-pack24": (1259741184, 1259773952),
    "O3b-pack25": (1259773952, 1259806720),
    "O3b-pack26": (1259884544, 1259917312),
    "O3b-pack27": (1259982848, 1260015616),
    "O3b-pack28": (1260146688, 1260179456),
    "O3b-pack29": (1260224512, 1260257280),
    "O3b-pack30": (1260290048, 1260322816),
    "O3b-pack31": (1260322816, 1260355584),
    "O3b-pack32": (1260392448, 1260425216),
    "O3b-pack33": (1260437504, 1260470272),
    "O3b-pack34": (1260470272, 1260503040),
    "O3b-pack35": (1260503040, 1260535808),
    "O3b-pack36": (1260535808, 1260568576),
    "O3b-pack37": (1260662784, 1260695552),
    "O3b-pack38": (1260756992, 1260789760),
    "O3b-pack39": (1260789760, 1260822528),
    "O3b-pack40": (1260822528, 1260855296),
    "O3b-pack41": (1260924928, 1260957696),
    "O3b-pack42": (1260957696, 1260990464),
    "O3b-pack43": (1261027328, 1261060096),
    "O3b-pack44": (1261088768, 1261121536),
    "O3b-pack45": (1261187072, 1261219840),
    "O3b-pack46": (1261301760, 1261334528),
    "O3b-pack47": (1261355008, 1261387776),
    "O3b-pack48": (1261424640, 1261457408),
    "O3b-pack49": (1261457408, 1261490176),
    "O3b-pack50": (1261498368, 1261531136),
    "O3b-pack51": (1261531136, 1261563904),
    "O3b-pack52": (1261588480, 1261621248),
    "O3b-pack53": (1261707264, 1261740032),
    "O3b-pack54": (1261740032, 1261772800),
    "O3b-pack55": (1261772800, 1261805568),
    "O3b-pack56": (1261969408, 1262002176),
    "O3b-pack57": (1262002176, 1262034944),
    "O3b-pack58": (1262047232, 1262080000),
    "O3b-pack59": (1262215168, 1262247936),
    "O3b-pack60": (1262350336, 1262383104),
    "O3b-pack61": (1262473216, 1262505984),
    "O3b-pack62": (1262604288, 1262637056),
    "O3b-pack63": (1262641152, 1262673920),
    "O3b-pack64": (1262923776, 1262956544),
    "O3b-pack65": (1263013888, 1263046656),
    "O3b-pack66": (1263075328, 1263108096),
    "O3b-pack67": (1263108096, 1263140864),
    "O3b-pack68": (1263181824, 1263214592),
    "O3b-pack69": (1263276032, 1263308800),
    "O3b-pack70": (1263333376, 1263366144),
    "O3b-pack71": (1263366144, 1263398912),
    "O3b-pack72": (1263403008, 1263435776),
    "O3b-pack73": (1263439872, 1263472640),
    "O3b-pack74": (1263534080, 1263566848),
    "O3b-pack75": (1263689728, 1263722496),
    "O3b-pack76": (1263722496, 1263755264),
    "O3b-pack77": (1264300032, 1264332800),
    "O3b-pack78": (1264332800, 1264365568),
    "O3b-pack79": (1264480256, 1264513024),
    "O3b-pack80": (1264599040, 1264631808),
    "O3b-pack81": (1264631808, 1264664576),
    "O3b-pack82": (1264664576, 1264697344),
    "O3b-pack83": (1265856512, 1265889280),
    "O3b-pack84": (1265889280, 1265922048),
    "O3b-pack85": (1265946624, 1265979392),
    "O3b-pack86": (1266003968, 1266036736),
    "O3b-pack87": (1266221056, 1266253824),
    "O3b-pack88": (1266282496, 1266315264),
    "O3b-pack89": (1266315264, 1266348032),
    "O3b-pack90": (1266364416, 1266397184),
    "O3b-pack91": (1266397184, 1266429952),
    "O3b-pack92": (1266450432, 1266483200),
    "O3b-pack93": (1266483200, 1266515968),
    "O3b-pack94": (1266548736, 1266581504),
    "O3b-pack95": (1266638848, 1266671616),
    "O3b-pack96": (1266794496, 1266827264),
    "O3b-pack97": (1266839552, 1266872320),
    "O3b-pack98": (1266966528, 1266999296),
    "O3b-pack99": (1266999296, 1267032064),
    "O3b-pack100": (1267032064, 1267064832),
    "O3b-pack101": (1267064832, 1267097600),
    "O3b-pack102": (1267097600, 1267130368),
    "O3b-pack103": (1267146752, 1267179520),
    "O3b-pack104": (1267240960, 1267273728),
    "O3b-pack105": (1267404800, 1267437568),
    "O3b-pack106": (1267511296, 1267544064),
    "O3b-pack107": (1267642368, 1267675136),
    "O3b-pack108": (1267675136, 1267707904),
}
DEFAULT_PACKS = list(range(1, 109))
DEFAULT_TSFT_VALUES = [2, 3, 4, 5, 6, 9, 12, 15, 21, 29, 39, 54, 74, 101, 132, 181, 248, 340] #quite 1 !
REQUEST_CPUS_PER_JOB = 16
DEFAULT_N_JOBS = 5
DEFAULT_JOB_ID = 0

# Depend on the number of CPUs requested. See strong_scale_test
MAKE_SFT_THREADS = 256

# Output folders
FREQ_PATH = ensure_dir(OUTPUTS_DATA_DIR / "tracks-frequencies_remote" / "noise")
INDEX_PATH = ensure_dir(OUTPUTS_DATA_DIR / "track-index_remote" / "noise")
POWER_PATH = ensure_dir(OUTPUTS_DATA_DIR / "Chuster-powers_remote" / "noise")

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


def process_single_tsft(pack, tsft, t_start, t_end, framecache_path_raw_strain, num_threads_per_worker):
    delta_f = 1 / tsft
    nbins = round(FBAND / delta_f)
    nSFT_per_pack = int((NUM_FRAMES * FRAME_LENGTH) / tsft)

    output_txt = FREQ_PATH / f"track-freqs_Tsft-{tsft}_pack-{pack}.txt"
    output_index = INDEX_PATH / f"index-vit_Tsft-{tsft}_pack-{pack}.txt"
    output_power = POWER_PATH / f"power_Tsft-{tsft}_pack-{pack}.npy"
    output_png = None

    with tempfile.TemporaryDirectory(prefix=f"sft-pack{pack}-tsft{tsft}-") as sft_output_path:
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
        )

        data = load_sfts(sft_output_path, t_start, tsft, nbins, nSFT_per_pack)
        cshuster, freqs = preprocess_data(data, tsft, FMIN, FMAX)
        x_remap, cshuster_remap = remap_CShuster_to_fm83(cshuster, freqs, fill_value=np.nan)
        run_viterbi(
            cshuster_remap,
            x_remap,
            tsft,
            FMIN,
            FMAX,
            output_txt=output_txt,
            output_power=output_power,
            output_index=output_index,
            output_png=output_png,
        )
    return tsft


def process_single_pack(pack, tsft_values, num_threads_per_worker):
    run_label = O3_WINDOWS[f"O3b-pack{pack}"]
    t_start, t_end = run_label
    with tempfile.TemporaryDirectory(prefix=f"o3-gwf-pack{pack}-") as temp_gwf_root:
        raw_data_dir = os.path.join(temp_gwf_root, f"O3b-pack{pack}-512HZ")
        print(f"Usando carpeta temporal para .gwf (pack {pack}): {temp_gwf_root}", flush=True)
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

        framecache_path_raw_strain = generate_framecache_raw_strain(
            input_dir=raw_data_dir,
            det=IFO[0],
            num_frames=NUM_FRAMES,
            frame_length=FRAME_LENGTH,
            t_start=t_start,
        )

        for tsft in tsft_values:
            print(f"[pack {pack}] Iniciando Tsft={tsft}", flush=True)
            process_single_tsft(
                pack=pack,
                tsft=tsft,
                t_start=t_start,
                t_end=t_end,
                framecache_path_raw_strain=framecache_path_raw_strain,
                num_threads_per_worker=num_threads_per_worker,
            )
            print(f"[pack {pack}] Tsft completado: {tsft}", flush=True)
    return pack


def split_packs_for_job(packs, base_jobs, job_id):
    """
    Reparto en 2 fases:
    - Jobs base: `base_jobs`, cada uno con floor(npacks/base_jobs) packs.
    - Si hay resto, se crea un job extra para ese resto.
    """
    if base_jobs <= 0:
        raise ValueError("base_jobs debe ser > 0")

    npacks = len(packs)
    packs_per_job = npacks // base_jobs
    remainder = npacks - (base_jobs * packs_per_job)
    total_jobs = base_jobs + (1 if remainder > 0 else 0)

    if job_id < 0 or job_id >= total_jobs:
        raise ValueError(f"job_id debe estar en [0, {total_jobs - 1}]")

    # Jobs base.
    if job_id < base_jobs:
        start = job_id * packs_per_job
        end = start + packs_per_job
        return packs[start:end], packs_per_job, remainder, total_jobs

    # Job extra (solo resto).
    start = base_jobs * packs_per_job
    end = npacks
    return packs[start:end], packs_per_job, remainder, total_jobs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Noise data generation repartido en jobs Condor."
    )
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--job-id", type=int, default=None)
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

    # 3) Procesado secuencial de packs por job.
    # IMPORTANTE: cada make_SFT ya paraleliza internamente (hasta los CPUs del nodo).
    # Evitamos lanzar varios packs a la vez para no sobrecargar el job.
    for pack in packs:
        print(f"Iniciando pack {pack}", flush=True)
        pack_done = process_single_pack(pack, tsft_values, num_threads_per_worker)
        print(f"Pack completado: {pack_done}", flush=True)


if __name__ == "__main__":
    main()
