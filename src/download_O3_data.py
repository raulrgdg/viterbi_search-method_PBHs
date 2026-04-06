import argparse
import os
import time
import urllib.error

import numpy as np
from gwpy.timeseries import TimeSeries as GWpyTimeSeries
from pycbc import frame as pycbc_frame
from pycbc import types as pycbc_types

from pipeline_paths import INPUTS_O3_DATA_DIR, ensure_dir


O3_WINDOWS = {
    "O3b-pack1": (1256775680, 1256808448), "O3b-pack2": (1256841216, 1256873984), "O3b-pack3": (1257029632, 1257062400),
    "O3b-pack4": (1257201664, 1257234432), "O3b-pack5": (1257328640, 1257361408), "O3b-pack6": (1257467904, 1257500672),
    "O3b-pack7": (1257500672, 1257533440), "O3b-pack8": (1257574400, 1257607168), "O3b-pack9": (1257660416, 1257693184),
    "O3b-pack10": (1257984000, 1258016768), "O3b-pack11": (1258344448, 1258377216), "O3b-pack12": (1258434560, 1258467328),
    "O3b-pack13": (1258500096, 1258532864), "O3b-pack14": (1258631168, 1258663936), "O3b-pack15": (1258696704, 1258729472),
    "O3b-pack16": (1258844160, 1258876928), "O3b-pack17": (1258967040, 1258999808), "O3b-pack18": (1259155456, 1259188224),
    "O3b-pack19": (1259368448, 1259401216), "O3b-pack20": (1259483136, 1259515904), "O3b-pack21": (1259642880, 1259675648),
    "O3b-pack22": (1259675648, 1259708416), "O3b-pack23": (1259708416, 1259741184), "O3b-pack24": (1259741184, 1259773952),
    "O3b-pack25": (1259773952, 1259806720), "O3b-pack26": (1259884544, 1259917312), "O3b-pack27": (1259982848, 1260015616),
    "O3b-pack28": (1260146688, 1260179456), "O3b-pack29": (1260224512, 1260257280), "O3b-pack30": (1260290048, 1260322816),
    "O3b-pack31": (1260322816, 1260355584), "O3b-pack32": (1260392448, 1260425216), "O3b-pack33": (1260437504, 1260470272),
    "O3b-pack34": (1260470272, 1260503040), "O3b-pack35": (1260503040, 1260535808), "O3b-pack36": (1260535808, 1260568576),
    "O3b-pack37": (1260662784, 1260695552), "O3b-pack38": (1260756992, 1260789760), "O3b-pack39": (1260789760, 1260822528),
    "O3b-pack40": (1260822528, 1260855296), "O3b-pack41": (1260924928, 1260957696), "O3b-pack42": (1260957696, 1260990464),
    "O3b-pack43": (1261027328, 1261060096), "O3b-pack44": (1261088768, 1261121536), "O3b-pack45": (1261187072, 1261219840),
    "O3b-pack46": (1261301760, 1261334528), "O3b-pack47": (1261355008, 1261387776), "O3b-pack48": (1261424640, 1261457408),
    "O3b-pack49": (1261457408, 1261490176), "O3b-pack50": (1261498368, 1261531136), "O3b-pack51": (1261531136, 1261563904),
    "O3b-pack52": (1261588480, 1261621248), "O3b-pack53": (1261707264, 1261740032), "O3b-pack54": (1261740032, 1261772800),
    "O3b-pack55": (1261772800, 1261805568), "O3b-pack56": (1261969408, 1262002176), "O3b-pack57": (1262002176, 1262034944),
    "O3b-pack58": (1262047232, 1262080000), "O3b-pack59": (1262215168, 1262247936), "O3b-pack60": (1262350336, 1262383104),
    "O3b-pack61": (1262473216, 1262505984), "O3b-pack62": (1262604288, 1262637056), "O3b-pack63": (1262641152, 1262673920),
    "O3b-pack64": (1262923776, 1262956544), "O3b-pack65": (1263013888, 1263046656), "O3b-pack66": (1263075328, 1263108096),
    "O3b-pack67": (1263108096, 1263140864), "O3b-pack68": (1263181824, 1263214592), "O3b-pack69": (1263276032, 1263308800),
    "O3b-pack70": (1263333376, 1263366144), "O3b-pack71": (1263366144, 1263398912), "O3b-pack72": (1263403008, 1263435776),
    "O3b-pack73": (1263439872, 1263472640), "O3b-pack74": (1263534080, 1263566848), "O3b-pack75": (1263689728, 1263722496),
    "O3b-pack76": (1263722496, 1263755264), "O3b-pack77": (1264300032, 1264332800), "O3b-pack78": (1264332800, 1264365568),
    "O3b-pack79": (1264480256, 1264513024), "O3b-pack80": (1264599040, 1264631808), "O3b-pack81": (1264631808, 1264664576),
    "O3b-pack82": (1264664576, 1264697344), "O3b-pack83": (1265856512, 1265889280), "O3b-pack84": (1265889280, 1265922048),
    "O3b-pack85": (1265946624, 1265979392), "O3b-pack86": (1266003968, 1266036736), "O3b-pack87": (1266221056, 1266253824),
    "O3b-pack88": (1266282496, 1266315264), "O3b-pack89": (1266315264, 1266348032), "O3b-pack90": (1266364416, 1266397184),
    "O3b-pack91": (1266397184, 1266429952), "O3b-pack92": (1266450432, 1266483200), "O3b-pack93": (1266483200, 1266515968),
    "O3b-pack94": (1266548736, 1266581504), "O3b-pack95": (1266638848, 1266671616), "O3b-pack96": (1266794496, 1266827264),
    "O3b-pack97": (1266839552, 1266872320), "O3b-pack98": (1266966528, 1266999296), "O3b-pack99": (1266999296, 1267032064),
    "O3b-pack100": (1267032064, 1267064832), "O3b-pack101": (1267064832, 1267097600), "O3b-pack102": (1267097600, 1267130368),
    "O3b-pack103": (1267146752, 1267179520), "O3b-pack104": (1267240960, 1267273728), "O3b-pack105": (1267404800, 1267437568),
    "O3b-pack106": (1267511296, 1267544064), "O3b-pack107": (1267642368, 1267675136), "O3b-pack108": (1267675136, 1267707904),
}

DEFAULT_IFO = "H1"
DEFAULT_FS_TARGET = 512
DEFAULT_FRAME_LENGTH = 4096
DEFAULT_NUM_FRAMES = 8
DEFAULT_CHANNEL_NAME = "H1:GWOSC-4KHZ_R1_STRAIN"
DEFAULT_OUTPUT_ROOT = str(INPUTS_O3_DATA_DIR)
DEFAULT_PACKS = sorted(int(key.split("pack")[1]) for key in O3_WINDOWS)
DEFAULT_N_JOBS = 1
DEFAULT_JOB_ID = 0


def download_o3_real_data(
    *,
    input_dir: str,
    ifo: str,
    run_label: tuple[int, int],
    num_frames: int,
    frame_length: int,
    fs_target: int,
    channel_name: str,
    verbose: bool = True,
    retry_attempts: int = 3,
    retry_wait_seconds: int = 5,
):
    """Download open O3 strain, resample it, and persist it as GWF frames."""
    t_start, t_end = run_label

    os.makedirs(input_dir, exist_ok=True)
    expected_n = frame_length * fs_target
    generated_files = []
    retryable_errors = (
        urllib.error.ContentTooShortError,
        urllib.error.URLError,
        urllib.error.HTTPError,
        TimeoutError,
        ConnectionError,
        OSError,
    )

    for i in range(num_frames):
        seg_start = t_start + i * frame_length
        seg_end = seg_start + frame_length

        ts = None
        last_error = None
        for attempt in range(1, retry_attempts + 2):
            try:
                # Force fresh download on retries to avoid reusing partial cache files.
                ts = GWpyTimeSeries.fetch_open_data(
                    ifo,
                    seg_start,
                    seg_end,
                    sample_rate=4096,
                    verbose=verbose,
                    cache=(attempt == 1),
                )
                break
            except retryable_errors as exc:
                last_error = exc
                if attempt > retry_attempts:
                    raise RuntimeError(
                        f"No se pudo descargar el segmento [{seg_start}, {seg_end}) "
                        f"despues de {retry_attempts + 1} intentos."
                    ) from exc

                wait_s = retry_wait_seconds * attempt
                if verbose:
                    print(
                        f"Descarga fallida para [{seg_start}, {seg_end}) "
                        f"(intento {attempt}/{retry_attempts + 1}): {exc}. "
                        f"Reintentando en {wait_s}s..."
                    )
                time.sleep(wait_s)

        if ts is None:
            raise RuntimeError(
                f"No se pudo obtener datos para [{seg_start}, {seg_end})"
            ) from last_error

        if int(ts.sample_rate.value) != fs_target:
            ts = ts.resample(fs_target)

        data = np.asarray(ts.value, dtype=np.float64)
        if data.size < expected_n:
            data = np.pad(data, (0, expected_n - data.size), mode="constant")
        elif data.size > expected_n:
            data = data[:expected_n]

        pycbc_ts = pycbc_types.TimeSeries(data, delta_t=1.0 / fs_target, epoch=seg_start)
        output_file = os.path.join(
            input_dir,
            f"{ifo[0]}-{ifo}_GWOSC_O3b_4KHZ_R1-{seg_start}-{frame_length}_resampled_512HZ.gwf",
        )
        pycbc_frame.write_frame(output_file, channel_name, pycbc_ts)
        generated_files.append(output_file)
        if verbose:
            print(f"Real data descargada y guardada: {output_file}")

    return generated_files


def download_o3_packs(packs: list[int] | None = None):
    """Download the selected O3 packs into inputs/O3_data."""
    ensure_dir(INPUTS_O3_DATA_DIR)
    selected_packs = (
        sorted(int(key.split("pack")[1]) for key in O3_WINDOWS)
        if packs is None
        else list(packs)
    )

    generated_by_pack = {}
    total_packs = len(selected_packs)
    for index, pack in enumerate(selected_packs, start=1):
        run_key = f"O3b-pack{pack}"
        if run_key not in O3_WINDOWS:
            raise ValueError(
                f"Pack no soportado: {pack}. "
                f"Packs disponibles: {sorted(int(key.split('pack')[1]) for key in O3_WINDOWS)}"
            )

        output_dir = os.path.join(DEFAULT_OUTPUT_ROOT, f"O3b-pack{pack}")
        print(f"[{index}/{total_packs}] Descargando O3b-pack{pack} -> {output_dir}", flush=True)
        generated_by_pack[pack] = download_o3_real_data(
            input_dir=output_dir,
            ifo=DEFAULT_IFO,
            run_label=O3_WINDOWS[run_key],
            num_frames=DEFAULT_NUM_FRAMES,
            frame_length=DEFAULT_FRAME_LENGTH,
            fs_target=DEFAULT_FS_TARGET,
            channel_name=DEFAULT_CHANNEL_NAME,
            verbose=True,
        )

    return generated_by_pack


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
    parser = argparse.ArgumentParser(description="Download O3 packs into inputs/O3_data.")
    parser.add_argument(
        "--packs",
        type=str,
        default="all",
        help='Lista de packs separada por comas o "all". Ejemplo: 1,2,3',
    )
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--job-id", type=int, default=DEFAULT_JOB_ID)
    return parser.parse_args()


def parse_pack_selection(raw_packs: str) -> list[int] | None:
    """Parse the CLI pack selector."""
    if raw_packs.strip().lower() == "all":
        return None
    return [int(token.strip()) for token in raw_packs.split(",") if token.strip()]


def main():
    args = parse_args()
    requested_packs = parse_pack_selection(args.packs)
    selected_packs = DEFAULT_PACKS if requested_packs is None else requested_packs
    assigned_packs, packs_per_job, remainder, total_jobs = split_packs_for_job(
        selected_packs,
        args.n_jobs,
        args.job_id,
    )
    if not assigned_packs:
        print(
            f"Job sin packs asignados: job_id={args.job_id}, n_jobs={args.n_jobs}. "
            "No hay trabajo que hacer.",
            flush=True,
        )
        return

    print(
        f"Reparto descarga O3: npacks={len(selected_packs)}, njobs={args.n_jobs}, "
        f"packs_por_job={packs_per_job}, resto={remainder}, total_jobs_reales={total_jobs}",
        flush=True,
    )
    print(
        f"Job actual: job_id={args.job_id}/{total_jobs - 1}, packs_asignados={len(assigned_packs)}, "
        f"primer_pack={assigned_packs[0]}, ultimo_pack={assigned_packs[-1]}",
        flush=True,
    )

    generated_by_pack = download_o3_packs(assigned_packs)
    print(
        f"Descarga completada: {len(generated_by_pack)} pack(s) en {DEFAULT_OUTPUT_ROOT}",
        flush=True,
    )


if __name__ == "__main__":
    main()
