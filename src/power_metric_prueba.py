import numpy as np
import csv
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from fit_per_window_prueba_new_recorte_v2 import fit_slope_candidate_blocks, expansion_block, fit_slope_candidate_significant__blocks
from pipeline_paths import OUTPUTS_PLOTS_DIR, OUTPUTS_REPORTS_DIR, ensure_dir, get_pack_product_dir

PLOTS_CANDIDATE_DIR = ensure_dir(OUTPUTS_PLOTS_DIR / "plots-candidate")
REPORTS_DIR = ensure_dir(OUTPUTS_REPORTS_DIR)
DEBUG = False
SIGNIFICANT_BLOCK_Z_THRESHOLD = 7.6869 # 90% percentil
TOP_N_BLOCKS = 2
POWER_THRESHOLD_K = 0.5
LINEAR_THRESHOLD_SLOPE = 255.5
LINEAR_THRESHOLD_INTERCEPT = -0.671286


def build_index_track_path(tsft, pack, noise, mchirp=None, distance_str=None):
    """Build the index-track path for signal or noise mode."""
    track_index_dir = get_pack_product_dir(pack, "track-index_remote", "noise" if noise else "signal")
    if noise:
        return track_index_dir / f"index-vit_Tsft-{tsft}_pack-{pack}.txt"
    return (
        track_index_dir
        / f"index-vit_Tsft-{tsft}_pack-{pack}_mc-{mchirp:.0e}_dl-{distance_str}.txt"
    )


def build_power_path(tsft, pack, noise, mchirp=None, distance_str=None):
    """Build the spectrogram-power path for signal or noise mode."""
    power_dir = get_pack_product_dir(pack, "Chuster-powers_remote", "noise" if noise else "signal")
    if noise:
        return power_dir / f"power_Tsft-{tsft}_pack-{pack}.npy"
    return (
        power_dir
        / f"power_Tsft-{tsft}_pack-{pack}_mc-{mchirp:.0e}_dl-{distance_str}.npy"
    )


def build_freq_track_path(tsft, pack, noise, mchirp=None, distance_str=None):
    """Build the frequency-track path for signal or noise mode."""
    track_freq_dir = get_pack_product_dir(pack, "tracks-frequencies_remote", "noise" if noise else "signal")
    if noise:
        return track_freq_dir / f"track-freqs_Tsft-{tsft}_pack-{pack}.txt"
    return (
        track_freq_dir
        / f"track-freqs_Tsft-{tsft}_pack-{pack}_mc-{mchirp:.0e}_dl-{distance_str}.txt"
    )


def load_track_txt(path_txt):
    """Load a 1D track from a text file."""
    return np.loadtxt(path_txt)


def load_spectrogram_npy(path_npy, expected_time_len=None):
    """Load a 2D spectrogram and align it as [time, frequency]."""
    data = np.load(path_npy, mmap_mode="r")
    if data.ndim != 2:
        raise ValueError(f"El espectrograma debe ser 2D, recibido: {data.shape}")

    if expected_time_len is not None:
        if data.shape[0] != expected_time_len and data.shape[1] == expected_time_len:
            data = data.T
        if data.shape[0] < expected_time_len:
            raise ValueError(
                f"El espectrograma tiene {data.shape[0]} pasos de tiempo y el track {expected_time_len}"
            )
        if data.shape[0] > expected_time_len:
            data = data[:expected_time_len]

    return data


def split_track_windows(track, n_windows):
    """Split a track into `n_windows` non-overlapping windows."""
    if n_windows <= 0:
        raise ValueError("n_windows debe ser > 0")

    n = len(track)
    if n_windows > n:
        raise ValueError("n_windows no puede ser mayor que la longitud del track")

    edges = np.linspace(0, n, n_windows + 1, dtype=int)
    starts = edges[:-1]
    ends = edges[1:]
    return starts, ends


def window_power_metric(track, data, n_windows, Nsft):
    """Compute per-window track power and normalized fractions."""
    starts, ends = split_track_windows(track, n_windows)

    if data.ndim != 2:
        raise ValueError(f"data debe ser 2D [tiempo, frecuencia], recibido: {data.shape}")
    if data.shape[0] < len(track):
        raise ValueError(
            f"Longitud temporal insuficiente en data ({data.shape[0]}) para track ({len(track)})"
        )

    track_idx = np.rint(track).astype(np.int64, copy=False)
    track_idx = np.clip(track_idx, 0, data.shape[1] - 1)
    point_power = data[np.arange(len(track_idx)), track_idx]
    powers = np.add.reduceat(point_power, starts)

    total_power = np.sum(powers)
    print(f"Total power across all windows normalized by Nsft: {total_power/Nsft:.6e}")
    if n_windows == 1:
        return total_power
    
    if total_power <= 0:
        fractions = np.zeros_like(powers)
    else:
        fractions = powers / total_power

    return starts, ends, total_power, powers, fractions


def significant_block(powers, top_block_idx, z_threshold):
    """Evalúa si el bloque top destaca de forma robusta frente al resto."""
    powers = np.asarray(powers, dtype=float)
    if powers.ndim != 1 or len(powers) == 0:
        raise ValueError("powers debe ser un array 1D no vacío")
    if not (0 <= int(top_block_idx) < len(powers)):
        raise IndexError("top_block_idx fuera de rango")

    med = np.median(powers)
    mad = np.median(np.abs(powers - med))
    robust_std = 1.4826 * mad

    block_power = powers[int(top_block_idx)]
    score = (block_power - med) / robust_std

    flag = (
        np.isclose(block_power, np.max(powers))
        and (score > z_threshold)
    )
    return flag, score


def select_top_windows(starts, ends, fractions, n_top, k):
    """
    Selecciona las n_top ventanas individuales con fracción por encima de:
      mediana(fractions) + k * MAD(fractions)

    No agrupa ventanas contiguas. Cada elemento devuelto representa una única
    ventana y tiene formato:
      (window_start, window_end, window_fraction)

    Además devuelve un flag que indica si la mejor ventana es claramente
    dominante frente al resto según `significant_block`.
    """
    if n_top <= 0:
        raise ValueError("n_top debe ser > 0")
    if k < 0:
        raise ValueError("k debe ser >= 0")

    starts = np.asarray(starts, dtype=int)
    ends = np.asarray(ends, dtype=int)
    fractions = np.asarray(fractions, dtype=float)

    if not (len(starts) == len(ends) == len(fractions)):
        raise ValueError("starts, ends y fractions deben tener la misma longitud")
    if len(fractions) == 0:
        return [], False

    ratio_median = np.median(fractions)
    mad = np.median(np.abs(fractions - ratio_median))
    threshold = ratio_median + k * mad
    mask = fractions > threshold

    blocks = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1

        block_start = starts[i]
        block_end = ends[j]
        block_ratio = float(np.mean(fractions[i : j + 1]))
        blocks.append((block_start, block_end, block_ratio, i))
        i = j + 1

    if not blocks:
        return [], False

    blocks.sort(key=lambda x: x[2], reverse=True)
    windows = blocks

    best_idx = windows[0][3]
    flag, score = significant_block(
        fractions,
        best_idx,
        z_threshold=SIGNIFICANT_BLOCK_Z_THRESHOLD,
    )
    csv_path = REPORTS_DIR / "signal_significant_block-6-test.csv"
    with csv_path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([flag, score])

    print(
        f"Best window significance: idx={best_idx}, "
        f"score={score:.4f}, flag={flag}"
    )

    if n_top >= 2 and len(windows) >= 2:
        best_ratio = windows[0][2]
        second_ratio = windows[1][2]
        if best_ratio >= 3.0 * second_ratio:
            return [windows[0][:3]], flag

    return [w[:3] for w in windows[:n_top]], flag


def _print_window_power_summary(starts, ends, powers, fractions, opt_tsft):
    """Print per-window power diagnostics for a selected tsft."""
    print(f"Power por ventana (tsft={opt_tsft} s):")
    for s, e, p, f in zip(starts, ends, powers, fractions):
        t0 = s * opt_tsft
        t1 = e * opt_tsft
        print(f"[{t0}:{t1}] power={p:.6e} frac={f:.4f}")


def _select_optimal_block(best_blocks, opt_tsft):
    """Pick the lowest-NMSE block, breaking ties by longer duration."""
    best_nmse = np.inf
    optimal_block = None
    optimal_length = 0.0

    for block in best_blocks:
        nmse_values = np.asarray(block.get("best_nmse", []), dtype=float)
        if nmse_values.size == 0:
            continue
        current_nmse = float(np.min(nmse_values))
        if not np.isfinite(current_nmse):
            continue

        t0 = block["block_start"] * opt_tsft
        t1 = block["block_end"] * opt_tsft
        block_length = t1 - t0
        if current_nmse < best_nmse or (
            np.isclose(current_nmse, best_nmse, rtol=0.05, atol=1e-6)
            and block_length > optimal_length
        ):
            optimal_block = block
            best_nmse = current_nmse
            optimal_length = block_length

    return optimal_block


def _plot_optimal_block(track, block, noise, pack, opt_tsft, mchirp, distance_str):
    """Persist a diagnostic plot for the selected block."""
    if noise:
        picture = PLOTS_CANDIDATE_DIR / "noise" / f"track_with_optimum_block_noise_pack-{pack}.png"
    else:
        picture = (
            PLOTS_CANDIDATE_DIR
            / "signal"
            / f"track_with_optimum_block_mc-{mchirp:.0e}_dl-{distance_str}_pack-{pack}.png"
        )

    plot_track_with_best_block_windows(track, block, picture=str(picture), tsft=opt_tsft)
    plt.close()


def power_noise_track(n_windows, tsft_list, packs_list):
    """Compute and persist noise track power metrics for all packs/tsfts."""
    csv_path = REPORTS_DIR / "power_noise_result_2-test-version2.csv"
    with csv_path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pack", "tsft", "total_power"])

    for pack in packs_list:
        for tsft in tsft_list:
            print(f"Procesando pack {pack} con tsft={tsft}...")
            path_track = build_index_track_path(tsft=tsft, pack=pack, noise=True)
            path_data = build_power_path(tsft=tsft, pack=pack, noise=True)
            
            track = load_track_txt(path_track)
            data = load_spectrogram_npy(path_data, expected_time_len=len(track))
            Nsft = len(track)
            print(f'For Tsftt={tsft}, nsft={Nsft}')

            total_power = window_power_metric(track, data, n_windows=n_windows, Nsft=Nsft)
            with csv_path.open("a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([pack, tsft, total_power])
    return csv_path

def first_power_check(mchirp, distance_str, noise, tsft_list, pack):
    """Select the best tsft using signal-to-noise power ratio against noise stats."""

    csv_path = REPORTS_DIR / "noise_metrics_2-test.csv"
    if not csv_path.exists():
        csv_path = REPORTS_DIR / "noise_metrics_2-test-version2.csv"
    df = pd.read_csv(csv_path)
    df["tsft"] = df["tsft"].astype(int)
    mean_noise_power_map = dict(zip(df["tsft"], df["mean_total_power"], strict=False))
    std_noise_power_map = dict(zip(df["tsft"], df["std_total_power"], strict=False))

    n_sigma_threshold = 1.0001
    best_n_sigma_above_threshold = -np.inf
    best_n_sigma_overall = -np.inf
    opt_tsft = 0
    opt_nsigma = -np.inf  
    found_above_threshold = False

    for tsft in tsft_list:
        print(f"Procesando pack {pack} con tsft={tsft}...")
        path_index_track = build_index_track_path(
            tsft=tsft, pack=pack, noise=noise, mchirp=mchirp, distance_str=distance_str
        )
        path_data = build_power_path(
            tsft=tsft, pack=pack, noise=noise, mchirp=mchirp, distance_str=distance_str
        )

        track = load_track_txt(path_index_track)
        data = load_spectrogram_npy(path_data, expected_time_len=len(track))
        Nsft = len(track)

        total_power= window_power_metric(track, data, n_windows=1, Nsft=Nsft)
        mean_noise = mean_noise_power_map[tsft]
        std_noise = std_noise_power_map[tsft]
        n_sigma = (total_power - mean_noise) / std_noise # n_sigma

        # Fallback: keep best global n_sigma if no threshold pass appears.
        if n_sigma > best_n_sigma_overall:
            best_n_sigma_overall = n_sigma
            opt_nsigma = n_sigma
            opt_tsft = tsft

        # Primary criterion: best n_sigma above threshold.
        if n_sigma > n_sigma_threshold and n_sigma > best_n_sigma_above_threshold:
            found_above_threshold = True
            best_n_sigma_above_threshold = n_sigma
            opt_nsigma = n_sigma
            opt_tsft = tsft
            print(
                f"Nuevo mejor n_sigma > {n_sigma_threshold:.3f}: "
                f"n_sigma={opt_nsigma:.6f}, tsft={opt_tsft}"
            )
        else:
            print(
                f"tsft={tsft}: n_sigma={n_sigma:.6f}. "
                f"Óptimo actual -> tsft={opt_tsft}, n_sigma={opt_nsigma:.6f}"
            )

    if not found_above_threshold:
        print(
            f" !!!!! No se encontró ningún tsft con n_sigma > {n_sigma_threshold:.3f}. "
            f"Se toma el mayor n_sigma disponible: tsft={opt_tsft}, n_sigma={opt_nsigma:.6f}. !!!!!"
        )
    
    return opt_tsft, opt_nsigma, found_above_threshold

def second_power_check(path_track, path_data, opt_tsft, n_windows):
    """Run the second-stage power window screening and return candidate blocks."""
    track = load_track_txt(path_track)
    data = load_spectrogram_npy(path_data, expected_time_len=len(track))
    Nsft = len(track)

    starts, ends, total_power, powers, fractions = window_power_metric(track, data, n_windows=n_windows, Nsft=Nsft)
    _print_window_power_summary(starts, ends, powers, fractions, opt_tsft)

    blocks, flag = select_top_windows(starts, ends, fractions, n_top=TOP_N_BLOCKS, k=POWER_THRESHOLD_K)
    print("Bloques con más contribución:")
    for s, e, f in blocks:
        t0 = s * opt_tsft
        t1 = e * opt_tsft
        print(f"[{t0}:{t1}] frac={f:.4f} del total power track")

    return blocks, flag

def second_fit_check(path_track, path_data, opt_tsft, n_windows):
    """Legacy variant of second-stage check retained for compatibility."""
    track = load_track_txt(path_track)
    data = load_spectrogram_npy(path_data, expected_time_len=len(track))
    Nsft = len(track)

    starts, ends, total_power, powers, fractions = window_power_metric(track, data, n_windows=n_windows, Nsft=Nsft)
    _print_window_power_summary(starts, ends, powers, fractions, opt_tsft)

    blocks, _ = select_top_windows(starts, ends, fractions, n_top=TOP_N_BLOCKS, k=POWER_THRESHOLD_K)
    print("Bloques con más contribución:")
    for s, e, f in blocks:
        t0 = s * opt_tsft
        t1 = e * opt_tsft
        print(f"[{t0}:{t1}] frac={f:.4f} del total power track")

    return blocks


def plot_track_with_best_block_windows(track, best_blocks, picture, tsft):
    """Plot full track and mark selected block/window boundaries."""
    y = np.asarray(track, dtype=float)
    x = np.arange(len(y), dtype=float) * float(tsft)

    blocks = best_blocks if isinstance(best_blocks, list) else [best_blocks]
    pairs = []
    for b in blocks:
        if len(blocks) == 1:
            starts = b["starts"][0]
            ends = b["ends"][-1]
            pairs.append((int(b["starts"][0]), int(b["ends"][-1])))
            continue
        else:
            starts = b.get("starts")
            ends = b.get("ends")
            if starts is not None and ends is not None:
                pairs.extend(zip(np.asarray(starts, dtype=int), np.asarray(ends, dtype=int)))
            elif "block_start" in b and "block_end" in b:
                pairs.append((int(b["block_start"]), int(b["block_end"])))

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(x, y, color="black", lw=1.0, label="track")

    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(pairs))))
    for i, (s, e) in enumerate(pairs):
        c = colors[i]
        ax.axvline(s * tsft, color=c, ls="--", lw=1.5, alpha=0.9)
        ax.axvline(e * tsft, color=c, ls="-", lw=1.5, alpha=0.9)

    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Track completo con ventanas de best_block")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    Path(picture).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(picture, dpi=300)


def _candidate_output_path(noise, pack):
    """Return candidate report output path."""
    if noise:
        return str(REPORTS_DIR / "candidates_noise.txt")
    return str(REPORTS_DIR / f"candidates_signal_pack-{pack}-prueba-corta2.txt")


def _write_candidate_report(
    output_txt,
    is_noise,
    passed,
    pack,
    mchirp,
    distance_str,
    block,
    opt_tsft,
    slope_eval,
    mass_eval,
    nmse_raw_eval,
    nmse_eval,
    nmse_candidate_th,
):
    """Write a candidate report entry for both signal and noise flows."""
    if not DEBUG:
        return

    if is_noise:
        header = (
            f"------ Candidate found for noise pack {pack}: \n"
            if passed
            else f"------ Candidate !!! NOT found !!! for noise pack {pack}: \n"
        )
    else:
        header = (
            f"------ Candidate found for injected signal: mchirp={mchirp:.0e}, distance={distance_str} Mpc: \n"
            if passed
            else f"------ Candidate !!! NOT found !!! for injected signal: mchirp={mchirp:.0e}, distance={distance_str} Mpc: \n"
        )

    report_lines = [
        header.rstrip("\n"),
        f"Optimal Block: t_start={block['block_start']*opt_tsft:.2f}s, t_end={block['block_end']*opt_tsft:.2f}s",
        f"Slope: {slope_eval}",
        f"Chirp Mass: {mass_eval}",
        f"NMSE_raw: {nmse_raw_eval}",
        f"NMSE_penalized: {nmse_eval}",
        f"Threshold_used: {nmse_candidate_th}",
    ]

    with open(output_txt, "a") as f:
        f.write(header)
        for line in report_lines[1:]:
            f.write(f"{line}\n")


def _candidate_threshold_nsigma(nmse_eval):
    """Return the nsigma threshold from the calibrated line."""
    return LINEAR_THRESHOLD_SLOPE * nmse_eval + LINEAR_THRESHOLD_INTERCEPT

def search_candidates(mchirp, distance, tsft_list, pack, noise):
    nmse_eval = None
    extended_optimal_block = None
    distance_str = f"{distance:.3f}".replace('.', '_') if distance is not None else "noise"
    nmse_candidate_th = 0.05
    nmse_nref = 64
    nmse_len_alpha = 1.0

    # ------- First Check 
    n_windows = 1
    
    opt_tsft, opt_nsigma, found_above_threshold = first_power_check(mchirp=mchirp, distance_str=distance_str, noise=noise, tsft_list=tsft_list, pack=pack)
    print(f"Results: opt_tsft={opt_tsft} with ratio signal/noise={opt_nsigma:.4f}, found_above_threshold={found_above_threshold}")

    # ------- Second Check
    n_windows = 8  # 4096s windows
    path_index_track = build_index_track_path(tsft=opt_tsft, pack=pack, noise=noise, mchirp=mchirp, distance_str=distance_str)
    path_data = build_power_path(tsft=opt_tsft, pack=pack, noise=noise, mchirp=mchirp, distance_str=distance_str)

    top_blocks, flag = second_power_check(path_track=path_index_track, path_data=path_data, opt_tsft=opt_tsft, n_windows=n_windows)
    if not top_blocks:
        print("No se encontraron bloques con contribución suficiente en second_power_check. Fin de ejecución.")
        return None, nmse_eval, opt_nsigma, extended_optimal_block
    
    # ------- Ejemplo Third Check
    for block_start, block_end, ratio in top_blocks:
        t0 = block_start * opt_tsft
        t1 = block_end * opt_tsft
        print(f"Analizando bloque [{t0}:{t1}] con contribución {ratio:.4f} del total power track")

    path_freq_track = build_freq_track_path(tsft=opt_tsft, pack=pack, noise=noise, mchirp=mchirp, distance_str=distance_str)
    track = load_track_txt(path_freq_track)

    best_blocks, significant_block= fit_slope_candidate_blocks(
                track=track,
                tsft=opt_tsft,
                candidate_blocks=top_blocks,
                flag=flag,
                n_windows_per_block=1,
                blocks_in_time=False,
                )

    # if not best_blocks:
    #     print("! No se encontraron bloques que ajusten el modelo en ventana de t=4096 seg, probando ventana  t=1024seg ")
    #     best_blocks, significant_block= fit_slope_candidate_blocks(
    #             track=track,
    #             tsft=opt_tsft,
    #             candidate_blocks=top_blocks,
    #             flag=flag,
    #             n_windows_per_block=4,
    #             blocks_in_time=False,
    #             )

    if flag:
        print("Bloque muy significativo, con alto nmse. Alta probabilidad de señal de alta masa: efímera pero potente")

        best_blocks, significant_block = fit_slope_candidate_significant__blocks(
            track=track,
            tsft=opt_tsft,
            candidate_blocks=top_blocks,
            n_windows_per_block=8,
            blocks_in_time=False,
    )
    # if not best_blocks:
    #     print("No se encontraron bloques candidatos en best_blocks")
    # else:
    #     best_best_block = best_blocks[0]

    #     if best_best_block['best_nmse'][0] > 0.05 and flag:
    #         print("Bloque muy significativo, con alto nmse. Alta probabilidad de señal de alta masa: efímera pero potente")

    #         best_blocks, significant_block = fit_slope_candidate_significant__blocks(
    #             track=track,
    #             tsft=opt_tsft,
    #             candidate_blocks=top_blocks,
    #             n_windows_per_block=8,
    #             blocks_in_time=False,
    #         )

    if not best_blocks:
        print("Ningún bloque candidato pasó el Third check fit (probablemente por masa negativa o ajuste inválido). Fin de ejecución.")
        return None, nmse_eval, opt_nsigma, extended_optimal_block

    for i in range(len(best_blocks)):
        b = best_blocks[i] if isinstance(best_blocks, list) else best_blocks
        print(
            f"Mejor bloque: t_start={b['starts'][0]*opt_tsft:.2f}s, "
            f"t_end={b['ends'][-1]*opt_tsft:.2f}s, "
            f"best_slope={float(b['best_slope'][0]):.6e}, "
            f"best_mass={float(b['best_mass'][0]):.6e}, "
            f"best_nmse={float(b['best_nmse'][0]):.6e}"
        )
    # if noise==False:
    #    plot_track_with_best_block_windows(track, best_blocks, picture=str(PLOTS_CANDIDATE_DIR / "signal" / f"track_with_best_blocks_mc-{mchirp:.0e}_dl-{distance_str}_pack-{pack}.png"), tsft=opt_tsft)
    # else:
    #    plot_track_with_best_block_windows(track, best_blocks, picture=str(PLOTS_CANDIDATE_DIR / "noise" / f"track_with_best_blocks_noise_pack-{pack}.png"), tsft=opt_tsft)
    # plt.close()

    # ------- Ejemplo Fourth Check
    optimal_block = _select_optimal_block(best_blocks, opt_tsft)

    if optimal_block is None:
        print("No se pudo seleccionar un bloque óptimo válido (SSE no finito o sin datos). Fin de ejecución.")
        return None, nmse_eval, opt_nsigma, extended_optimal_block

    extended_optimal_block = expansion_block(
        optimal_block,
        track,
        tsft=opt_tsft,
        nmse_nref=nmse_nref,
        nmse_len_alpha=nmse_len_alpha,
    )
    if extended_optimal_block is None:
        print("No fue posible expandir el bloque óptimo con los criterios actuales. Fin de ejecución.")
        return None, nmse_eval, opt_nsigma, extended_optimal_block

    print(
        f"Bloque óptimo extendido: t_start={extended_optimal_block['block_start']*opt_tsft:.2f}s, "
        f"t_end={extended_optimal_block['block_end']*opt_tsft:.2f}s, "
        f"final_slope={float(extended_optimal_block.get('final_slope', extended_optimal_block['best_slope'][0])):.6e}, "
        f"final_mass={float(extended_optimal_block.get('final_mass', extended_optimal_block['best_mass'][0])):.6e}, "
        f"final_nmse_raw={float(extended_optimal_block.get('final_nmse_raw', np.min(np.asarray(extended_optimal_block.get('best_nmse', [np.inf]), dtype=float)))):.6e}, "
        f"final_nmse_pen={float(extended_optimal_block.get('final_nmse', np.min(np.asarray(extended_optimal_block.get('best_nmse', [np.inf]), dtype=float)))):.6e}"
    )
    _plot_optimal_block(track, extended_optimal_block, noise, pack, opt_tsft, mchirp, distance_str)
    
    # ------- Ejemplo Fifth Check
    output_txt = _candidate_output_path(noise=noise, pack=pack)

    nmse_eval = float(extended_optimal_block.get("final_nmse", np.inf))
    if not np.isfinite(nmse_eval):
        nmse_values_ext = np.asarray(extended_optimal_block.get("best_nmse", []), dtype=float)
        nmse_eval = float(np.min(nmse_values_ext)) if nmse_values_ext.size > 0 else np.inf
    nmse_raw_eval = float(extended_optimal_block.get("final_nmse_raw", nmse_eval))
    n_points_ext = int(extended_optimal_block.get("final_n_points", 0))
    len_factor_ext = float(extended_optimal_block.get("final_len_factor", 1.0))

    slope_eval = float(extended_optimal_block.get("final_slope", extended_optimal_block["best_slope"][0]))
    mass_eval = float(extended_optimal_block.get("final_mass", extended_optimal_block["best_mass"][0]))

    print(
        f"Fifth check NMSE: nmse_raw={nmse_raw_eval:.6e}, nmse_pen={nmse_eval:.6e}, "
        f"nsigma_th_line={_candidate_threshold_nsigma(nmse_eval):.6e}, "
        f"n_points_fit={n_points_ext}, len_factor={len_factor_ext:.3f}"
    )

    nsigma_candidate_th = _candidate_threshold_nsigma(nmse_eval)
    candidate_passed = opt_nsigma >= nsigma_candidate_th
    _write_candidate_report(
        output_txt=output_txt,
        is_noise=noise,
        passed=candidate_passed,
        pack=pack,
        mchirp=mchirp,
        distance_str=distance_str,
        block=extended_optimal_block,
        opt_tsft=opt_tsft,
        slope_eval=slope_eval,
        mass_eval=mass_eval,
        nmse_raw_eval=nmse_raw_eval,
        nmse_eval=nmse_eval,
        nmse_candidate_th=nsigma_candidate_th,
    )
    if candidate_passed:
        print(
            "El bloque extendido cumple el umbral lineal "
            f"opt_nsigma >= {LINEAR_THRESHOLD_SLOPE:.6f} * nmse + {LINEAR_THRESHOLD_INTERCEPT:.6f} "
            f"(nsigma_th={nsigma_candidate_th:.6e}). Se ha guardado como candidato válido en {output_txt}."
        )
    else:
        print(
            "El bloque extendido no cumple el umbral lineal "
            f"opt_nsigma >= {LINEAR_THRESHOLD_SLOPE:.6f} * nmse + {LINEAR_THRESHOLD_INTERCEPT:.6f} "
            f"(nsigma_th={nsigma_candidate_th:.6e}). No se considera un candidato válido."
        )
        return None, nmse_eval, opt_nsigma, extended_optimal_block
    return True, nmse_eval, opt_nsigma, extended_optimal_block

def search_candidates_fit(mchirp, distance, pack, noise): # Up to now, is not th best method, look at comparison between both metric plots.
    nmse_eval = None
    extended_optimal_block = None
    distance_str = f"{distance:.3f}".replace('.', '_') if distance is not None else "noise"
    nmse_candidate_th = 0.05
    nmse_nref = 64
    nmse_len_alpha = 1.0

    # ------- Ejemplo First Check 
    n_windows = 1
    tsft_list = [2,4,8,16,32,64,128]

    opt_tsft, opt_nsigma, found_above_threshold = first_power_check(mchirp=mchirp, distance_str=distance_str, noise=noise, tsft_list=tsft_list, pack=pack)
    print(f"Results: opt_tsft={opt_tsft} with ratio signal/noise={opt_nsigma:.4f}, found_above_threshold={found_above_threshold}")

    # ------- Ejemplo Second+Third Check (fit por ventana)
    n_windows = 8
    path_freq_track = build_freq_track_path(
        tsft=opt_tsft, pack=pack, noise=noise, mchirp=mchirp, distance_str=distance_str
    )
    
    track = load_track_txt(path_freq_track)
    starts, ends = split_track_windows(track, n_windows=n_windows)
    candidate_blocks = [(int(s), int(e), 1.0) for s, e in zip(starts, ends)]

    print(f"Step 2+3: fit en {len(candidate_blocks)} ventanas (tsft={opt_tsft}s)")
    for s, e, _ in candidate_blocks:
        print(f"Ventana candidata [{s*opt_tsft}:{e*opt_tsft}]")

    best_blocks, significant_block= fit_slope_candidate_blocks(
                track=track,
                tsft=opt_tsft,
                candidate_blocks=candidate_blocks,
                n_windows_per_block=1,
                blocks_in_time=False,
                )

    nmse_windows = []
    best_nmse_seen = np.inf
    for b in best_blocks:
        nmse_values = np.asarray(b.get("best_nmse", []), dtype=float)
        print(nmse_values)
        if nmse_values.size == 0:
            continue
        nmse_win = float(np.min(nmse_values))
        if not np.isfinite(nmse_win):
            continue
        if nmse_win < best_nmse_seen:
            best_nmse_seen = nmse_win
        if nmse_win > 5e-2:
            continue
        nmse_windows.append((nmse_win, b))
        print(
            f"Ventana fit: t_start={b['block_start']*opt_tsft:.2f}s, "
            f"t_end={b['block_end']*opt_tsft:.2f}s, "
            f"best_slope={float(b['best_slope'][0]):.6e}, "
            f"best_mass={float(b['best_mass'][0]):.6e}, "
            f"best_nmse={nmse_win:.6e}"
        )

    if not nmse_windows:
        nmse_windows = []
        print("! No se encontraron bloques viables que ajusten el modelo en ventana de t=4096 seg, probando ventana  t=1024seg ")
        best_blocks, significant_block= fit_slope_candidate_blocks(
                track=track,
                tsft=opt_tsft,
                candidate_blocks=candidate_blocks,
                n_windows_per_block=4,
                blocks_in_time=False,
                )
        for b in best_blocks:
            nmse_values = np.asarray(b.get("best_nmse", []), dtype=float)
            print(nmse_values)
            if nmse_values.size == 0:
                continue
            nmse_win = float(np.min(nmse_values))
            if not np.isfinite(nmse_win):
                continue
            if nmse_win < best_nmse_seen:
                best_nmse_seen = nmse_win
            if nmse_win > 5e-2:
                continue
            nmse_windows.append((nmse_win, b))
            print(
                f"Ventana fit: t_start={b['block_start']*opt_tsft:.2f}s, "
                f"t_end={b['block_end']*opt_tsft:.2f}s, "
                f"best_slope={float(b['best_slope'][0]):.6e}, "
                f"best_mass={float(b['best_mass'][0]):.6e}, "
                f"best_nmse={nmse_win:.6e}"
            )

    if not best_blocks:
        print("Ninguna ventana pasó el Step 2+3 fit (probablemente por masa negativa o ajuste inválido). Fin de ejecución.")
        nmse_out = best_nmse_seen if np.isfinite(best_nmse_seen) else None
        return None, nmse_out, opt_nsigma, extended_optimal_block

    if not nmse_windows:
        print("No hubo ventanas con NMSE finito tras el fit en Step 2+3. Fin de ejecución.")
        nmse_out = best_nmse_seen if np.isfinite(best_nmse_seen) else None
        return None, nmse_out, opt_nsigma, extended_optimal_block

    nmse_windows.sort(key=lambda x: x[0])
    best_blocks = [nmse_windows[0][1]]
    print(
        f"Step 2+3 seleccionado -> "
        f"[{best_blocks[0]['block_start']*opt_tsft:.2f}:{best_blocks[0]['block_end']*opt_tsft:.2f}] s "
        f"con NMSE={nmse_windows[0][0]:.6e}"
    )
    if noise==False:
        plot_track_with_best_block_windows(
            track,
            best_blocks,
            picture=str(PLOTS_CANDIDATE_DIR / "signal" / f"track_with_best_blocks_mc-{mchirp:.0e}_dl-{distance_str}_pack-{pack}.png"),
            tsft=opt_tsft,
        )
    else:
        plot_track_with_best_block_windows(
            track,
            best_blocks,
            picture=str(PLOTS_CANDIDATE_DIR / "noise" / f"track_with_best_blocks_noise_pack-{pack}.png"),
            tsft=opt_tsft,
        )
    plt.close()

    # ------- Ejemplo Fourth Check
    optimal_block = _select_optimal_block(best_blocks, opt_tsft)

    if optimal_block is None:
        print("No se pudo seleccionar un bloque óptimo válido (SSE no finito o sin datos). Fin de ejecución.")
        return None, nmse_eval, opt_nsigma, extended_optimal_block

    extended_optimal_block = expansion_block(
        optimal_block,
        track,
        tsft=opt_tsft,
        nmse_nref=nmse_nref,
        nmse_len_alpha=nmse_len_alpha,
    )
    if extended_optimal_block is None:
        print("No fue posible expandir el bloque óptimo con los criterios actuales. Fin de ejecución.")
        return None, nmse_eval, opt_nsigma, extended_optimal_block

    print(
        f"Bloque óptimo extendido: t_start={extended_optimal_block['block_start']*opt_tsft:.2f}s, "
        f"t_end={extended_optimal_block['block_end']*opt_tsft:.2f}s, "
        f"final_slope={float(extended_optimal_block.get('final_slope', extended_optimal_block['best_slope'][0])):.6e}, "
        f"final_mass={float(extended_optimal_block.get('final_mass', extended_optimal_block['best_mass'][0])):.6e}, "
        f"final_nmse_raw={float(extended_optimal_block.get('final_nmse_raw', np.min(np.asarray(extended_optimal_block.get('best_nmse', [np.inf]), dtype=float)))):.6e}, "
        f"final_nmse_pen={float(extended_optimal_block.get('final_nmse', np.min(np.asarray(extended_optimal_block.get('best_nmse', [np.inf]), dtype=float)))):.6e}"
    )
    _plot_optimal_block(track, extended_optimal_block, noise, pack, opt_tsft, mchirp, distance_str)
    
    # ------- Ejemplo Fifth Check
    output_txt = _candidate_output_path(noise=noise, pack=pack)

    nmse_eval = float(extended_optimal_block.get("final_nmse", np.inf))
    if not np.isfinite(nmse_eval):
        nmse_values_ext = np.asarray(extended_optimal_block.get("best_nmse", []), dtype=float)
        nmse_eval = float(np.min(nmse_values_ext)) if nmse_values_ext.size > 0 else np.inf
    nmse_raw_eval = float(extended_optimal_block.get("final_nmse_raw", nmse_eval))
    n_points_ext = int(extended_optimal_block.get("final_n_points", 0))
    len_factor_ext = float(extended_optimal_block.get("final_len_factor", 1.0))

    slope_eval = float(extended_optimal_block.get("final_slope", extended_optimal_block["best_slope"][0]))
    mass_eval = float(extended_optimal_block.get("final_mass", extended_optimal_block["best_mass"][0]))

    print(
        f"Fifth check NMSE: nmse_raw={nmse_raw_eval:.6e}, nmse_pen={nmse_eval:.6e}, "
        f"nsigma_th_line={_candidate_threshold_nsigma(nmse_eval):.6e}, "
        f"n_points_fit={n_points_ext}, len_factor={len_factor_ext:.3f}"
    )

    nsigma_candidate_th = _candidate_threshold_nsigma(nmse_eval)
    candidate_passed = opt_nsigma >= nsigma_candidate_th
    _write_candidate_report(
        output_txt=output_txt,
        is_noise=noise,
        passed=candidate_passed,
        pack=pack,
        mchirp=mchirp,
        distance_str=distance_str,
        block=extended_optimal_block,
        opt_tsft=opt_tsft,
        slope_eval=slope_eval,
        mass_eval=mass_eval,
        nmse_raw_eval=nmse_raw_eval,
        nmse_eval=nmse_eval,
        nmse_candidate_th=nsigma_candidate_th,
    )
    if candidate_passed:
        print(
            "El bloque extendido cumple el umbral lineal "
            f"opt_nsigma >= {LINEAR_THRESHOLD_SLOPE:.6f} * nmse + {LINEAR_THRESHOLD_INTERCEPT:.6f} "
            f"(nsigma_th={nsigma_candidate_th:.6e}). Se ha guardado como candidato válido en {output_txt}."
        )
    else:
        print(
            "El bloque extendido no cumple el umbral lineal "
            f"opt_nsigma >= {LINEAR_THRESHOLD_SLOPE:.6f} * nmse + {LINEAR_THRESHOLD_INTERCEPT:.6f} "
            f"(nsigma_th={nsigma_candidate_th:.6e}). No se considera un candidato válido."
        )
        return None, nmse_eval, opt_nsigma, extended_optimal_block
    return True, nmse_eval, opt_nsigma, extended_optimal_block

if __name__ == "__main__":
    # ------- Ejemplo noise metrics
    n_windows = 1
    tsft_list = [2, 3, 4, 5, 6, 9, 12, 15, 21, 29, 39, 54, 74, 101, 132, 181, 248, 340]
    packs_list = list(range(1, 109))

    path_power_noise = power_noise_track(n_windows=n_windows, tsft_list=tsft_list, packs_list=packs_list)
    print(f"Results in: {path_power_noise}")
