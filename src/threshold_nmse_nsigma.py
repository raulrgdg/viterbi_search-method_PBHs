import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext, SymmetricalLogLocator

from pipeline_paths import OUTPUTS_PLOTS_DIR, OUTPUTS_REPORTS_DIR, ensure_dir

REPORTS_DIR = ensure_dir(OUTPUTS_REPORTS_DIR)
PLOTS_DIR = ensure_dir(OUTPUTS_PLOTS_DIR)

# Edita estos nombres a mano antes de ejecutar el script.
NOISE_CSV = REPORTS_DIR / "search_results_noise-01_04.csv"
SIGNAL_CSV = REPORTS_DIR / "search_results_signal-pack5_log_log.csv"
COMPUTE_THRESHOLDS = True
PFAR_TARGET = 0.03
TOTAL_NOISE_CASES = 109
SLOPE_GRID = np.linspace(0.0, 400.0, 8000)
LINEAR_THRESHOLD_SLOPE = 255.5
LINEAR_THRESHOLD_INTERCEPT = -0.671286
OUTPUT_PLOT = PLOTS_DIR / f"threshold_roc_nmse_nsigma-{PFAR_TARGET}-pack5-1.png"

NMSE_COLUMN = "nmse"
NSIGMA_COLUMN = "nsigma"



def _load_metrics(csv_path):
    """Load valid nmse/nsigma pairs from a search-results CSV."""
    df = pd.read_csv(csv_path)
    required_columns = {NMSE_COLUMN, NSIGMA_COLUMN}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Faltan columnas en {csv_path}: {missing_list}")

    data = df[[NMSE_COLUMN, NSIGMA_COLUMN]].apply(pd.to_numeric, errors="coerce").dropna()
    data = data[np.isfinite(data[NMSE_COLUMN]) & np.isfinite(data[NSIGMA_COLUMN])]
    if data.empty:
        raise ValueError(f"No hay pares válidos de nmse/nsigma en {csv_path}")

    return data


def _line_from_pfar(noise_df, signal_df, pfar_target, slope_grid):
    """Find the line y = m*x + b that maximizes signal recovery at fixed PFAR."""
    x_noise = noise_df[NMSE_COLUMN].to_numpy()
    y_noise = noise_df[NSIGMA_COLUMN].to_numpy()
    x_signal = signal_df[NMSE_COLUMN].to_numpy()
    y_signal = signal_df[NSIGMA_COLUMN].to_numpy()

    best_result = None
    max_false_alarms = int(np.floor(pfar_target * TOTAL_NOISE_CASES))

    for slope in slope_grid:
        noise_scores = y_noise - slope * x_noise
        order = np.argsort(noise_scores)
        sorted_scores = noise_scores[order]

        if max_false_alarms <= 0:
            intercept = float(sorted_scores[-1] + np.finfo(float).eps)
            noise_selected = np.zeros_like(noise_scores, dtype=bool)
        elif max_false_alarms >= len(sorted_scores):
            intercept = float(sorted_scores[0] - np.finfo(float).eps)
            noise_selected = np.ones_like(noise_scores, dtype=bool)
        else:
            intercept = float(sorted_scores[-max_false_alarms])
            noise_selected = np.zeros_like(noise_scores, dtype=bool)
            noise_selected[order[-max_false_alarms:]] = True

        signal_scores = y_signal - slope * x_signal
        signal_selected = signal_scores >= intercept

        selected_noise_count = int(np.count_nonzero(noise_selected))
        pfar = selected_noise_count / float(TOTAL_NOISE_CASES)
        tpr = float(np.mean(signal_selected))
        selected_signal_count = int(np.count_nonzero(signal_selected))

        result = {
            "slope": float(slope),
            "intercept": intercept,
            "pfar": pfar,
            "tpr": tpr,
            "selected_signal_count": selected_signal_count,
            "selected_noise_count": selected_noise_count,
            "noise_mask": noise_selected,
            "signal_mask": signal_selected,
        }

        if best_result is None:
            best_result = result
            continue

        if result["tpr"] > best_result["tpr"]:
            best_result = result
            continue

        if np.isclose(result["tpr"], best_result["tpr"]) and result["pfar"] < best_result["pfar"]:
            best_result = result
            continue

        if (
            np.isclose(result["tpr"], best_result["tpr"])
            and np.isclose(result["pfar"], best_result["pfar"])
            and result["slope"] < best_result["slope"]
        ):
            best_result = result

    return best_result


def _result_from_fixed_line(noise_df, signal_df, slope, intercept):
    """Evaluate a fixed linear threshold and build the plotting masks."""
    x_noise = noise_df[NMSE_COLUMN].to_numpy()
    y_noise = noise_df[NSIGMA_COLUMN].to_numpy()
    x_signal = signal_df[NMSE_COLUMN].to_numpy()
    y_signal = signal_df[NSIGMA_COLUMN].to_numpy()

    noise_mask = (y_noise - slope * x_noise) >= intercept
    signal_mask = (y_signal - slope * x_signal) >= intercept

    selected_noise_count = int(np.count_nonzero(noise_mask))
    selected_signal_count = int(np.count_nonzero(signal_mask))

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "pfar": selected_noise_count / float(TOTAL_NOISE_CASES),
        "tpr": float(np.mean(signal_mask)),
        "selected_signal_count": selected_signal_count,
        "selected_noise_count": selected_noise_count,
        "noise_mask": noise_mask,
        "signal_mask": signal_mask,
    }


def _plot_threshold(noise_df, signal_df, result, output_plot):
    """Plot noise/signal points and the selected linear threshold."""
    fig, ax = plt.subplots(figsize=(8, 6))
    noise_mask = result["noise_mask"]
    signal_mask = result["signal_mask"]

    ax.scatter(
        noise_df[NMSE_COLUMN],
        noise_df[NSIGMA_COLUMN],
        s=20,
        alpha=0.35,
        color="tab:blue",
        label="TN",
    )
    ax.scatter(
        signal_df[NMSE_COLUMN],
        signal_df[NSIGMA_COLUMN],
        s=20,
        alpha=0.25,
        color="tab:orange",
        label="FN",
    )
    ax.scatter(
        noise_df.loc[noise_mask, NMSE_COLUMN],
        noise_df.loc[noise_mask, NSIGMA_COLUMN],
        s=60,
        alpha=0.95,
        color="crimson",
        marker="x",
        label="FP",
    )
    ax.scatter(
        signal_df.loc[signal_mask, NMSE_COLUMN],
        signal_df.loc[signal_mask, NSIGMA_COLUMN],
        s=42,
        alpha=0.85,
        color="goldenrod",
        edgecolors="black",
        linewidths=0.4,
        label="TP",
    )

    x_min = 1e-7
    x_max = 1e2
    x_line = np.logspace(np.log10(x_min), np.log10(x_max), 800)
    y_line = result["slope"] * x_line + result["intercept"]
    y_min = -5
    y_max = 81e2

    ax.plot(
        x_line,
        y_line,
        color="black",
        linewidth=2.0,
    )

    ax.set_xlabel(r"$\mathrm{NMSE}$")
    ax.set_ylabel(r"$n_{\sigma}$")
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs='auto'))
    ax.set_title(r"Threshold -- $\mathrm{NMSE}$-$n_{\sigma}$ map -- FAR < 3%")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=10, frameon=True)
    fig.tight_layout()
    fig.savefig(output_plot, dpi=150)
    plt.close(fig)


def main():
    """Build a 2D threshold line at fixed PFAR using nmse and nsigma."""
    noise_df = _load_metrics(NOISE_CSV)
    signal_df = _load_metrics(SIGNAL_CSV)

    if COMPUTE_THRESHOLDS:
        if not 0.0 < PFAR_TARGET < 1.0:
            raise ValueError(f"PFAR_TARGET debe estar entre 0 y 1, recibido: {PFAR_TARGET}")

        result = _line_from_pfar(
            noise_df=noise_df,
            signal_df=signal_df,
            pfar_target=PFAR_TARGET,
            slope_grid=SLOPE_GRID,
        )
    else:
        result = _result_from_fixed_line(
            noise_df=noise_df,
            signal_df=signal_df,
            slope=LINEAR_THRESHOLD_SLOPE,
            intercept=LINEAR_THRESHOLD_INTERCEPT,
        )

    _plot_threshold(noise_df, signal_df, result, OUTPUT_PLOT)

    print(f"Noise file: {NOISE_CSV}")
    print(f"Signal file: {SIGNAL_CSV}")
    print(f"Número de muestras noise válidas: {len(noise_df)}")
    print(f"Número total de casos noise para PFAR: {TOTAL_NOISE_CASES}")
    print(f"Número de muestras signal válidas: {len(signal_df)}")
    if COMPUTE_THRESHOLDS:
        print(f"PFAR objetivo: {PFAR_TARGET:.4f}")
    else:
        print("PFAR objetivo: no calculado, usando threshold fijo")
    print(
        "Recta umbral: "
        f"nsigma = {result['slope']:.6f} * nmse + {result['intercept']:.6f}"
    )
    print(f"PFAR medido sobre noise: {result['pfar']:.6f}")
    print(f"False alarms seleccionadas: {result['selected_noise_count']}")
    print(
        "Noise cases que saltan como falsos positivos con esta recta: "
        f"{result['selected_noise_count']}"
    )
    print(f"Signals escogidas por el threshold: {result['selected_signal_count']}")
    print(f"Eficiencia sobre signal: {result['tpr']:.6f}")
    print(f"Plot guardado en: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
