import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.common.paths import OUTPUTS_PLOTS_DIR, OUTPUTS_REPORTS_DIR, ensure_dir

REPORTS_DIR = ensure_dir(OUTPUTS_REPORTS_DIR)
PLOTS_DIR = ensure_dir(OUTPUTS_PLOTS_DIR)
NOISE_CSV = REPORTS_DIR / "noise_significant_block-flag.csv"
SIGNAL_CSV = REPORTS_DIR / "signal_significant_block-flag.csv"
OUTPUT_PLOT = PLOTS_DIR / "threshold_powers-noise_signal.png"


def main():
    df_noise = pd.read_csv(NOISE_CSV)
    df_signal = pd.read_csv(SIGNAL_CSV)
    z_column = "MAD"

    z_signal = df_signal[z_column].dropna().to_numpy()
    z_noise = df_noise[z_column].dropna().to_numpy()
    z_th = np.percentile(z_noise, 87.5)

    print(f"Número de muestras noise: {len(z_noise)}")
    print(f"Threshold P99 para z: {z_th:.6f}")

    false_alarm_rate = np.mean(z_noise > z_th)
    detection_efficiency = np.mean(z_signal > z_th)

    for p in [80, 82, 85, 87.5, 90, 99, 99.5]:
        print(f"P{p:>4}: {np.percentile(z_noise, p):.6f}")

    print(f"Threshold P99 (noise): {z_th:.4f}")
    print(f"False alarm rate: {false_alarm_rate:.4f}")
    print(f"Detection efficiency: {detection_efficiency:.4f}")

    plt.figure(figsize=(7, 5))
    plt.hist(z_noise, bins=30, alpha=0.6, label="Noise", density=True)
    plt.hist(z_signal, bins=30, alpha=0.6, label="Signal+Noise", density=True)
    plt.axvline(z_th, color="red", linestyle="--", label=f"P99 threshold = {z_th:.2f}")
    plt.xlabel("z statistic")
    plt.xscale("log")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT)


if __name__ == "__main__":
    main()
