import csv

from pipeline_paths import OUTPUTS_REPORTS_DIR, ensure_dir

REPORTS_DIR = ensure_dir(OUTPUTS_REPORTS_DIR)
NOISE_CSV = REPORTS_DIR / "search_results_noise_final-big-search-flag_perc-90-new-recorte-total-window-v4.csv"
SIGNAL_CSV = REPORTS_DIR / "search_results_signal_final-big-search-flag_perc-90-new-recorte-total-window-v4.csv"
OUTPUT_CSV = REPORTS_DIR / "metrics.csv"

COL_NMSE = "NMSE"
COL_NSIGMA = "nsigma"
COL_LABEL = "label"  # 0 = noise, 1 = signal


def _load_rows(csv_path, label):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            rows.append(
                {
                    COL_NMSE: row["final_nmse"],
                    COL_NSIGMA: row["opt_nsigma"],
                    COL_LABEL: label,
                }
            )
    return rows


def main():
    rows = []
    rows.extend(_load_rows(NOISE_CSV, 0))
    rows.extend(_load_rows(SIGNAL_CSV, 1))

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=[COL_NMSE, COL_NSIGMA, COL_LABEL])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Guardado en {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
