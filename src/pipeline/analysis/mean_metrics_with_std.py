import csv
import math
import statistics
from collections import defaultdict

from pipeline.common.paths import OUTPUTS_REPORTS_DIR, ensure_dir

REPORTS_DIR = ensure_dir(OUTPUTS_REPORTS_DIR)
CSV_PATH = REPORTS_DIR / "power_noise_results-con20.csv"
OUTPUT_CSV = REPORTS_DIR / "noise_metrics.csv"


def main():
    values_by_tsft = defaultdict(list)

    with CSV_PATH.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            values_by_tsft[row["tsft"]].append(float(row["total_power"]))

    rows = []
    for tsft in sorted(values_by_tsft, key=float):
        values = values_by_tsft[tsft]
        rows.append(
            {
                "tsft": tsft,
                "mean_total_power": statistics.mean(values),
                "std_total_power": statistics.stdev(values) if len(values) > 1 else math.nan,
            }
        )

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["tsft", "mean_total_power", "std_total_power"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Guardado en {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
