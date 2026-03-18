import csv
import math
import statistics
from collections import defaultdict

from pipeline_paths import OUTPUTS_REPORTS_DIR, ensure_dir

REPORTS_DIR = ensure_dir(OUTPUTS_REPORTS_DIR)
CSV_PATH = REPORTS_DIR / "power_noise_result_2-test-version2.csv"
OUTPUT_CSV = REPORTS_DIR / "noise_metrics_2-test.csv"


def main():
    """Aggregate mean and standard deviation of noise power per tsft."""
    values_by_tsft = defaultdict(list)

    with open(CSV_PATH, newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            tsft = row["tsft"]
            total_power = float(row["total_power"])
            values_by_tsft[tsft].append(total_power)

    rows = []
    for tsft in sorted(values_by_tsft, key=float):
        values = values_by_tsft[tsft]
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if len(values) > 1 else math.nan
        rows.append(
            {
                "tsft": tsft,
                "mean_total_power": mean_val,
                "std_total_power": std_val,
            }
        )

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(
            outfile,
            fieldnames=["tsft", "mean_total_power", "std_total_power"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Guardado en {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
