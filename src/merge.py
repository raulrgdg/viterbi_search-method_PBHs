from pathlib import Path
import csv

tmp_dir = Path("/home/raul.rodriguez/viterbi/codes/final_pipeline/viterbi_search-method_PBHs/outputs/tmp/search_shards")
out_csv = tmp_dir.parent.parent / "reports" / "search_results_signal.csv"

shards = sorted(tmp_dir.glob("search_results_signal.job-*-of-500.csv"))
print(f"Encontrados {len(shards)} shards")

if not shards:
    raise SystemExit("No se encontraron shards")

with shards[0].open(newline="", encoding="utf-8") as f:
    fieldnames = csv.DictReader(f).fieldnames

with out_csv.open("w", newline="", encoding="utf-8") as out:
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()

    for shard in shards:
        print(f"Merging {shard.name}")
        with shard.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                writer.writerow(row)

print(f"Merge completo en: {out_csv}")
