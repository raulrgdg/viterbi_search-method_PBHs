#!/usr/bin/env python3
"""Collect the per-Tsft products of the Tsft-selection study.

Merges any shard left unmerged by the jobs, checks that every chunk has all the
Tsft of the "union" list and writes two tables to <results>/collected/:

  per_tsft_noise.csv     one row per noise pack and Tsft
  per_tsft_injected.csv  one row per injection and Tsft, joined with the
                         injection parameters (mchirp, distance, sky, t_to_merger)

    python studies/tsft_selection/collect.py [--results results/tsft_selection] [--n-jobs 40]

Exits with an error listing the incomplete chunks (use --allow-incomplete to
write the tables anyway).
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "tsft_selection")
    parser.add_argument("--n-jobs", type=int, default=40, help="Jobs per injection pack used at submission.")
    parser.add_argument("--n-noise-jobs", type=int, default=141)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    os.environ["PBH_VITERBI_RESULTS"] = str(args.results.resolve())  # before importing pbh_viterbi.paths

    from pbh_viterbi.config import TSFT_LISTS
    from pbh_viterbi.o3.packs import parse_pack_list
    from pbh_viterbi.results import FIELDNAMES, PER_TSFT_FIELDNAMES, merge_shards_if_ready

    search = args.results / "search"
    shards = args.results / "tmp" / "shards"
    # Merge what the jobs left unmerged (harmless if already merged).
    names = {p.name.split(".job-")[0] + ".csv" for p in shards.glob("*.job-*.csv")} if shards.exists() else set()
    for name in sorted(names):
        n_jobs = args.n_noise_jobs if "noise" in name else args.n_jobs
        merge_shards_if_ready(name, n_jobs, PER_TSFT_FIELDNAMES if name.startswith("per_tsft") else FIELDNAMES)
    leftover = sorted({p.name.split(".job-")[0] for p in shards.glob("*.job-*.csv")}) if shards.exists() else []

    n_tsft = len(TSFT_LISTS["union"])
    problems = [f"unmerged shards: {name}" for name in leftover]
    out = args.results / "collected"
    out.mkdir(parents=True, exist_ok=True)

    noise_path = search / "per_tsft_noise.csv"
    if noise_path.exists():
        noise = pd.read_csv(noise_path)
        counts = noise.groupby("pack").size()
        missing = sorted(set(parse_pack_list("all")) - set(counts.index))
        problems += [f"noise: {len(missing)} packs missing {missing[:10]}"] if missing else []
        problems += [f"noise pack {p}: {n} Tsft" for p, n in counts.items() if n != n_tsft]
        noise.to_csv(out / "per_tsft_noise.csv", index=False)
        print(f"noise: {counts.size} packs, {len(noise)} rows")
    else:
        problems.append("noise: per_tsft_noise.csv missing")

    frames = []
    for path in sorted(search.glob("per_tsft_injected_pack-*.csv")):
        pack = int(path.stem.split("pack-")[1])
        per_tsft = pd.read_csv(path)
        params = pd.read_csv(search / f"search_results_injected_pack-{pack}.csv")
        params = params[["pack", "signal_id", "mchirp", "distance", "ra", "dec", "pol", "inclination", "t_to_merger"]]
        counts = per_tsft.groupby("signal_id").size()
        problems += [f"pack {pack}: {counts.size} injections (expected 200)"] if counts.size != 200 else []
        problems += [f"pack {pack} signal {s}: {n} Tsft" for s, n in counts.items() if n != n_tsft]
        problems += [f"pack {pack}: duplicated injections"] if params.duplicated(["pack", "signal_id"]).any() else []
        frames.append(per_tsft.merge(params, on=["pack", "signal_id"], how="left", validate="many_to_one"))
        print(f"injected pack {pack}: {counts.size} injections")
    if frames:
        injected = pd.concat(frames, ignore_index=True)
        injected.to_csv(out / "per_tsft_injected.csv", index=False)
        print(f"injected: {injected.groupby(['pack', 'signal_id']).ngroups} injections, {len(injected)} rows")
    else:
        problems.append("injected: no per_tsft_injected_pack-*.csv")

    if problems:
        print("INCOMPLETE:\n  " + "\n  ".join(problems))
        if not args.allow_incomplete:
            sys.exit(1)
    print(f"Tables written to {out}")


if __name__ == "__main__":
    main()
