#!/usr/bin/env python3

import argparse
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHARDS_DIR = PROJECT_ROOT / "outputs" / "tmp" / "search_shards"
SHARD_PATTERN = re.compile(r"^(?P<stem>.+)\.job-(?P<job>\d+)-of-(?P<total>\d+)\.(?P<suffix>[^.]+)$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Comprueba si existen todos los .done esperados en outputs/tmp/search_shards/."
    )
    parser.add_argument("n_jobs", type=int, help="Numero total de jobs esperados.")
    return parser.parse_args()


def build_expected_done_paths(base_name: str, n_jobs: int):
    stem, suffix = base_name.rsplit(".", 1)
    return [
        SHARDS_DIR / f"{stem}.job-{job_id:03d}-of-{n_jobs:03d}.{suffix}.done"
        for job_id in range(n_jobs)
    ]


def discover_base_names(n_jobs: int):
    base_names = set()
    if not SHARDS_DIR.exists():
        return base_names

    for path in SHARDS_DIR.iterdir():
        candidate_name = path.name[:-5] if path.name.endswith(".done") else path.name
        match = SHARD_PATTERN.match(candidate_name)
        if not match:
            continue
        if int(match.group("total")) != n_jobs:
            continue
        base_names.add(f"{match.group('stem')}.{match.group('suffix')}")

    return sorted(base_names)


def main():
    args = parse_args()
    if args.n_jobs <= 0:
        raise SystemExit("n_jobs debe ser mayor que 0")

    base_names = discover_base_names(args.n_jobs)
    if not base_names:
        print(f"No se encontraron shards para n_jobs={args.n_jobs} en {SHARDS_DIR}")
        return

    missing = []
    for base_name in base_names:
        for done_path in build_expected_done_paths(base_name, args.n_jobs):
            if not done_path.exists():
                missing.append(done_path)

    if missing:
        print("Faltan los siguientes .done:")
        for path in missing:
            print(path)
        raise SystemExit(1)

    print(f"Estan todos los .done necesarios para n_jobs={args.n_jobs}.")


if __name__ == "__main__":
    main()
