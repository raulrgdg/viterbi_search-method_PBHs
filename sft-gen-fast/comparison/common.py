"""Shared helpers of the MakeSFTs vs fast_sft comparison scripts."""

import csv
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from fast_sft import sft_matrix  # noqa: E402

from pbh_viterbi.config import FRAME_LENGTH, NUM_FRAMES  # noqa: E402
from pbh_viterbi.sft.load import load_sft_matrix  # noqa: E402
from pbh_viterbi.sft.make_sfts import make_sfts  # noqa: E402
from pbh_viterbi.sft.maps import (  # noqa: E402
    build_remap_geometry,
    n_frequency_bins,
    normalized_power,
    remap_to_fm83,
    viterbi_track,
)

# Old-repo copy of the 106 downloaded O3b packs (identical to pipeline_v1.0/data/o3).
O3_DIR = HERE.parents[2] / "viterbi_search-method_PBHs" / "data" / "raw" / "O3_data"
RESULTS = HERE / "results"
LOGS = HERE / "logs"
PACK_SEED = 20261008  # seed of the random pack selection


def random_packs(n, available, exclude=(), seed=PACK_SEED, stream=0):
    """Reproducible random choice of ``n`` packs among ``available``."""
    pool = sorted(set(available) - set(exclude))
    rng = np.random.default_rng([seed, stream])
    return sorted(int(p) for p in rng.choice(pool, size=n, replace=False))


def available_packs(o3_dir=O3_DIR):
    return sorted(int(p.name.replace("O3b-pack", "")) for p in Path(o3_dir).glob("O3b-pack*")
                  if len(list(p.glob("*.gwf"))) == NUM_FRAMES)


def concat_strain(segments):
    """Contiguous float64 strain and sample rate of a list of pycbc TimeSeries."""
    return np.concatenate([np.asarray(s, dtype=np.float64) for s in segments]), 1.0 / segments[0].delta_t


def products_from_sfts(sfts, tsft, fmin):
    """Remapped map and Viterbi track of an SFT matrix (n_bins, n_sft), as in process_tsft."""
    geometry = build_remap_geometry(tsft, fmin, sfts.shape[0])
    power = remap_to_fm83(normalized_power(sfts), geometry["x_inc"], geometry["x_new"])
    track = viterbi_track(power)
    return {"tsft": tsft, "track_index": track, "track_freq": np.asarray(geometry["x_new"][track], dtype=float),
            "power": np.asarray(power, dtype=float)}


def makesfts_matrix(t_start, t_end, tsft, framecache, threads, fmin, band):
    nbins = n_frequency_bins(tsft, band)
    n_sft = int(NUM_FRAMES * FRAME_LENGTH / tsft)
    with tempfile.TemporaryDirectory(prefix=f"cmp-sft-tsft{tsft}-") as sft_dir:
        make_sfts(t_start, t_end, tsft, framecache, sft_dir, threads, fmin=fmin, band=band)
        return load_sft_matrix(sft_dir, t_start, tsft, nbins, n_sft)


def compare_chunk(t_start, t_end, framecache, strain, sample_rate, tsft_values, threads, fmin, fmax, label):
    """Run both SFT methods on the same chunk for every Tsft.

    Returns (rows, products_makesfts, products_fast); one row per Tsft with the
    SFT, map and track differences and the time of each method.
    """
    band = fmax - fmin
    rows, ref_products, fast_products = [], [], []
    for tsft in tsft_values:
        t0 = time.perf_counter()
        ref_sfts = makesfts_matrix(t_start, t_end, tsft, framecache, threads, fmin, band)
        t1 = time.perf_counter()
        fast_sfts = sft_matrix(strain, sample_rate, tsft, fmin, band)
        t2 = time.perf_counter()
        ref, fast = products_from_sfts(ref_sfts, tsft, fmin), products_from_sfts(fast_sfts, tsft, fmin)
        ref_products.append(ref)
        fast_products.append(fast)

        same_shape = ref_sfts.shape == fast_sfts.shape
        sft_rel = (np.abs(ref_sfts - fast_sfts) / np.abs(ref_sfts)) if same_shape else np.array([np.inf])
        map_ok = ref["power"].shape == fast["power"].shape
        map_rel = (np.abs(ref["power"] - fast["power"]) / np.abs(ref["power"])) if map_ok else np.array([np.inf])
        track_diff = (int(np.sum(ref["track_index"] != fast["track_index"]))
                      if ref["track_index"].shape == fast["track_index"].shape else -1)
        row = {
            **label,
            "tsft": tsft,
            "n_sft": ref_sfts.shape[1],
            "n_bins": ref_sfts.shape[0],
            "same_shape": same_shape,
            "sft_identical": same_shape and bool(np.array_equal(ref_sfts, fast_sfts)),
            "sft_max_rel_diff": float(np.nanmax(sft_rel)),
            "map_identical": map_ok and bool(np.array_equal(ref["power"], fast["power"], equal_nan=True)),
            "map_max_rel_diff": float(np.nanmax(map_rel)),
            "map_pixels_different": int(np.sum(ref["power"] != fast["power"]) -
                                        np.sum(np.isnan(ref["power"]) & np.isnan(fast["power"]))) if map_ok else -1,
            "track_identical": track_diff == 0,
            "track_steps_different": track_diff,
            "time_makesfts_s": round(t1 - t0, 2),
            "time_fast_s": round(t2 - t1, 2),
        }
        rows.append(row)
        print(", ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    return rows, ref_products, fast_products


def append_csv(rows, path):
    """Append rows to a CSV, writing the header if the file is new."""
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if new:
            writer.writeheader()
        writer.writerows(rows)


def same_value(a, b):
    """Equality of two CSV-like values, treating empty/None/NaN as equal."""
    def norm(v):
        if v is None or v == "" or (isinstance(v, float) and not np.isfinite(v)):
            return None
        return v
    a, b = norm(a), norm(b)
    if a is None or b is None:
        return a is None and b is None
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)
