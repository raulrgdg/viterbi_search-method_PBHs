"""Helpers shared by the figure scripts: one injection, its SFTs and plotting style."""

import logging
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _setup  # noqa: F401  (adds src/ to sys.path)
from pbh_viterbi.config import t_to_merger_for_mchirp
from pbh_viterbi.o3.frames import read_pack_frames, write_injected_framecache
from pbh_viterbi.o3.packs import pack_window
from pbh_viterbi.paths import O3_DIR, RESULTS_DIR, o3_pack_dir
from pbh_viterbi.sft.load import load_sft_dir
from pbh_viterbi.sft.make_sfts import make_sfts
from pbh_viterbi.waveform.injection import equal_component_masses, inject_signal

log = logging.getLogger("figures")
FIGURE_WORK_DIR = RESULTS_DIR / "figure_work"


def injected_sfts(name, pack, tsft, mchirp, distance, ra, dec, pol, t_to_merger=None, threads=32,
                  regenerate=False, o3_dir=O3_DIR):
    """SFTs of one pack with one injected signal, cached in results/figure_work/<name>.

    The SFTs are generated on the first call (or with ``regenerate``) and
    reused afterwards. Returns the loaded SFTData.
    """
    sft_dir = FIGURE_WORK_DIR / name / f"sfts_tsft{tsft}"
    if regenerate or not any(sft_dir.glob("*.sft")):
        from pycbc.conversions import mchirp_from_mass1_mass2

        log.info("Generating SFTs in %s", sft_dir)
        for old in sft_dir.glob("*.sft"):
            old.unlink()
        t_start, t_end = pack_window(pack)
        m1, m2 = equal_component_masses(mchirp)
        mchirp_exact = mchirp_from_mass1_mass2(m1, m2)
        if t_to_merger is None:
            t_to_merger = t_to_merger_for_mchirp(mchirp_exact)
        raw = read_pack_frames(o3_pack_dir(pack, o3_dir), t_start)
        with tempfile.TemporaryDirectory(prefix="figure-injection-") as work_dir:
            coal_time, frame_dir = inject_signal(m1, m2, distance, t_to_merger, ra, dec, pol, t_start, raw, work_dir)
            cache = write_injected_framecache(f"{work_dir}/framecache", frame_dir, mchirp_exact, distance,
                                              coal_time, t_start)
            make_sfts(t_start, t_end, tsft, cache, sft_dir, threads)
    return load_sft_dir(sft_dir)


def hours_since_start(sft_data):
    return (np.asarray(sft_data.epochs, dtype=float) - float(sft_data.epochs[0])) / 3600.0


def normalized_log_power(values):
    """log10 of the power rescaled to [0, 1], as shown in Figs. 2 and 4."""
    values = np.log10(values + np.finfo(float).tiny)
    vmin, vmax = np.nanmin(values), np.nanmax(values)
    return (values - vmin) / (vmax - vmin) if vmax > vmin else values


def style_axis(ax):
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.grid(False)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def save(fig, path, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, **kwargs)
    plt.close(fig)
    print(f"Figure saved to {path}")


def add_common_arguments(parser):
    parser.add_argument("--regenerate", action="store_true", help="Regenerate the injected SFTs.")
    parser.add_argument("--threads", type=int, default=32, help="Parallel MakeSFTs processes.")
    parser.add_argument("--o3-dir", type=Path, default=O3_DIR)
    return parser
