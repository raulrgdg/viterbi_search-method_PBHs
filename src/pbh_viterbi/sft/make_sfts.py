"""Python wrapper around make_sfts.sh (parallel lalpulsar_MakeSFTs)."""

import os
import subprocess

from pbh_viterbi.config import CHANNEL, FBAND, FMIN, SFT_WINDOW
from pbh_viterbi.paths import MAKE_SFTS_SCRIPT


def make_sfts(t_start, t_end, tsft, framecache, output_dir, num_threads, fmin=FMIN, band=FBAND,
              window=SFT_WINDOW, channel=CHANNEL, verbose=False):
    """Write the SFTs covering [t_start, t_end) with duration ``tsft`` into ``output_dir``.

    A tail shorter than ``tsft`` is dropped (``remainder_mode=trim``).
    """
    os.makedirs(output_dir, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "t_start": str(t_start),
            "t_end": str(t_end),
            "num_threads": str(num_threads),
            "Tseg": str(tsft),
            "remainder_mode": "trim",
            "SFTPATH": str(output_dir),
            "framecache": str(framecache),
            "Band": str(band),
            "fmin": str(fmin),
            "windowtype": window,
            "channel_name": channel,
            "sft_verbose": "1" if verbose else "0",
        }
    )
    subprocess.run(["bash", str(MAKE_SFTS_SCRIPT)], env=env, check=True)
