"""Search configuration shared by every stage of the pipeline.

All values reproduce the benchmark search of arXiv:2607.18352 (O3b LIGO Hanford,
Table I). Change them here rather than in individual scripts so that noise
searches, injected searches and figures stay consistent.
"""

import numpy as np

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
IFO = "H1"
CHANNEL = "H1:GWOSC-4KHZ_R1_STRAIN"
OBSERVING_RUN = "O3b"
GWOSC_SAMPLE_RATE = 4096  # Hz, rate of the public GWOSC strain
SAMPLE_RATE = 512  # Hz, rate after resampling (Table I)
FRAME_LENGTH = 4096  # s, duration of one stored GWF frame
NUM_FRAMES = 8  # frames per pack -> 32768 s chunk (Table I)
PACK_DURATION = FRAME_LENGTH * NUM_FRAMES

# --------------------------------------------------------------------------
# Time-frequency maps (Sec. II C, III A)
# --------------------------------------------------------------------------
# Optimal band for the O3 H1 PSD, maximising Eq. (13).
FMIN = 61.1
FMAX = 126.8
FBAND = FMAX - FMIN

# Thirteen coherence times covering 1e-4 <= Mc/Msun <= 1e-1 with < 1% SNR loss.
TSFT_VALUES = (2, 3, 4, 5, 7, 10, 13, 18, 25, 35, 47, 63, 88)

SFT_WINDOW = "rectangular"
MAKE_SFT_THREADS = 256  # parallel lalpulsar_MakeSFTs workers per Tsft

# --------------------------------------------------------------------------
# Viterbi tracking (Sec. II B)
# --------------------------------------------------------------------------
# Log-probabilities of the three allowed jumps per step, in the order expected
# by soapcw.single_detector. In the (t, f^-8/3) map they favour centre/down moves.
VITERBI_TRANSITION_PROBS = (0.30, 0.35, 0.35)
VITERBI_TRANSITION_LOG_PROBS = np.log(VITERBI_TRANSITION_PROBS)

# --------------------------------------------------------------------------
# Candidate isolation (Sec. III A)
# --------------------------------------------------------------------------
N_POWER_WINDOWS = 8  # windows the Viterbi track is split into (stage 1)
TOP_N_BLOCKS = 2  # windows kept after the power screening
POWER_THRESHOLD_K = 0.5  # window kept if fraction > median + k * MAD
DOMINANCE_RATIO = 3.0  # keep only the best window if it dominates the second one
SIGNIFICANT_BLOCK_Z_THRESHOLD = 7.6869  # 90th percentile of the noise robust z-score
NSIGMA_SELECTION_THRESHOLD = 1.0001  # preferred minimum nsigma when choosing Tsft

# NMSE fit of the track against the inspiral model, Eqs. (21)-(25).
FIT_MASS_MIN = 1e-5  # Msun
FIT_MASS_MAX = 1e-1  # Msun
FIT_MASS_SAMPLES = 20000
NMSE_NREF = 64
NMSE_LEN_ALPHA = 1.0

# Provisional linear decision line applied inside the search (status column).
# The paper results use the polynomial FAR = 3% threshold computed afterwards
# with analysis/compute_threshold.py.
PROVISIONAL_THRESHOLD_SLOPE = 255.5
PROVISIONAL_THRESHOLD_INTERCEPT = -0.671286

# --------------------------------------------------------------------------
# Injected population (Table I)
# --------------------------------------------------------------------------
MCHIRP_GRID = np.unique(
    np.concatenate([np.logspace(np.log10(2e-4), np.log10(1e-1), 19), [0.085]])
)  # Msun
DISTANCE_GRID = np.unique(
    np.concatenate(
        [
            np.logspace(-4, -3, 4, endpoint=False),
            np.logspace(-3, -2, 4, endpoint=False),
            np.logspace(-2, -1, 16, endpoint=False),
            np.logspace(-1, np.log10(0.145), 6),
        ]
    )
)  # Mpc
MASS_RATIO = 1.0
INCLINATION = 0.0
RA_RANGE = (0.0, 2 * np.pi)
DEC_RANGE = (-np.pi / 2, np.pi / 2)
POL_RANGE = (0.0, np.pi)
DEFAULT_SKY_SEED = 20260401

# Time to merger at the start of the pack, chosen per chirp-mass range so that
# the signal crosses a useful fraction of the band within the 32768 s chunk
# (see tools/tmerger_mass_windows.py). Rows: (mchirp_min, mchirp_max, seconds).
T_TO_MERGER_BY_MCHIRP = (
    (1.0000000000000000e-04, 2.2053231801153470e-04, 1.3939520546672462e07),
    (2.2053234006476653e-04, 4.8634508413599280e-04, 3.7307317803086136e06),
    (4.8634513277050120e-04, 1.0725482012884173e-03, 9.9848193808004100e05),
    (1.0725483085432376e-03, 2.3653157626732702e-03, 2.6723073068630840e05),
    (2.3653159992048467e-03, 5.2162864185682700e-03, 7.1520831350329030e04),
    (5.2162869401969120e-03, 8.3302027440089410e-03, 3.2780000000000000e04),
)
DEFAULT_T_TO_MERGER = 32780  # s, used above the last range


def t_to_merger_for_mchirp(mchirp):
    """Return the injection time to merger (s) for a chirp mass (Msun)."""
    for mchirp_min, mchirp_max, t_to_merger in T_TO_MERGER_BY_MCHIRP:
        if mchirp_min <= mchirp <= mchirp_max:
            return t_to_merger
    return DEFAULT_T_TO_MERGER
