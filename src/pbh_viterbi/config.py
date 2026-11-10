"""Search configuration shared by every stage of the pipeline.

Values follow the benchmark search of arXiv:2607.18352 (O3b LIGO Hanford,
Table I), with the v1.1 changes: in-memory SFTs, SFT durations chosen with the
tracking limit of the remapped map, and one band for every search product.
Change them here rather than in individual scripts so that noise searches,
injected searches, the nsigma background and figures stay consistent.
"""

import numpy as np

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
IFO = "H1"
CHANNEL = "H1:GWOSC-4KHZ_R1_STRAIN"
# Detectors of a search: H1 alone (default, the paper search) or H1 and L1 (maps summed, studies/multi_detector).
DETECTORS = ("H1",)
SUPPORTED_DETECTORS = ("H1", "L1")


def channel_for(ifo):
    """GWOSC strain channel of a detector."""
    return f"{ifo}:GWOSC-4KHZ_R1_STRAIN"

OBSERVING_RUN = "O3b"
GWOSC_SAMPLE_RATE = 4096  # Hz, rate of the public GWOSC strain
SAMPLE_RATE = 512  # Hz, rate after resampling (Table I)
FRAME_LENGTH = 4096  # s, duration of one stored GWF frame
NUM_FRAMES = 8  # frames per pack -> 32768 s chunk (Table I)
PACK_DURATION = FRAME_LENGTH * NUM_FRAMES

# --------------------------------------------------------------------------
# Time-frequency maps (Sec. II C, III A)
# --------------------------------------------------------------------------
# Optimal band for the O3 noise curve, maximising Eq. (13) (tools/optimal_freq_range.py).
# Used by the noise search, the injected search and the nsigma background alike.
FMIN = 61.1
FMAX = 126.8
FBAND = FMAX - FMIN

# SFT durations (s). Viterbi can follow an inspiral in the remapped map only if
# Tsft <= T_lim = 17.7 s (Mc / 1e-2 Msun)^(-5/6), and below T_lim
# nsigma / max nsigma ~ sqrt(Tsft / T_lim) (docs/tsft_optimum.md). The lists are
# the smallest sets, in multiples of 1/8 s, that keep nsigma / max nsigma above
# 0.95 or 0.90 for every chirp mass in [2e-4, 0.1] Msun
# (tools/opt_tsft_remapped/compare_tsft_lists.py). "paper" is the list of
# arXiv:2607.18352, chosen with Eq. (14).
TSFT_LISTS = {
    "95": (2.625, 2.875, 3.125, 3.5, 3.875, 4.25, 4.75, 5.25, 5.875, 6.625, 7.375, 8.25, 9.25, 10.375, 11.625,
           13.125, 14.75, 16.625, 18.75, 21.0, 23.625, 26.75, 29.875, 33.75, 38.125, 42.875, 48.125, 54.25,
           61.375, 69.25, 78.25, 88.125, 99.5, 112.375, 126.875, 142.5, 160.0, 180.75, 204.125, 230.5, 259.625,
           293.25, 331.125, 373.875, 422.25),
    "90": (2.625, 3.0, 3.75, 4.5, 5.625, 6.875, 8.5, 10.625, 13.375, 16.75, 20.75, 26.0, 32.375, 40.875, 50.75,
           63.5, 79.5, 99.5, 124.5, 154.5, 193.875, 238.125, 298.75, 377.0),
    "paper": (2, 3, 4, 5, 7, 10, 13, 18, 25, 35, 47, 63, 88),
}
# "union": every Tsft of the three lists (79 values), used only by the
# Tsft-selection study (studies/tsft_selection), which evaluates several
# selection rules offline on the per-Tsft products of one run.
TSFT_LISTS["union"] = tuple(sorted(set().union(*TSFT_LISTS.values())))
DEFAULT_TSFT_LIST = "95"
TSFT_VALUES = TSFT_LISTS[DEFAULT_TSFT_LIST]

# SFT generation: "fast" (in memory, pbh_viterbi.sft.fast) or "makesfts"
# (lalpulsar_MakeSFTs, as in the paper; integer Tsft only, ~100x slower).
SFT_METHOD = "fast"
# High-pass of the in-memory SFTs (default "chunk" since v1.3): "chunk" filters the whole chunk
# once; "segment" filters every SFT segment separately, as lalpulsar_MakeSFTs (paper, v1.0-v1.2).
# With the rectangular window, the filter start transient of every segment leaks broadband power
# into the band, raising the noise floor x1.5-17 above the PSD (studies/multi_detector).
SFT_HIGHPASS = "chunk"
SFT_HIGHPASS_MODES = ("segment", "chunk")
# Normalisation of the |SFT| map (default "bin+sft" since v1.3): "bin" divides every frequency bin
# by its median over the chunk (paper, v1.0-v1.2); "bin+sft" then also divides every SFT by its
# median over the band, which flattens broadband glitches (studies/multi_detector).
# Every (detectors, high-pass, normalisation) combination has its own nsigma background
# (paths.noise_background_path). The v1.2 maps are --sft-highpass segment --map-normalisation bin.
MAP_NORMALISATION = "bin+sft"
MAP_NORMALISATION_MODES = ("bin", "bin+sft")
SFT_WINDOW = "rectangular"
MAKE_SFT_THREADS = 256  # parallel lalpulsar_MakeSFTs workers per Tsft (makesfts method only)

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

# Provisional linear decision line of the max-nsigma selection (v1.0/v1.1,
# status column). The paper results use the polynomial FAR = 3% threshold
# computed afterwards with analysis/compute_threshold.py.
PROVISIONAL_THRESHOLD_SLOPE = 255.5
PROVISIONAL_THRESHOLD_INTERCEPT = -0.671286

# --------------------------------------------------------------------------
# Tsft selection and ranking statistic (studies/tsft_selection,
# studies/ranking_statistic)
# --------------------------------------------------------------------------
# "hybrid": isolate a candidate on every Tsft and keep the one of largest
#   R_a = nsigma - a log10(NMSE_eff), the statistic of the detection threshold.
# "max_nsigma": the Tsft of largest nsigma, then isolate (v1.0 and v1.1).
TSFT_SELECTION = "hybrid"
# a of R_a. Calibrated on injections outside the campaign: 2000 signals in packs 95, 104,
# 113, 115, 126, 127, 128, 131, 134, 140 (every signal slice), with the noise search of the
# 141 packs; largest efficiency at FAR 1% (studies/a_calibration, results/a_calibration).
# The 3-fold noise cross-validation keeps 1/141 noise packs above the threshold out of fold.
RANKING_NMSE_WEIGHT = 2.0
# NMSE_eff = NMSE + NMSE_QUANT_COEFF / D^2, with D the drift of the fitted inspiral
# across the isolated block, in remapped bins (search.ranking.quantised_nmse).
# The Viterbi track lives on integer bins: rounding a line to the grid leaves
# residuals uniform in [-1/2, 1/2] bin (mean square 1/12), and a ramp of D bins has
# mean square displacement D^2/3, so no fit resolves an NMSE below
# (1/12) / (D^2/3) = 1/(4 D^2). Without it, short noise tracks that jump one bin per
# step (straight lines on the grid) were fitted with NMSE down to 1e-9.
# The coefficient follows from this argument, it is not fitted
# (tests/test_ranking.py checks it on the fit of quantised ramps).
NMSE_QUANT_COEFF = 0.25
# Provisional R_a threshold for the candidate/status columns: FAR 3% (4/141) of the noise
# search with a = 2. The final threshold comes from analysis/compute_threshold_linear.py.
PROVISIONAL_RANKING_THRESHOLD = 7.41

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
