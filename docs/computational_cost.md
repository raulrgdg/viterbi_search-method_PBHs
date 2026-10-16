# Computational cost

Wall-clock time, hardware, memory and scaling of the search.

## v1.1: in-memory SFTs

From v1.1 the SFTs are built in memory (`docs/fast_sft.md`). Measured on an
AMD EPYC 9475F node (shared, ±30%), one process per job:

| | Time | Memory |
|---|---|---|
| Fixed part per chunk (reading 8 frames, injecting a signal) | ~8 s | |
| Per Tsft and chunk (SFTs, remapping, Viterbi) | ~1.5 s | |
| Noise search, 13 paper Tsft | 31 s per chunk | 1.6 GB |
| Noise search, 45-Tsft list (`95`) | 64 s per chunk | 2.2 GB |
| Injection, 45-Tsft list (`95`) | 63–75 s | 2.3 GB |
| Injection, 24-Tsft list (`90`) | ~44 s | |
| nσ background, 108 packs × 79 Tsft (the three lists), 8 processes | 24 min | |

The cost grows linearly with the number of Tsft. It is nearly independent of
their values: the per-segment high-pass filter dominates, and its cost scales
with the number of samples, not of segments. No disk space is needed.

**Paper campaign** (18000 injections, plus the noise search and nσ background
on 108 chunks):

| Tsft list | Core-hours | One node, 8 processes | One node, 24 processes |
|---|---|---|---|
| `95` (45 Tsft) | ~380 | ~47 h | ~16 h |
| `90` (24 Tsft) | ~220 | ~28 h | ~9 h |
| `paper` (13 Tsft) | ~140 | ~17 h | ~6 h |

The paper used ~2.1×10⁵ allocated core-hours with MakeSFTs. The rest of this
document describes that campaign (v1.0, MakeSFTs).

## Summary (v1.0, MakeSFTs)

For one data chunk (T_obs = 32768 s ≈ 9.1 h, 13 SFT lengths from 2 to 88 s):

| Stage | Wall-clock time | Share |
|---|---|---|
| Frame reading, signal injection, start-up | ~3 min | ~3% |
| SFT generation (`lalpulsar_MakeSFTs`, 13 Tsft) | **86 min** (median) | ~97% |
| Normalisation and (t, f⁻⁸ᐟ³) remapping, 13 maps | ~1 s | <0.1% |
| Viterbi tracking, 13 maps | ~3 s | <0.1% |
| Candidate isolation and ranking | 0.01–2 s | <0.1% |
| **Total per chunk** | **89 min** (median; 46–179 min, 10th–90th percentile) | |

Everything after SFT generation takes a few seconds per chunk. SFT
generation dominates and scales linearly with the observation time and
with the sum of 1/Tsft over the SFT lengths used.

## Hardware and resources

**Campaign jobs (LIGO Caltech, HTCondor).** 7675 injected-search jobs, one
injected signal per job, each requesting 8 cores, 12 GB of memory and 12 GB
of disk. They ran on 753 nodes of a heterogeneous pool; the main node types were:

| CPU | Jobs | Wall time (median) | CPU time (median) |
|---|---|---|---|
| AMD EPYC 7313 (16 cores, 2021) | 2305 | 64 min | 26 h |
| Intel Xeon Gold 6136 (12 cores, 2017) | 2091 | 76 min | 26 h |
| Intel Xeon E5-2650 v4 (12 cores, 2016) | 1368 | 128 min | 38 h |
| Intel Xeon E5-2670 (8 cores, 2012) | 604 | 147 min | 38 h |
| Intel Xeon E3-1240 v5 (4 cores, 2015) | 516 | 180 min | 23 h |

Each job runs 256 concurrent `lalpulsar_MakeSFTs` processes (see
`studies/strong_scaling`), so the CPU time (median 26 core-hours per job)
exceeds the 8 requested cores whenever the node has idle capacity. The
processes are largely I/O bound: wall time saturates at ~256 processes.

**Memory.** Median 2.0 GB per job; 90% of the jobs used less than 9.9 GB and
none exceeded the 12 GB request. The Python process itself (all 13 remapped
maps in memory) peaks at ~2 GB; the rest is used by the concurrent MakeSFTs
processes and the frame files.

**Disk.** Each job reads 8 frames (134 MB) and writes the injected copy
(134 MB) and the SFTs of one Tsft at a time (~20–40 MB) to scratch space; all
of it is deleted when the job ends.

**Local machine.** On one 24-core AMD EPYC 9475F node (125 GB RAM) with
24 MakeSFTs processes, a noise chunk takes ~30 min when the node is idle and
~60 min when shared.

## Scaling

**With the SFT length.** The SFT generation time of one Tsft is inversely
proportional to Tsft, i.e. proportional to the number of SFTs
N_SFT = T_obs / Tsft:

| Tsft (s) | 2 | 3 | 4 | 5 | 7 | 10 | 13 | 18 | 25 | 35 | 47 | 63 | 88 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MakeSFTs (s, median) | 1422 | 961 | 724 | 579 | 415 | 291 | 226 | 161 | 117 | 83 | 62 | 47 | 34 |
| Tsft × time (s²) | 2844 | 2883 | 2897 | 2893 | 2904 | 2910 | 2942 | 2903 | 2925 | 2905 | 2928 | 2942 | 2983 |

Each SFT costs ~0.09 s of wall time on the campaign nodes, so
t_SFT(Tsft) ≈ 2900 s × (T_obs / 32768 s) / Tsft.

**With the number of SFT lengths.** The total is the sum over the Tsft
used: t ≈ 2900 s × (T_obs / 32768 s) × Σ 1/Tsft. For the 13 values of the
paper Σ 1/Tsft = 1.78, giving 86 min (measured median: 86 min). The four
shortest SFTs (2–5 s) account for 64% of the time, so a search restricted to
a narrower chirp-mass range (fewer, longer SFTs) is much cheaper: e.g.
Mc ≲ 5×10⁻³ M☉ only needs Tsft ≥ 13 s (Σ 1/Tsft = 0.25) and costs ~1/7 of
the full search, ~12 min per chunk.

**With the observation time.** Every stage is linear in T_obs at fixed Tsft:

* SFT generation: ∝ N_SFT ∝ T_obs.
* Remapping and Viterbi: ∝ number of map pixels = N_SFT × N_bins
  = T_obs × bandwidth (≈ 2.15×10⁶ per map for 32768 s and 65.7 Hz),
  independent of Tsft; ∝ T_obs × N_Tsft overall.
* Candidate isolation: ∝ track length N_SFT of the selected Tsft
  (0.01 s for 88 s SFTs, 0.5–2 s for 2 s SFTs).

The analysis of independent chunks is embarrassingly parallel: the noise
search uses one job per chunk and the injected search one job per signal.

## Cost of the paper campaigns

* Injected search: 600 signals × 30 noise realisations = 18000 injections.
  At the median of 89 min on 8-core slots this is ~2.1×10⁵ allocated
  core-hours (~4.7×10⁵ core-hours of CPU time with the oversubscription
  above, extrapolating the Caltech per-job values), spread over three clusters. The Caltech share alone used
  2.2×10⁵ CPU core-hours over ~6 weeks (including test runs).
* Noise search: 108 chunks (~983 h of data), ~1.5 h each, ~1.3×10³ allocated
  core-hours.

## How the numbers were obtained

* Wall time, CPU time, memory and node of each job: HTCondor event logs of
  the Caltech campaign (job execution and termination records).
* MakeSFTs time per Tsft: timing lines written by each job for every Tsft.
* Remapping, Viterbi and candidate search: `studies/computational_cost/benchmark_stages.py`
  on synthetic maps of the real size (AMD EPYC 9475F, after a warm-up call;
  the first call to soapcw adds ~5 s of import and compilation).
* Local chunk times: `noise_search` runs on O3b packs 4, 37 and 49.
