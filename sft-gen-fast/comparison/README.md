# MakeSFTs vs fast_sft comparison

Checks that `fast_sft` (in-memory SFTs) gives the same search results as
`lalpulsar_MakeSFTs`. Packs drawn at random (fixed seed, `common.py`); all runs
on one AMD EPYC 9475F node (24 cores).

## Tests

| Test | Script | What | Output |
|---|---|---|---|
| 1. Noise search | `noise_50packs.py` | fast_sft noise search on 50 packs vs the paper CSV `noise_search_O3b_H1_108packs.csv` (made with MakeSFTs) | `results/noise_50packs.csv` |
| 2. Maps | `maps_2packs.py` | both methods on the same raw frames, packs 40 and 77, all 13 Tsft: SFTs, remapped maps and Viterbi tracks | `results/maps_2packs.csv` |
| 3. Injections | `injections_3packs.py` | one injected frame set per signal (Mc = 1.1e-3, 4.5e-3, 7.1e-2 M☉ near the detection limit; packs 21, 41, 107), read by both methods; full search with each | `results/injections_maps.csv`, `results/injections_results.csv` |

## Results

Search results (candidate, nσ, NMSE, mass, Tsft, status) are the same in all
cases; all 65 Viterbi tracks are identical.

* **Noise:** 45/50 packs identical to all digits; in the other 5, only the last
  digit of mass or NMSE differs (relative 1–3 × 10⁻¹⁶).
* **Maps (tests 2 and 3, 65 maps):** 61 bitwise identical; in 4, 1–3 pixels out
  of ≥ 2 × 10⁶ differ by ≤ 6 × 10⁻⁸ (one single-precision rounding step,
  numpy vs FFTW FFT).
* **Injections:** the 3 signals are recovered with identical results by both
  methods (selected Tsft 88, 25 and 3 s).

The differences are at the level of floating-point rounding and do not change
any result.

## Computational time per chunk

32768 s chunk, SFT generation only, median of the 5 chunks of tests 2 and 3
(MakeSFTs with 12 parallel processes, includes reading the SFT files back;
fast_sft in one process):

| Tsft (s) | 2 | 3 | 4 | 5 | 7 | 10 | 13 | 18 | 25 | 35 | 47 | 63 | 88 | **Total** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MakeSFTs (s) | 909 | 613 | 456 | 365 | 266 | 194 | 148 | 108 | 80 | 54 | 41 | 29 | 23 | **3270 (55 min)** |
| fast_sft (s) | 1.3 | 1.2 | 1.7 | 1.3 | 1.4 | 1.2 | 1.3 | 1.2 | 1.2 | 1.2 | 1.3 | 1.2 | 1.2 | **17** |
| Speed-up | 700 | 500 | 270 | 280 | 200 | 160 | 110 | 90 | 70 | 46 | 33 | 25 | 20 | **~190** |

MakeSFTs time scales as 1/Tsft (number of SFT files); fast_sft is ~1.2 s for
every Tsft (the FFT work is the same, only the segment length changes).

**Whole noise search per chunk with fast_sft** (reading frames, SFTs,
remapping, Viterbi and candidate search, 13 Tsft, one process), run alone on
packs 3, 9, 40, 77 and 108 (`results/timing_noise_standalone.csv`):

| Stage | Read frames | SFTs (13 Tsft) | Remap + Viterbi | Candidate | **Total** |
|---|---|---|---|---|---|
| Median (s) | 7.6 | 18.1 | 4.1 | 0.1 | **31** (28–33) |

For comparison, the same search with MakeSFTs takes ~55 min per chunk on this
node, so the whole 108-chunk noise search drops from a cluster run to ~1 h in a
single process. The node had other users' jobs running (load ~15/24); on an
idle node the total was ~24 s (`../results/timing_fmax127.2.csv`). During
test 1, sharing the node with tests 2 and 3, it was 41 s (27–64 s).

## Memory per chunk

Peak memory (maximum resident set size, `/usr/bin/time -v`) of the whole
fast_sft noise search of one chunk, 13 Tsft, one process (packs 9 and 40):

| | Peak memory |
|---|---|
| fast_sft search, total | **1.6 GB** |
| of which Python imports (LAL, PyCBC, soapcw) | 0.66 GB |
| of which data (strain 134 MB, SFTs of one Tsft, 13 remapped maps ≈ 0.2 GB) | ~0.9 GB |

No disk space is needed: nothing is written. With MakeSFTs the campaign jobs
used a median of 2.0 GB (90% below 9.9 GB, mostly the concurrent MakeSFTs
processes) plus scratch disk for the frames and SFT files
(`../../docs/computational_cost.md`).
