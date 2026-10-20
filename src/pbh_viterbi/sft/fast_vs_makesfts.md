# In-memory SFTs vs `lalpulsar_MakeSFTs`

Summary of the study that replaced `lalpulsar_MakeSFTs` (v1.0, used for the
paper) with in-memory SFT generation (`fast.py`, default from v1.1). The two
methods give the same maps and the same search results; the in-memory method
is ~190× faster for the SFT step and ~100× faster for a whole chunk.

* `fast.py`: in-memory SFTs (`--sft-method fast`, default).
* `make_sfts.py`, `make_sfts.sh`: `lalpulsar_MakeSFTs` on GWF frames
  (`--sft-method makesfts`, paper method, integer Tsft only).
* `maps.py`: remapping to (t, f^-8/3) and Viterbi tracking, common to both.

## 1. Motivation

With MakeSFTs, SFT generation took ~97% of the run time of a chunk: 86 of
89 min (median) on the campaign nodes, for one 32768 s chunk and the 13 Tsft
of the paper. The cost was not the FFTs (all the SFTs of a chunk take < 1 s
with numpy) but the overhead of MakeSFTs: it re-reads the frames for every
SFT, runs hundreds of processes, and writes, validates and reads back ~58000
`.sft` files per chunk.

## 2. What MakeSFTs computes

Found by comparing its output with a direct FFT of the same strain. With the
options of the search (`-w rectangular -f FMIN -F FMIN -B BAND`), MakeSFTs
processes **each SFT segment independently**:

1. high-pass filters the segment with a 10th-order Butterworth filter,
   amplitude 0.5 at FMIN (`lal.ButterworthREAL8TimeSeries`);
2. FFTs it and multiplies by dt;
3. keeps round(BAND·Tsft) bins starting at bin round(FMIN·Tsft);
4. stores the result in single precision.

`fast.sft_matrix` does the same four steps on the strain array of the chunk,
calling the same LAL filter. The per-segment filter is essential: without it,
the large low-frequency power of the strain leaks into the search band through
the rectangular window. Filtering each segment (and not the whole chunk) is
also why the MakeSFTs results do not depend on the number of parallel
processes.

## 3. Validation

O3b LIGO Hanford data, 13 paper Tsft (2–88 s), packs drawn at random with a
fixed seed; both methods on the same raw frames.

| Test | Data | Result |
|---|---|---|
| Noise search | 50 packs, compared with the paper noise results (MakeSFTs) | 45 identical to all digits; 5 differ only in the last digit of the mass or NMSE (relative 1–3×10⁻¹⁶) |
| SFTs, remapped maps and Viterbi tracks | 2 noise packs (40, 77) and 3 injections, all 13 Tsft: 65 maps | 61 maps bitwise identical; 4 differ in 1–3 pixels out of ≥ 2×10⁶ by ≤ 6×10⁻⁸ (one single-precision rounding step, numpy vs FFTW); **all 65 Viterbi tracks identical** |
| Injected search | 3 signals near the detection limit (𝓜 = 1.1×10⁻³, 4.5×10⁻³, 7.1×10⁻² M☉; packs 21, 41, 107) | identical candidate, nσ, NMSE, mass and selected Tsft (88, 25 and 3 s) |
| In-memory injection | waveform added to the strain in memory vs written to and read from GWF frames | strain bitwise identical |
| v1.1 workflows | one pack: noise search and one injection, `fast` vs `makesfts` | identical result rows |

The only differences are floating-point rounding at the level of single
precision, and none of them changes a search result.

## 4. Computational cost

AMD EPYC 9475F node (24 cores, shared), one 32768 s chunk, 13 paper Tsft.
MakeSFTs with 12 parallel processes (including reading the SFT files back),
in-memory method in one process; median of 5 chunks.

**SFT generation per Tsft:**

| Tsft (s) | 2 | 3 | 4 | 5 | 7 | 10 | 13 | 18 | 25 | 35 | 47 | 63 | 88 | **Total** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MakeSFTs (s) | 909 | 613 | 456 | 365 | 266 | 194 | 148 | 108 | 80 | 54 | 41 | 29 | 23 | **3270 (55 min)** |
| In memory (s) | 1.3 | 1.2 | 1.7 | 1.3 | 1.4 | 1.2 | 1.3 | 1.2 | 1.2 | 1.2 | 1.3 | 1.2 | 1.2 | **17** |
| Speed-up | 700 | 500 | 270 | 280 | 200 | 160 | 110 | 90 | 70 | 46 | 33 | 25 | 20 | **~190** |

MakeSFTs scales as 1/Tsft (number of SFT files). The in-memory method takes
~1.2 s for every Tsft: the FFT work is the same, only the segment length
changes, and the time is dominated by the per-segment Butterworth filter.

**Whole noise search per chunk** (in memory, packs 3, 9, 40, 77, 108):

| Stage | Read frames | SFTs (13 Tsft) | Remap + Viterbi | Candidate | **Total** |
|---|---|---|---|---|---|
| Median (s) | 7.6 | 18.1 | 4.1 | 0.1 | **31** (28–33) |

The same search with MakeSFTs takes ~55 min per chunk on this node (89 min
median on the campaign nodes), so a full noise search goes from a cluster run
to about one hour in a single process.

**Memory and disk:**

| | MakeSFTs | In memory |
|---|---|---|
| Peak memory per chunk | median 2.0 GB per campaign job (90% below 9.9 GB, mostly concurrent MakeSFTs processes) | 1.6 GB (0.66 GB of imports, ~0.9 GB of data) |
| Disk | scratch for the injected frames and the `.sft` files | none |

## 5. Cost in v1.1

The cost grows linearly with the number of Tsft (~1.5 s per Tsft and chunk,
plus ~8 s fixed). With the 45-Tsft list of v1.1 a noise chunk takes ~64 s and
an injection 63–75 s (2.2–2.3 GB), on the same node. The whole paper
injection campaign (18000 injections) drops from ~2.1×10⁵ allocated
core-hours with MakeSFTs to ~380 core-hours on this node (older cluster
nodes are 2–3× slower per injection).

## 6. Reproducing

The comparison scripts are in the v1.0 working copy (`sft-gen-fast/`):
`comparison/noise_50packs.py`, `comparison/maps_2packs.py`,
`comparison/injections_3packs.py` and `diagnose_makesfts.py` (bin alignment
and filter diagnosis). In v1.1, either method can be selected in both
workflows:

```bash
python -m pbh_viterbi.workflows.noise_search --packs 3 --sft-method fast
python -m pbh_viterbi.workflows.noise_search --packs 3 --sft-method makesfts --tsft-list paper
```
