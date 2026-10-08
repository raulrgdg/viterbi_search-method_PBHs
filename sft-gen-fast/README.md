# sft-gen-fast

In-memory SFT generation for the pbh_viterbi search, replacing
`lalpulsar_MakeSFTs`. Same results, ~75× faster per data chunk.

## Why

In the pipeline, SFT generation takes ~97% of the run time (86 of 89 min per
32768 s chunk on the campaign nodes; see `../docs/computational_cost.md`).
That time is not spent on FFTs (all 13 SFT lengths of a chunk take < 1 s with
numpy) but on the overhead of MakeSFTs: re-reading the frames, starting
hundreds of processes, and writing and re-reading ~58000 `.sft` files per chunk.

## What MakeSFTs does

Determined by comparing its output with a direct FFT of the same strain
(`diagnose_makesfts.py`). With the options used by the pipeline
(`-w rectangular -f FMIN -F FMIN -B BAND`), for **each SFT segment
independently** MakeSFTs:

1. high-pass filters the segment with a 10th-order Butterworth filter, with
   amplitude 0.5 at FMIN (`lal.ButterworthREAL8TimeSeries`);
2. FFTs it and multiplies by dt;
3. keeps round(BAND·Tsft) bins starting at bin round(FMIN·Tsft);
4. stores it in single precision.

Without step 1 the result differs completely: the large low-frequency power
of the strain leaks into the search band through the rectangular window.
Filtering per segment (not over the whole chunk) is also why MakeSFTs results
do not depend on how many parallel processes are used.

`fast_sft.py` reproduces these four steps in memory.

## Validation

On O3b H1 packs 4, 37 and 49:

| Check | Result |
|---|---|
| Remapped power maps, Tsft = 88 s and 7 s (pack 4) | bitwise identical to MakeSFTs |
| Viterbi tracks, same cases | identical |
| Noise search, all 13 Tsft, band 61.1–127.2 Hz | nσ, NMSE, mass, status and candidate identical to all digits to `noise_search_O3b_H1_108packs.csv` |

| pack | nσ | NMSE | mass (M☉) | Tsft |
|---|---|---|---|---|
| 4 | -0.311038 | 0.338677 | 3.213e-3 | 5 s |
| 37 | -0.230624 | 0.009746 | 3.203e-3 | 35 s |
| 49 | -0.887085 | 0.684194 | 2.154e-3 | 5 s |

## Speed

Per chunk (32768 s, 13 Tsft), one process on an AMD EPYC 9475F node
(`results/timing_fmax127.2.csv`):

| Stage | MakeSFTs pipeline | fast_sft |
|---|---|---|
| Read frames | (inside MakeSFTs) | 6.7 s |
| SFT generation | ~30 min on this node (24 processes); 86 min median on the campaign nodes | 13.3 s |
| Remap + Viterbi (13 maps) | ~4 s | 3.6 s |
| Candidate search | < 1 s | 0.2 s |
| **Total** | **~30 min** (here), 89 min (campaign median) | **~24 s** |

The whole noise search (108 chunks, ~983 h of data) would take ~45 min in a
single process on one machine, instead of a 108-job cluster run. The SFT step
is now dominated by the per-segment Butterworth filter (a Python loop over
segments calling LAL); it could be vectorised further if needed.

## Usage

```bash
export PBH_VITERBI_O3_DIR=/path/to/O3-data          # folders O3b-packN
python run_noise_search_fast.py --packs 4,37,49 --fmax 127.2 \
    --output results/noise_fast.csv --timing results/timing.csv
python compare_maps.py --pack 4 --tsft 88 7          # map-by-map check against MakeSFTs
python diagnose_makesfts.py --pack 4 --tsft 88       # bin alignment and filter diagnosis
```

`--fmax 127.2` reproduces the noise search of the paper; the default is the
pipeline band 61.1–126.8 Hz. Requires the `pbh_viterbi` package of this
repository (`../src`).

## Next steps

* Add `fast_sft` to `pbh_viterbi.sft` as an option (`--sft-method fast`) of
  both workflows.
* Injected search: inject the waveform directly into the in-memory strain,
  skipping the GWF frames written today, and check it against the campaign
  results (identical injections need the same sky positions).
* Validate on more packs and on Tsft = 2 s maps before using it for new results.
