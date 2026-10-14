# Upgrade: bin-summed remapping to f^-8/3

Proposed change to the remapping step; not implemented yet.

## Problem

The remapped grid is uniform in x = f^-8/3, so its bins have different widths
in frequency. A remapped bin spans Δf_remap(f) = Δx · (3/8) f^(11/3):
~0.6 linear bins (1/T_SFT) at 61 Hz and ~4.4 linear bins at 127 Hz.

When Viterbi can follow the signal (s ≤ 1, see `fast-test/README.md`), the
signal moves less than one remapped bin during one SFT. At high frequency
that can still be several linear bins, so the power of one SFT is split over
up to ~4 linear bins. `remap_to_fm83` (`src/pbh_viterbi/sft/maps.py`)
**interpolates** the linear spectrum at each remapped frequency
(`np.interp`). It therefore samples one linear bin and loses the rest of the
signal power in that remapped bin, up to ~3/4 of it near 127 Hz.

## Proposal

Build each remapped pixel by **summing the power of all the linear bins that
fall inside it**, weighted by overlap, instead of interpolating.

1. **Edges.** For remapped bin j with centre x_j, the edges are
   x_j ± Δx/2, which correspond to the frequency edges f = x^(-3/8).
   Linear bin k covers f_k ± 1/(2 T_SFT).
2. **Overlap matrix.** Build W (n_remap × n_linear, sparse), with
   W[j, k] = the fraction of linear bin k inside remapped bin j. Each row
   covers ~0.6–4.4 linear bins, and each column sums to 1. Build it once per
   T_SFT, since it depends only on T_SFT and the band.
3. **Remap.** remapped = P @ W.T, where P (n_sft × n_linear) is the
   normalised **power** |SFT|² / noise. This is one sparse matrix product per
   map, as fast as the current loop.
4. **Renormalise.** Each pixel now sums a different number of linear bins
   (more at high f), so its noise mean and variance depend on frequency.
   Without a correction, Viterbi would favour high-f bins. Divide each
   remapped bin by its noise level: subtract the expected noise sum
   Σ_k W[j,k] and divide by its standard deviation (or take the median over
   time of each remapped bin), so that all pixels have the same noise
   distribution.
5. **Recompute the noise background.** μ and σ of the track power for every
   T_SFT change, because the pixel statistics change. Rerun the noise
   background on the 108 packs (`fast-test/step1_noise_background.py`, ~15 min).

Note: the current map uses the normalised amplitude |SFT| / median. Summing
power (|SFT|²) is the natural choice for the sum (χ² statistics), but it
changes the statistic, so steps 4 and 5 are needed in any case.

## Where

* `src/pbh_viterbi/sft/maps.py`: add `remap_summed(power, tsft, fmin, band)`
  next to `remap_to_fm83`, with a cached W per (T_SFT, band). Select it with an
  option, keeping interpolation as the default so the paper results stay
  reproducible.
* `sft-gen-fast/fast-test/common.py` (`products`): the same switch, for
  testing.

## Validation

* **Noise.** Check that the pixel distribution is flat across the band after
  step 4, and compare the noise nσ of the 108 packs (false-alarm rate).
* **Injections.** Repeat step 4 of `fast-test` (100 injections,
  4.5e-3–0.1 M☉) with both remappings. The expected gain is at high mass,
  where the track reaches the top of the band. Measure the change in nσ, and
  check that the selected T_SFT still follows T_lim (the s ≤ 1 condition does
  not change).
* **Cost.** Negligible: one sparse product per map.
