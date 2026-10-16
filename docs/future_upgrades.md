# Possible upgrades (not implemented)

Ideas identified while preparing v1.1, with how to implement and test them.
None of them changes the v1.1 results.

## 1. Bin-summed remapping to f⁻⁸ᐟ³

**Problem.** The remapped grid is uniform in x = f⁻⁸ᐟ³, so its bins have
different widths in frequency: Δf_remap(f) = Δx · (3/8) f^(11/3), i.e.
~0.6 linear bins (1/Tsft) at 61 Hz and ~4.4 at 127 Hz.

When Viterbi can follow the signal (Tsft ≤ T_lim), the signal moves less than
one *remapped* bin per SFT. Near the top of the band this can still be up to
~4 *linear* bins, over which the power of one SFT is split.
`remap_to_fm83` (`src/pbh_viterbi/sft/maps.py`) interpolates the linear
spectrum at each remapped frequency, so it samples one linear bin and loses
up to ~3/4 of the signal power there.

**Proposal.** Build each remapped pixel by summing the power of the linear
bins inside it, weighted by overlap:

1. **Edges.** The edges of remapped bin j are x_j ± Δx/2, i.e. frequency
   edges f = x^(-3/8). Linear bin k covers f_k ± 1/(2 Tsft).
2. **Overlap matrix.** Build a sparse matrix W (n_remap × n_linear) with
   W[j, k] = the fraction of linear bin k inside remapped bin j. It depends
   only on Tsft and the band, so it is built once per Tsft.
3. **Remap.** remapped = P @ Wᵀ, with P the normalised power |SFT|² / noise.
   This is one sparse product per map.
4. **Renormalise.** Pixels now sum different numbers of linear bins, so
   normalise each remapped bin to the same noise distribution (subtract
   Σ_k W[j,k] and divide by its standard deviation, or divide by its median
   over time). Otherwise Viterbi would favour high frequencies.
5. **Recompute the nσ background** of every Tsft list (`noise_search --mode
   background`). Summing power instead of the current |SFT| changes the
   statistic.

**Where.** Add `remap_summed(...)` next to `remap_to_fm83` with a cached W,
selected by an option, and keep interpolation as the default.

**Test.**
* **Noise.** The pixel distribution should be flat across the band; compare
  the noise nσ of the 108 packs (false-alarm rate).
* **Injections.** Inject heavy masses, whose tracks reach the top of the band,
  and compare nσ with both remappings. The selected Tsft should still follow
  T_lim.

## 2. One-directional Viterbi transitions

In the remapped map the signal always moves in the same direction (frequency
only increases). The current transitions allow −1, 0 and +1 bins per step, so
one of the three moves is never taken by a signal.

**Proposal.** Use transitions {0, 1, 2} in the direction of the chirp. This
keeps three options per step, so the noise background should change little,
while doubling the maximum drift. T_lim would grow by √2, allowing longer
SFTs, which is up to ~+19% in nσ (~+9% in distance) where the Tsft list is
not capped.

**Where.** `soapcw.single_detector` takes three transition probabilities for
(−1, 0, +1). Either write a small Viterbi in numpy with the new moves (cheap
at these map sizes), or shear each map by one bin per step and keep soapcw.

**Test.** Recompute the nσ background, repeat the T_lim check
(`tools/opt_tsft_remapped/injection_check_dense`) with √2 longer Tsft, and
compare nσ and the noise nσ distribution.
