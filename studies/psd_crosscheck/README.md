# PSD cross-check with lalpulsar_ComputePSD

Compares the paper PSD of Appendix A
(`paper_data/snr/average_noise_psd_O3b_H1.csv`: Welch, 512 s Hann segments,
50% overlap, median average) with `lalpulsar_ComputePSD` (defaults: running
median over 101 bins, harmonic mean over SFTs) computed with the same
segmentation: 512 s Hann SFTs with 50% overlap, packs 1, 4, 13, 25, 37, 49
(~73 h of O3b H1).

```bash
bash compute_psd_lalpulsar.sh   # ~1.5 min per pack
python compare_psds.py
```

**Result** (search band 61.1-126.8 Hz): ComputePSD is lower by a median 15%
(S_ComputePSD/S_Welch = 0.85, 16-84%: 0.80-0.89), which raises the optimal SNR
by 9.5%. That is within the ±10% frame-to-frame spread of the SNR caused by the
non-stationarity of the data, so the SNRs in the paper do not change
significantly. Figure: `output/psd_crosscheck.png`.
