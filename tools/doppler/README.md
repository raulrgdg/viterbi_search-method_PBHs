# Doppler drift of the signal frequency

`doppler_estimate.py` bounds the frequency change caused by the motion of the
detector (Earth rotation at the H1 site and orbital motion), analytically and
with the H1 velocity over O3b (astropy ephemeris), at f = 126.8 Hz:

| | Value | Fraction of the bin |
|---|---|---|
| Drift within one SFT, Tsft = 88 s | 1×10⁻⁶ Hz | 1×10⁻⁴ of 1/Tsft |
| Total excursion over one 32768 s chunk | 3×10⁻⁴ Hz | 3% of 1/88 Hz |

The fraction grows as Tsft², so even the longest SFTs of the v1.1 lists
(422 s) keep the drift per SFT below ~2×10⁻³ of a bin. Rotation dominates the
change (acceleration 2.3×10⁻² m/s² vs 5.9×10⁻³ m/s² for the orbit); the orbit
mainly shifts the whole signal by a constant factor (~1×10⁻⁴), which keeps the
track a straight line in the (t, f⁻⁸ᐟ³) map. The Doppler modulation can be
neglected.

    python tools/doppler/doppler_estimate.py
