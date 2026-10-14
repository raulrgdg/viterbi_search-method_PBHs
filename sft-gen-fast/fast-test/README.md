# Test: adding Tsft = 1 s

Does a 14th SFT length, Tsft = 1 s, improve the recovery of high-mass signals?
Run with fast_sft (in-memory SFTs) on one node, 8 processes, ~1 h in total.

## Steps

| Step | Script | What |
|---|---|---|
| 0 | `step0_check_injection.py` | In-memory injection vs injection written to GWF frames (1 signal) |
| 1 | `step1_noise_background.py` | Noise track power of the 108 packs for the 14 Tsft → μ, σ of Tsft = 1 s for nσ |
| 2 | `step2_injections.py` | Mc = 0.085 M☉, the 30 packs of the paper × 30 distances = 900 injections (random sky); search with 13 and with 14 Tsft on the same maps |
| 3 | `step3_analysis.py` | Summary per distance |
| 4 | `step4_tsft_formula.py` | Check of the optimal-Tsft condition in the remapped map (100 injections) |

`run_all.sh` / `run_step2_3.sh` chain them (one thread per process: the node
allows 500 processes/threads per user).

## Results

* **Step 0:** strain, maps and tracks (Tsft 1 and 88 s) bitwise identical.
* **Step 1:** Tsft = 1 s background μ = 57678, σ = 1270. The paper background
  of the 13 Tsft is reproduced exactly only with the band 61.1–127.2 Hz (that of
  the noise search), not with the 126.8 Hz used by the injected search
  (0.03–0.4% in μ, up to 1.8% in σ). With Tsft = 1 s both bands give the same
  66 bins, so its background is valid for both.
* **Step 2:** adding Tsft = 1 s changes almost nothing for Mc = 0.085 M☉:
  * Tsft = 1 s is selected in 10/900 injections, all in packs 71 and 72, at
    large distances (signal near or below the noise). These two packs have the
    highest Tsft = 1 s noise nσ of the 108 (4.2 and 4.4; the next is 2.3), so
    the selection is driven by a noise feature, not by the signal.
  * Only 1 of those 10 changes status (rejected → failed fit); the fraction of
    `candidate_found` per distance is identical with 13 and 14 Tsft.
  * In the other 890 injections the result is identical. The Tsft = 1 s track
    gives about half the nσ of Tsft = 3 s (median ratio 0.48), the Tsft
    selected in 94% of the injections.
* The 13-Tsft recovery fractions agree with the paper within the statistical
  scatter of 30 injections per distance (different sky positions).

## Step 4: optimal Tsft in the remapped map

Paper Eq. (14) asks the signal to stay in one *linear* frequency bin at the
upper cutoff f1, which gives 1.4 s for Mc = 0.085 M☉, while the search selects 3 s.
The search tracks the signal in x = f^-8/3, where it moves at constant speed
K = (256/5) π^(8/3) (G Mc/c³)^(5/3), and Viterbi allows at most one remapped bin
per SFT (Δx = (f0^-8/3 − f1^-8/3)/(B·Tsft), B = f1 − f0). The condition
K·Tsft ≤ Δx gives

    T_lim = sqrt((f0^-8/3 − f1^-8/3) / (K·B)) = sqrt(Δτ_band / B) ≈ 17.7 s (Mc / 1e-2 M☉)^(-5/6)

where Δτ_band is the time the signal takes to cross the band. This is
Eq. (12) with ḟ at f1 replaced by the mean ḟ over the band (factor ≈ 2.1 longer
than Eq. 14). Below T_lim nσ grows roughly as √Tsft, so the prediction is the
largest grid Tsft ≤ T_lim.

`step4_tsft_formula.py`: the 10 grid chirp masses from 4.5e-3 to 0.1 M☉
(0.085 excluded), 10 distances each (1.8e-3 to 2.4e-2 Mpc), random packs,
13 paper Tsft (`results/tsft_formula.csv`, `results/tsft_formula_summary.csv`):

| Mc (M☉) | 4.5e-3 | 6.3e-3 | 8.9e-3 | 1.3e-2 | 1.8e-2 | 2.5e-2 | 3.5e-2 | 5.0e-2 | 7.1e-2 | 0.1 |
|---|---|---|---|---|---|---|---|---|---|---|
| T_lim (s) | 34.7 | 26.0 | 19.5 | 14.6 | 11.0 | 8.2 | 6.2 | 4.6 | 3.5 | 2.6 |
| Predicted Tsft (s) | 25 | 25 | 18 | 13 | 10 | 7 | 5 | 4 | 3 | 2 |
| Selected = predicted | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 9/10 |

The selected Tsft matches the prediction in 99/100 injections. The one miss
(Mc = 0.1, 0.018 Mpc, pack 40) is a signal not seen by any Tsft (nσ ≤ 0.7),
where the choice follows the noise. The next Tsft above T_lim loses almost all
the signal (median nσ 0–28% of the predicted one), the next one below keeps
75–88%: the cut is sharp. It is not exact at the percent level: for
Mc = 0.085 (T_lim = 2.98 s) the search still prefers 3 s (step 2).

Outputs in `results/`: `injections_14tsft.csv` (one row per injection, nσ of
every Tsft), `summary_by_distance.csv`, `noise_background_14tsft.csv`,
`step0_injection_check.csv`.
