#!/usr/bin/env python3
"""Stage 0a of the H1+L1 study: which H1 pack windows also have full L1 data.

For every pack window of o3/packs.py (32768 s = 8 GWOSC frames of 4096 s, chosen
gap-free in H1), check that the 8 L1 frames at the same GPS starts exist with
100% duty cycle in the GWOSC O3b listing. The maps of the two detectors can only
be summed bin by bin on the same window (same SFT start times).

    python studies/multi_detector/l1_coverage.py      # -> calibration/l1_packs.txt
"""
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pbh_viterbi.config import FRAME_LENGTH, NUM_FRAMES  # noqa: E402
from pbh_viterbi.o3.packs import PACK_WINDOWS  # noqa: E402

URL = "https://gwosc.org/archive/links/O3b_4KHZ_R1/{ifo}/1256655618/1269363618/simple/"
OUT = Path(__file__).resolve().parents[2] / "calibration" / "l1_packs.txt"


def duty_cycles(ifo):
    html = urllib.request.urlopen(URL.format(ifo=ifo), timeout=300).read().decode()
    return {int(g): float(p) for g, p in re.findall(r'-(\d+)-4096\.gwf".*?--dutycycle:\s*([\d.]+)%', html, re.S)}


def main():
    duty = {ifo: duty_cycles(ifo) for ifo in ("H1", "L1")}
    rows, both = [], []
    for pack, (start, end) in sorted(PACK_WINDOWS.items()):
        frames = [start + i * FRAME_LENGTH for i in range(NUM_FRAMES)]
        h1 = min(duty["H1"].get(g, 0.0) for g in frames)
        l1 = min(duty["L1"].get(g, 0.0) for g in frames)
        rows.append((pack, start, end, h1, l1))
        if l1 >= 100.0:
            both.append(pack)
    with open(OUT, "w") as f:
        f.write("# H1 pack windows with 100% L1 duty cycle in every frame (studies/multi_detector/l1_coverage.py)\n")
        f.write("# pack gps_start gps_end min_duty_H1 min_duty_L1\n")
        f.writelines(f"{p} {a} {b} {h:.1f} {l:.1f}\n" for p, a, b, h, l in rows if l >= 100.0)
    partial = [(p, l) for p, a, b, h, l in rows if 0 < l < 100]
    print(f"GWOSC frames: H1 {len(duty['H1'])}, L1 {len(duty['L1'])}")
    print(f"H1 windows at 100% in H1: {sum(h >= 100 for *_, h, l in rows)}/{len(rows)}")
    print(f"with 100% L1 as well: {len(both)} (paper packs 1-108: {sum(p <= 108 for p in both)})")
    print(f"partial L1 (0-100%): {len(partial)}, no L1: {sum(l == 0 for *_, l in rows)}")
    print("packs:", " ".join(map(str, both)))


if __name__ == "__main__":
    main()
