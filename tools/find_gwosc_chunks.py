#!/usr/bin/env python3
"""Find gap-free data chunks in a GWOSC frame listing.

A chunk is 8 consecutive 4096 s frames with 100% duty cycle (32768 s, the pack
length of the search). Chunks do not overlap (greedy within each run of full
frames). Output: a text file with one 'gps_start gps_end' line per chunk.

    python tools/find_gwosc_chunks.py            # O3b H1 listing
    python tools/find_gwosc_chunks.py <URL>      # another GWOSC listing
"""
import re
import sys
import urllib.request

URL = "https://gwosc.org/archive/links/O3b_4KHZ_R1/H1/1256655618/1269363618/simple/"
OUT = "gwosc_chunks_O3b_H1.txt"
FRAME_LEN, N_FRAMES = 4096, 8

html = urllib.request.urlopen(sys.argv[1] if len(sys.argv) > 1 else URL, timeout=120).read().decode()
# (gps, duty cycle) of every frame in the table
rows = re.findall(r'-(\d+)-4096\.gwf".*?--dutycycle:\s*([\d.]+)%', html, re.S)
full = sorted({int(g) for g, p in rows if float(p) >= 100.0})

chunks, run = [], []
for g in full + [None]:
    if run and (g is None or g != run[-1] + FRAME_LEN):
        chunks += [(run[i], run[i + N_FRAMES - 1] + FRAME_LEN)
                   for i in range(0, len(run) - N_FRAMES + 1, N_FRAMES)]
        run = []
    if g is not None:
        run.append(g)

with open(OUT, "w") as f:
    f.write("# gps_start gps_end\n")
    f.writelines(f"{a} {b}\n" for a, b in chunks)
print(f"frames: {len(rows)}, at 100%: {len(full)}, chunks: {len(chunks)} -> {OUT}")
