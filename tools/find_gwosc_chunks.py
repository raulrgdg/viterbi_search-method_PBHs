#!/usr/bin/env python3
"""Busca chunks en un listado GWOSC: 8 frames consecutivos (4096 s) con Percent == 100.
Cada chunk = 8 frames -> 8*4096 s. Chunks no solapados (voraz dentro de cada racha).
Salida: texto con 'gps_inicio gps_final' por linea."""
import re
import sys
import urllib.request

URL = "https://gwosc.org/archive/links/O3b_4KHZ_R1/H1/1256655618/1269363618/simple/"
OUT = "gwosc_chunks_O3b_H1.txt"
FRAME_LEN, N_FRAMES = 4096, 8

html = urllib.request.urlopen(sys.argv[1] if len(sys.argv) > 1 else URL, timeout=120).read().decode()
# (gps, percent) por fila de la tabla
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
    f.write("# gps_inicio gps_final\n")
    f.writelines(f"{a} {b}\n" for a, b in chunks)
print(f"frames: {len(rows)}, al 100%: {len(full)}, chunks: {len(chunks)} -> {OUT}")
