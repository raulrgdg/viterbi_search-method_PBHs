#!/usr/bin/env python3
"""Doppler frequency drift within one SFT and over one chunk, H1, O3b.

Analytic bound: df_D(T_SFT) = f_max |a| T_SFT / c, with the centripetal
accelerations of the Earth's rotation at the H1 latitude and of the orbit
(1 AU). Cross-check: velocity of H1 with respect to the solar-system
barycentre over O3b (astropy builtin ephemeris + H1 position from LAL), its
maximum acceleration |dv/dt| and the largest change of the line-of-sight
velocity within a 32768 s chunk (max |v(t1) - v(t2)|, worst sky direction).

    python tools/doppler/doppler_estimate.py
"""

import lal
import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation, get_body_barycentric_posvel
from astropy.time import Time

F_MAX = 126.8  # Hz
T_CHUNK = 32768.0  # s
TSFT = (2, 3, 4, 5, 7, 10, 13, 18, 25, 35, 47, 63, 88)
C = lal.C_SI
O3B = ("2019-11-01T15:00:00", "2020-03-27T17:00:00")
STEP = 60.0  # s, ephemeris sampling


def analytic():
    lat = np.radians(46.455)
    a_rot = lal.TWOPI / lal.DAYSID_SI * lal.TWOPI / lal.DAYSID_SI * lal.REARTH_SI * np.cos(lat)
    a_orb = (lal.TWOPI / lal.YRSID_SI) ** 2 * lal.AU_SI
    v_rot = lal.TWOPI / lal.DAYSID_SI * lal.REARTH_SI * np.cos(lat)
    # Largest change of the line-of-sight velocity in one chunk: chord of the
    # rotation circle over T_CHUNK plus the orbital change a_orb * T_CHUNK.
    dv_chunk = 2 * v_rot * np.sin(np.pi * T_CHUNK / lal.DAYSID_SI) + a_orb * T_CHUNK
    return a_rot, a_orb, dv_chunk


def h1_velocity(start, end, step):
    """SSB velocity of H1 (m/s), shape (n, 3), sampled every ``step`` s."""
    times = Time(start, scale="utc") + np.arange(0, (Time(end) - Time(start)).sec, step) * u.s
    _, v_earth = get_body_barycentric_posvel("earth", times)
    site = EarthLocation.from_geocentric(*lal.CachedDetectors[lal.LHO_4K_DETECTOR].location, unit=u.m)
    _, v_site = site.get_gcrs_posvel(times)
    v = (v_earth.xyz + v_site.xyz).to(u.m / u.s).value.T
    return v


def max_dv_in_window(v, n_window, stride):
    """Largest |v(t1) - v(t2)| with |t1 - t2| <= window, over all windows."""
    best = 0.0
    for start in range(0, len(v) - n_window, stride):
        w = v[start:start + n_window]
        d = np.linalg.norm(w[:, None, :] - w[None, :, :], axis=-1)
        best = max(best, d.max())
    return best


def main():
    a_rot, a_orb, dv_chunk = analytic()
    print(f"Analytic: a_rot = {a_rot:.3e} m/s^2, a_orb = {a_orb:.3e} m/s^2, f_max = {F_MAX} Hz")
    print(f"{'T_SFT':>6} {'1/T_SFT':>9} {'dfD_rot':>9} {'dfD_orb':>9} {'dfD_tot':>9} {'ratio':>8}   (Hz)")
    for t in TSFT:
        rot, orb = F_MAX * a_rot / C * t, F_MAX * a_orb / C * t
        print(f"{t:6d} {1 / t:9.4f} {rot:9.2e} {orb:9.2e} {rot + orb:9.2e} {(rot + orb) * t:8.1e}")

    v = h1_velocity(*O3B, STEP)
    a = np.linalg.norm(np.diff(v, axis=0), axis=1) / STEP
    a_max = a.max()
    print(f"\nCross-check over O3b ({O3B[0][:10]} to {O3B[1][:10]}, astropy ephemeris):")
    print(f"  max |dv/dt| = {a_max:.3e} m/s^2 (analytic sum {a_rot + a_orb:.3e})")
    print(f"  max df/dt at {F_MAX} Hz = {F_MAX * a_max / C:.2e} Hz/s; "
          f"drift in 88 s = {F_MAX * a_max / C * 88:.2e} Hz = {F_MAX * a_max / C * 88 ** 2:.1e} of 1/88 Hz")

    coarse = 300.0
    vc = v[:: int(coarse / STEP)]
    dv = max_dv_in_window(vc, int(T_CHUNK / coarse) + 1, int(3 * 3600 / coarse))
    print(f"\nChunk of {T_CHUNK:.0f} s: max line-of-sight dv = {dv:.0f} m/s (analytic bound {dv_chunk:.0f} m/s)")
    print(f"  df_chunk = {F_MAX * dv / C:.2e} Hz (analytic {F_MAX * dv_chunk / C:.2e} Hz) "
          f"= {F_MAX * dv / C * 88:.2f} of the 88 s bin width")


if __name__ == "__main__":
    main()
