import argparse
import json
import math

T_SUN = 4.92549094830932e-6


def t_to_merger_from_mass_and_frequency(mchirp, frequency):
    if mchirp <= 0.0:
        raise ValueError("mchirp must be positive.")
    if frequency <= 0.0:
        raise ValueError("frequency must be positive.")
    return (5.0 / 256.0) * (T_SUN * mchirp) ** (-5.0 / 3.0) * (math.pi * frequency) ** (-8.0 / 3.0)


def shared_time_window(mchirp_min, mchirp_max, flow, fhigh, min_t_to_merger=0.0):
    t_min = max(t_to_merger_from_mass_and_frequency(mchirp_min, fhigh), min_t_to_merger)
    t_max = t_to_merger_from_mass_and_frequency(mchirp_max, flow)
    if t_min > t_max:
        return None
    return t_min, t_max


def geometric_midpoint(value_min, value_max):
    return math.sqrt(value_min * value_max)


def build_mass_groups(mchirp_min, mchirp_max, flow, fhigh, max_frequency_at_t_to_merger=100.0, max_groups=None, min_t_to_merger=0.0, tol=1e-7):
    if mchirp_min <= 0.0 or mchirp_max <= mchirp_min:
        raise ValueError("Invalid chirp-mass range.")
    if flow <= 0.0 or fhigh <= flow:
        raise ValueError("Invalid frequency band.")
    if max_frequency_at_t_to_merger < flow or max_frequency_at_t_to_merger > fhigh:
        raise ValueError("max_frequency_at_t_to_merger must lie inside the frequency band.")

    groups = []
    current_min = mchirp_min
    while current_min < mchirp_max:
        hi_ok = current_min
        hi_fail = mchirp_max

        if shared_time_window(current_min, hi_fail, flow, fhigh, min_t_to_merger=min_t_to_merger) is not None and t_to_merger_from_mass_and_frequency(current_min, max_frequency_at_t_to_merger) >= min_t_to_merger:
            hi_ok = hi_fail
        else:
            for _ in range(100):
                trial = geometric_midpoint(hi_ok, hi_fail)
                candidate_window = shared_time_window(current_min, trial, flow, fhigh, min_t_to_merger=min_t_to_merger)
                candidate_t_to_merger = max(min_t_to_merger, t_to_merger_from_mass_and_frequency(current_min, max_frequency_at_t_to_merger))
                if candidate_window is None or candidate_t_to_merger > candidate_window[1]:
                    hi_fail = trial
                else:
                    hi_ok = trial
                if hi_fail / hi_ok - 1.0 < tol:
                    break

        window = shared_time_window(current_min, hi_ok, flow, fhigh, min_t_to_merger=min_t_to_merger)
        if window is None:
            break
        representative_t_to_merger = max(min_t_to_merger, t_to_merger_from_mass_and_frequency(current_min, max_frequency_at_t_to_merger))
        if representative_t_to_merger > window[1]:
            break

        groups.append({"mchirp_min": current_min, "mchirp_max": hi_ok, "t_to_merger_s": representative_t_to_merger})

        if max_groups is not None and len(groups) >= max_groups and hi_ok < mchirp_max:
            groups[-1]["mchirp_max"] = mchirp_max
            window = shared_time_window(groups[-1]["mchirp_min"], mchirp_max, flow, fhigh, min_t_to_merger=min_t_to_merger)
            if window is None:
                raise ValueError("Requested max_groups is too small for this mass range and frequency band.")
            representative_t_to_merger = max(min_t_to_merger, t_to_merger_from_mass_and_frequency(groups[-1]["mchirp_min"], max_frequency_at_t_to_merger))
            if representative_t_to_merger > window[1]:
                raise ValueError("Requested max_groups is too small for the frequency-at-t_to_merger condition.")
            groups[-1]["t_to_merger_s"] = representative_t_to_merger
            break

        current_min = hi_ok * (1.0 + tol)

    return groups


def main():
    parser = argparse.ArgumentParser(description="Return valid t_to_merger values for contiguous chirp-mass ranges inside a target GW-frequency band.")
    parser.add_argument("--mchirp-min", type=float, required=True)
    parser.add_argument("--mchirp-max", type=float, required=True)
    parser.add_argument("--flow", type=float, default=61.1)
    parser.add_argument("--fhigh", type=float, default=126.8)
    parser.add_argument("--max-frequency-at-t-to-merger", type=float, default=100.0)
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--min-t-to-merger", type=float, default=32780)
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args()

    groups = build_mass_groups(
        mchirp_min=args.mchirp_min,
        mchirp_max=args.mchirp_max,
        flow=args.flow,
        fhigh=args.fhigh,
        max_frequency_at_t_to_merger=args.max_frequency_at_t_to_merger,
        max_groups=args.max_groups,
        min_t_to_merger=args.min_t_to_merger,
    )
    print(json.dumps(groups, indent=args.indent))


if __name__ == "__main__":
    main()
