"""Reading O3 frames and writing the frame caches consumed by lalpulsar_MakeSFTs."""

from pathlib import Path

from pbh_viterbi.config import CHANNEL, FRAME_LENGTH, IFO, NUM_FRAMES


def raw_frame_name(gps_start, ifo=IFO, frame_length=FRAME_LENGTH):
    """File name of one downloaded and resampled O3b frame."""
    return f"{ifo[0]}-{ifo}_GWOSC_O3b_4KHZ_R1-{gps_start}-{frame_length}_resampled_512HZ.gwf"


def frame_start_times(t_start, num_frames=NUM_FRAMES, frame_length=FRAME_LENGTH):
    """GPS start time of every frame in a pack."""
    return [t_start + i * frame_length for i in range(num_frames)]


def read_pack_frames(pack_dir, t_start, channel=CHANNEL, ifo=IFO, num_frames=NUM_FRAMES,
                     frame_length=FRAME_LENGTH):
    """Read the raw strain frames of one pack as a list of pycbc TimeSeries."""
    from pycbc import frame as pycbc_frame

    segments = []
    for start in frame_start_times(t_start, num_frames, frame_length):
        path = Path(pack_dir) / raw_frame_name(start, ifo, frame_length)
        segments.append(
            pycbc_frame.read_frame(str(path), channel, start_time=start, end_time=start + frame_length)
        )
    return segments


def _write_framecache(cache_path, entries, ifo):
    with open(cache_path, "w", encoding="utf-8") as handle:
        for gps_start, frame_length, label, frame_dir in entries:
            handle.write(f"{ifo[0]} {label} {gps_start} {frame_length} file://localhost{frame_dir}/{label}.gwf\n")
    return str(cache_path)


def write_raw_framecache(cache_path, pack_dir, t_start, ifo=IFO, num_frames=NUM_FRAMES,
                         frame_length=FRAME_LENGTH):
    """Write a frame cache pointing at the raw (noise-only) frames of a pack."""
    pack_dir = Path(pack_dir).resolve()
    entries = [
        (start, frame_length, raw_frame_name(start, ifo, frame_length)[: -len(".gwf")], pack_dir)
        for start in frame_start_times(t_start, num_frames, frame_length)
    ]
    return _write_framecache(cache_path, entries, ifo)


def injected_frame_label(mchirp, distance, coal_time, gps_start, ifo=IFO, frame_length=FRAME_LENGTH):
    """Label (file stem) of one frame with an injected signal."""
    return f"{ifo}_O3b_mc_{mchirp:.0e}_dL_{distance:.3f}_tc_{int(coal_time)}_{gps_start}-{frame_length}"


def write_injected_framecache(cache_path, frame_dir, mchirp, distance, coal_time, t_start, ifo=IFO,
                              num_frames=NUM_FRAMES, frame_length=FRAME_LENGTH):
    """Write a frame cache pointing at frames produced by injection.inject_signal."""
    frame_dir = Path(frame_dir).resolve()
    entries = [
        (start, frame_length, injected_frame_label(mchirp, distance, coal_time, start, ifo, frame_length), frame_dir)
        for start in frame_start_times(t_start, num_frames, frame_length)
    ]
    return _write_framecache(cache_path, entries, ifo)
