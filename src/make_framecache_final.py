import os


def generate_framecache(input_dir, mchirp, distance, det, num_frames, frame_length, t_start, coal_time):
    """Create a framecache file for injected strain frames."""
    coal_time = int(coal_time)
    framecache_path = os.path.join(input_dir, f"framecache_inject_data_mc-{mchirp:.0e}_dl-{distance:.3f}")

    with open(framecache_path, "w+") as f:
        for i in range(num_frames):
            time = t_start + frame_length * i
            label = f"{det}1_O3b_mc_{mchirp:.0e}_dL_{distance:.3f}_tc_{coal_time}_{time}-{frame_length}"
            frametype = label
            filename = f"file://localhost{input_dir}/{label}.gwf"
            f.write(f"{det} {frametype} {time} {frame_length} {filename}\n")

    print("Done: make framecache.")
    return framecache_path


def generate_framecache_raw_strain(input_dir, det, num_frames, frame_length, t_start):
    """Create a framecache file for downloaded raw strain frames."""
    framecache_path = os.path.join(input_dir, "framecache_raw_strain_512HZ")

    with open(framecache_path, "w+") as f:
        for i in range(num_frames):
            time = t_start + frame_length * i
            label = f"H-{det}1_GWOSC_O3b_4KHZ_R1-{time}-{frame_length}_resampled_512HZ"
            frametype = label
            filename = f"file://localhost{input_dir}/{label}.gwf"
            f.write(f"{det} {frametype} {time} {frame_length} {filename}\n")

    print("Done: make framecache.")
    return framecache_path
