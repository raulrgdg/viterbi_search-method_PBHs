import os
import subprocess


def run_make_sfts_script(
    bash_script_path,
    t_start,
    t_end,
    sft_output_path,
    framecache_path,
    num_threads,
    Tseg,
    fmin,
    Band,
    windowtype,
    channel,
    verbose=False,
):
    """Run the MakeSFTs helper bash script with the expected environment."""
    os.makedirs(sft_output_path, exist_ok=True)
    env_vars = os.environ.copy()
    env_vars.update(
        {
            "t_start": str(t_start),
            "t_end": str(t_end),
            "num_threads": str(num_threads),
            "Tseg": str(Tseg),
            "remainder_mode": "trim",
            "SFTPATH": sft_output_path,
            "framecache": framecache_path,
            "Band": str(Band),
            "fmin": str(fmin),
            "windowtype": windowtype,
            "channel_name": channel,
            "sft_verbose": "1" if verbose else "0",
        }
    )

    try:
        subprocess.run(["bash", bash_script_path], env=env_vars, check=True)
        if verbose:
            print("SFT generation completed successfully.", flush=True)
    except subprocess.CalledProcessError as exc:
        print(f"Error during SFT generation: {exc}", flush=True)
        raise
