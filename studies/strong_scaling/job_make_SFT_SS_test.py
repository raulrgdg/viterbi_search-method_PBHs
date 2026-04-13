from argparse import ArgumentParser
from pathlib import Path

import htcondor2 as htcondor


def parse_args():
    script_dir = Path(__file__).resolve().parent
    default_project_root = script_dir.parent.parent
    default_run_script = script_dir / "run_make_SFT-SS_test_condor.sh"
    default_framecache = script_dir / "data" / "framecache_raw_strain_512HZ"

    parser = ArgumentParser(description="Submit the strong-scale SFT benchmark to HTCondor.")
    parser.add_argument("--project-root", default=default_project_root, type=Path)
    parser.add_argument("--run-script", default=default_run_script, type=Path)
    parser.add_argument("--framecache", default=default_framecache, type=Path)
    parser.add_argument(
        "--threads",
        nargs="+",
        type=int,
        default=[16, 32, 64, 128, 256, 512],
        help="Thread counts to benchmark.",
    )
    parser.add_argument("--request-cpus", default="8")
    parser.add_argument("--request-memory", default="1GB")
    parser.add_argument("--request-disk", default="5GB")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = args.project_root.resolve()
    run_script = args.run_script.resolve()
    framecache = args.framecache.resolve()
    log_dir = project_root / "studies" / "strong_scaling" / "condor_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = log_dir / "make_sft_ss_test.out"
    stderr_path = log_dir / "make_sft_ss_test.err"
    eventlog_path = log_dir / "make_sft_ss_test.log"

    arguments = " ".join(
        [str(run_script), str(framecache)] + [str(thread_count) for thread_count in args.threads]
    )

    basic_submit = {
        "executable": "/bin/bash",
        "arguments": arguments,
        "output": str(stdout_path),
        "error": str(stderr_path),
        "log": str(eventlog_path),
        "request_cpus": args.request_cpus,
        "request_memory": args.request_memory,
        "request_disk": args.request_disk,
        "getenv": True,
        "initialdir": str(project_root),
        "on_exit_hold": "(ExitBySignal == True) || (ExitCode != 0)",
    }

    common_submit = htcondor.Submit(basic_submit)
    schedd = htcondor.Schedd()
    if hasattr(schedd, "transaction"):
        with schedd.transaction() as txn:
            cluster_id = common_submit.queue(txn)
    else:
        result = schedd.submit(common_submit)
        cluster_obj = result.cluster
        cluster_id = cluster_obj() if callable(cluster_obj) else cluster_obj

    print(f"Submitted cluster_id={cluster_id}")


if __name__ == "__main__":
    main()
