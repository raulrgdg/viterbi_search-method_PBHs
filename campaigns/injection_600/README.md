# 600-signal injection campaign automation

These helpers submit the injected-search campaign one pack at a time.

The Python entrypoint derives the correct signal slice from the pack id:

- HPC1: packs `1-12,37-48,73-84`, signals `[0,200)`, Condor.
- HPC2: packs `13-24,49-60,85-96`, signals `[200,400)`, Slurm.
- HPC3: packs `25-36,61-72,97-108`, signals `[400,600)`, Slurm.

Run all commands from the repository root.

## HPC1 / Condor

```bash
bash campaigns/injection_600/submit_condor_chain.sh
```

This submits one Condor pack job, waits for the full 200-job Condor cluster
to finish, then submits the next pack.

Override the pack list if needed:

```bash
PACKS="1 2 3" bash campaigns/injection_600/submit_condor_chain.sh
```

## HPC2 / Slurm

```bash
CLUSTER=HPC2 bash campaigns/injection_600/submit_slurm_chain.sh
```

This submits one Slurm array per pack. Each array depends on the previous
pack with `afterok`, so the next pack starts only if the previous pack
finishes successfully.

## HPC3 / Slurm

Make sure the local O3 data are visible at:

```text
data/raw/o3/O3b-packX
```

For example, if the staged data live at `/home/uib/uib729081/O3-data`:

```bash
mkdir -p data/raw
ln -s /home/uib/uib729081/O3-data data/raw/o3
```

Then submit:

```bash
CLUSTER=HPC3 bash campaigns/injection_600/submit_slurm_chain.sh
```

Override the dependency type if you want the chain to continue after failures:

```bash
CLUSTER=HPC3 DEPENDENCY_TYPE=afterany bash campaigns/injection_600/submit_slurm_chain.sh
```
