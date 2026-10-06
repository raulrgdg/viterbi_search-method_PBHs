"""Assignment of the 600-signal injection campaign of the paper across three clusters.

The 108 O3b packs are cycled in blocks of 36: within each block, the first 12
packs went to HPC1 (Condor), the next 12 to HPC2 (Slurm) and the last 12 to
HPC3 (Slurm). Each cluster injected its own 200-signal slice of the 600-signal
grid into every pack it received, so every signal was injected into 36
different noise realisations.

    HPC1  packs 1-12, 37-48, 73-84    signals [0, 200)
    HPC2  packs 13-24, 49-60, 85-96   signals [200, 400)
    HPC3  packs 25-36, 61-72, 97-108  signals [400, 600)
"""

from dataclasses import dataclass

from pbh_viterbi.o3.packs import ALL_PACKS

PACKS_PER_CYCLE = 36
PACKS_PER_CLUSTER_PER_CYCLE = 12

CLUSTERS = (
    {"cluster": "HPC1", "scheduler": "condor", "signal_start": 0, "signal_end": 200},
    {"cluster": "HPC2", "scheduler": "slurm", "signal_start": 200, "signal_end": 400},
    {"cluster": "HPC3", "scheduler": "slurm", "signal_start": 400, "signal_end": 600},
)


@dataclass(frozen=True)
class Assignment:
    pack: int
    cluster: str
    scheduler: str
    signal_start: int
    signal_end: int


def assignment_for_pack(pack):
    """Cluster and signal slice assigned to one pack."""
    if pack not in ALL_PACKS:
        raise ValueError(f"pack must be in [1, {max(ALL_PACKS)}], got {pack}")
    slot = ((pack - 1) % PACKS_PER_CYCLE) // PACKS_PER_CLUSTER_PER_CYCLE
    return Assignment(pack=pack, **CLUSTERS[slot])


def packs_for_cluster(cluster):
    """Packs assigned to one cluster (HPC1, HPC2 or HPC3)."""
    packs = [p for p in ALL_PACKS if assignment_for_pack(p).cluster == cluster.upper()]
    if not packs:
        raise ValueError(f"Unknown cluster {cluster!r}; expected one of HPC1, HPC2, HPC3.")
    return packs


if __name__ == "__main__":
    for slot in CLUSTERS:
        packs = packs_for_cluster(slot["cluster"])
        print(f"{slot['cluster']} ({slot['scheduler']}): signals [{slot['signal_start']}, {slot['signal_end']}) "
              f"packs {' '.join(map(str, packs))}")
