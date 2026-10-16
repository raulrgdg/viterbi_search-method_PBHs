"""Layout of the 600-signal injection campaign.

The 108 O3b packs are cycled in blocks of 36: within each block, the first 12
packs inject the signals [0, 200) of the population, the next 12 the signals
[200, 400) and the last 12 the signals [400, 600). In the paper each third
ran on a different cluster (HPC1 = LIGO Caltech, HPC2 = IFT Madrid,
HPC3 = MareNostrum 5); with in-memory SFTs any single cluster or machine can
run the whole campaign.

    HPC1  packs 1-12, 37-48, 73-84    signals [0, 200)
    HPC2  packs 13-24, 49-60, 85-96   signals [200, 400)
    HPC3  packs 25-36, 61-72, 97-108  signals [400, 600)

The paper campaign used 90 of the packs (32, 28 and 30 for the three slices,
i.e. every signal in ~30 noise realisations): PAPER_CAMPAIGN_PACKS, the default
campaign of v1.1, so that a re-run is directly comparable. ``campaign_packs("all")`` uses the 108
packs (36 realisations per signal).
"""

from dataclasses import dataclass

from pbh_viterbi.o3.packs import PAPER_PACKS

PACKS_PER_CYCLE = 36
PACKS_PER_CLUSTER_PER_CYCLE = 12

CLUSTERS = (
    {"cluster": "HPC1", "scheduler": "condor", "signal_start": 0, "signal_end": 200},
    {"cluster": "HPC2", "scheduler": "slurm", "signal_start": 200, "signal_end": 400},
    {"cluster": "HPC3", "scheduler": "slurm", "signal_start": 400, "signal_end": 600},
)

# Packs of the final paper campaign (paper_data/search_results/injected_search_O3b_H1_90packs.csv).
PAPER_CAMPAIGN_PACKS = (
    tuple(range(1, 12)) + tuple(range(37, 49)) + tuple(range(73, 82))  # HPC1: 32 packs
    + tuple(range(13, 25)) + tuple(range(49, 61)) + tuple(range(85, 89))  # HPC2: 28 packs
    + tuple(range(25, 37)) + tuple(range(61, 67)) + tuple(range(68, 73)) + tuple(range(97, 104))  # HPC3: 30 packs
)


@dataclass(frozen=True)
class Assignment:
    pack: int
    cluster: str
    scheduler: str
    signal_start: int
    signal_end: int


def assignment_for_pack(pack):
    """Signal slice (and paper cluster) assigned to one pack."""
    if pack not in PAPER_PACKS:
        raise ValueError(f"The injection campaign uses packs 1-{max(PAPER_PACKS)}, got {pack}")
    slot = ((pack - 1) % PACKS_PER_CYCLE) // PACKS_PER_CLUSTER_PER_CYCLE
    return Assignment(pack=pack, **CLUSTERS[slot])


def packs_for_cluster(cluster):
    """Packs assigned to one slice (HPC1, HPC2 or HPC3)."""
    packs = [p for p in PAPER_PACKS if assignment_for_pack(p).cluster == cluster.upper()]
    if not packs:
        raise ValueError(f"Unknown cluster {cluster!r}; expected one of HPC1, HPC2, HPC3.")
    return packs


def campaign_packs(name="paper"):
    """Packs of a campaign: "paper" (the 90 packs of the paper) or "all" (the 108 paper packs).

    Packs 109-141 (added in v1.1) are used for the noise background and the
    noise search only.
    """
    if name == "paper":
        return sorted(PAPER_CAMPAIGN_PACKS)
    if name == "all":
        return list(PAPER_PACKS)
    raise ValueError(f"Unknown campaign {name!r}; expected 'paper' or 'all'.")


if __name__ == "__main__":
    import sys

    print(*campaign_packs(sys.argv[1] if len(sys.argv) > 1 else "paper"))
