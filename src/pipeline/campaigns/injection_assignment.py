from __future__ import annotations

from dataclasses import dataclass


TOTAL_PACKS = 108
PACKS_PER_CYCLE = 36
PACKS_PER_CLUSTER_PER_CYCLE = 12
TOTAL_SIGNALS = 600
SIGNALS_PER_CLUSTER = 200


@dataclass(frozen=True)
class InjectionAssignment:
    """Assignment for one O3 pack in the 600-signal injection campaign."""

    pack_id: int
    cluster: str
    scheduler: str
    signal_start: int
    signal_end: int
    data_mode: str


CLUSTER_SLOTS = (
    {
        "cluster": "HPC1",
        "scheduler": "condor",
        "signal_start": 0,
        "signal_end": 200,
        "data_mode": "online",
    },
    {
        "cluster": "HPC2",
        "scheduler": "slurm",
        "signal_start": 200,
        "signal_end": 400,
        "data_mode": "online",
    },
    {
        "cluster": "HPC3",
        "scheduler": "slurm",
        "signal_start": 400,
        "signal_end": 600,
        "data_mode": "local",
    },
)


def assignment_for_pack(pack_id: int) -> InjectionAssignment:
    """Return the cluster and signal slice assigned to a 1-based pack id."""
    if pack_id < 1 or pack_id > TOTAL_PACKS:
        raise ValueError(f"pack_id must be in [1, {TOTAL_PACKS}], got {pack_id}")

    cycle_offset = (pack_id - 1) % PACKS_PER_CYCLE
    slot_index = cycle_offset // PACKS_PER_CLUSTER_PER_CYCLE
    slot = CLUSTER_SLOTS[slot_index]
    return InjectionAssignment(pack_id=pack_id, **slot)


def assignments_for_cluster(cluster: str) -> list[InjectionAssignment]:
    """Return all pack assignments for one cluster."""
    cluster = cluster.upper()
    assignments = [assignment_for_pack(pack_id) for pack_id in range(1, TOTAL_PACKS + 1)]
    selected = [assignment for assignment in assignments if assignment.cluster == cluster]
    if not selected:
        valid_clusters = ", ".join(slot["cluster"] for slot in CLUSTER_SLOTS)
        raise ValueError(f"Unknown cluster {cluster!r}. Expected one of: {valid_clusters}")
    return selected


def packs_for_cluster(cluster: str) -> list[int]:
    """Return the 1-based O3 pack ids assigned to one cluster."""
    return [assignment.pack_id for assignment in assignments_for_cluster(cluster)]


def validate_campaign_assignments() -> None:
    """Fail if the campaign assignment no longer matches the intended partition."""
    assignments = [assignment_for_pack(pack_id) for pack_id in range(1, TOTAL_PACKS + 1)]
    pack_ids = [assignment.pack_id for assignment in assignments]
    if pack_ids != list(range(1, TOTAL_PACKS + 1)):
        raise ValueError("Campaign assignments do not cover packs 1..108 exactly once.")

    for slot in CLUSTER_SLOTS:
        cluster = slot["cluster"]
        cluster_assignments = [assignment for assignment in assignments if assignment.cluster == cluster]
        if len(cluster_assignments) != PACKS_PER_CYCLE:
            raise ValueError(
                f"{cluster} must receive {PACKS_PER_CYCLE} packs, got {len(cluster_assignments)}"
            )

        for assignment in cluster_assignments:
            if assignment.signal_start != slot["signal_start"] or assignment.signal_end != slot["signal_end"]:
                raise ValueError(f"{cluster} has an inconsistent signal slice for pack {assignment.pack_id}")
            if assignment.signal_end - assignment.signal_start != SIGNALS_PER_CLUSTER:
                raise ValueError(f"{cluster} signal slice must contain {SIGNALS_PER_CLUSTER} signals")

    signal_ranges = sorted(
        (slot["signal_start"], slot["signal_end"])
        for slot in CLUSTER_SLOTS
    )
    if signal_ranges != [(0, 200), (200, 400), (400, 600)]:
        raise ValueError(f"Unexpected signal ranges: {signal_ranges}")
    if signal_ranges[0][0] != 0 or signal_ranges[-1][1] != TOTAL_SIGNALS:
        raise ValueError("Signal ranges do not cover [0, 600).")
