# Audit Checklist

## Review Order

1. Map orchestration path (`.sub -> .sh -> .py`).
2. List runtime dependencies and package imports.
3. Scan dead/unused elements.
4. Evaluate simplification and performance opportunities.
5. Evaluate reproducibility and interface consistency.
6. Stop for user approval when a sensitive change is needed.

## Issue Taxonomy

- `P0` correctness risk: likely wrong results, broken pipeline, or data corruption.
- `P1` reproducibility risk: non-determinism, implicit environment assumptions, missing version controls.
- `P2` maintainability/performance issue: dead code, duplicated logic, unnecessary heavy operations.
- `P3` style/documentation issue: readability friction without immediate runtime impact.

## What To Detect In Stage 1

- Unused imports, variables, constants, functions, and files.
- Duplicate helper logic across scripts.
- Hidden coupling through hardcoded paths and magic values.
- Repeated expensive I/O or parsing work.
- Shell scripts with missing `set -euo pipefail` or brittle assumptions.
- Condor submission files with duplicated parameters or weak portability.

## Sensitive Changes Requiring Approval

- Any variable renaming visible outside local scope.
- Function signature changes.
- Algorithm/statistics changes.
- Default threshold/parameter changes.
- Output schema/file naming contract changes.
- Dependency changes that alter runtime behavior.

## Consultation Template

Use this exact structure before applying sensitive edits:

1. `Change`: one-line summary.
2. `Reason`: why this improves speed, simplicity, cleanliness, or generalization.
3. `Risk`: low/medium/high.
4. `Compatibility`: impact on existing workflows and HTCondor chain.
5. `Minimal plan`: exact files and narrowest safe diff.

## Stage 2 Structural Organization Target

- `condor/`: `.sub` files and condor-specific config.
- `bin/`: shell wrappers and execution scripts.
- `src/`: python modules and scientific logic.
- `config/`: runtime profiles and parameter files.
- `outputs/`: generated artifacts; keep reproducible naming.

Keep interfaces explicit, documented, and backwards-compatible unless the user approves a breaking change.
