---
name: pipeline-scientific-cleanup
description: Rigorously audit, clean, and reorganize scientific pipelines intended for international reuse, especially HTCondor-style dependency chains (.sub -> .sh -> .py). Use when users request codebase cleanup, structural organization, reproducibility hardening, performance simplification, or generalization improvements with strict change control over core scientific logic.
---

# Pipeline Scientific Cleanup

## Overview

Apply a two-stage workflow for large scientific pipelines: first audit and simplify code rigorously, then reorganize the full pipeline structure for maintainability and reproducibility.
Protect scientific integrity by requiring user approval before any change to core logic, variables, or behavior.

## Operating Rules

- Start with Stage 1 on every request. Do not jump to restructuring first.
- Treat the current behavior as canonical until the user approves behavior-changing edits.
- Allow structural and cleanliness changes without prior approval only when semantics stay identical.
- Ask for permission before changing variable names, function signatures, algorithms, thresholds, data schema, command-line interfaces, or outputs.
- Preserve HTCondor execution dependency: `.sub -> .sh -> .py`.
- Keep recommendations evidence-based and reproducibility-oriented.

## Stage 1: Rigorous Audit And Cleanup

Perform a code-first audit and pause for approval before sensitive edits.

1. Inventory files and dependency flow.
2. Detect unused imports, variables, functions, duplicated blocks, dead code paths, obsolete outputs, and redundant scripts.
3. Flag package-level issues:
   - missing or implicit dependencies
   - unnecessary heavyweight dependencies
   - version pinning/reproducibility risks
4. Identify simplification and speed opportunities that preserve behavior:
   - remove redundant I/O
   - collapse duplicated logic
   - isolate pure helper modules
   - reduce repeated parsing and conversions
5. Produce findings in severity order with concrete file references.
6. Stop and consult the user before any behavior-affecting change.

Use [audit-checklist.md](references/audit-checklist.md) for the exact checklist and reporting format.

## Stage 2: Pipeline Organization

After Stage 1 review and user alignment:

1. Reorganize by execution flow and responsibility, not by ad hoc history.
2. Separate orchestration from science code:
   - `condor/` for `.sub`
   - `bin/` for `.sh`
   - `src/` for `.py`
   - `config/` for static parameters and run profiles
   - `outputs/` for generated artifacts (gitignored when needed)
3. Standardize interfaces between stages:
   - explicit CLI arguments
   - documented input/output contracts
   - deterministic file naming
4. Improve usability:
   - one clear entrypoint
   - concise run examples for local and cluster execution
   - minimal required configuration for new users
5. Preserve and document chain compatibility from `.sub` to `.sh` to `.py`.

## Stage 3: Python Deep Cleanup (File-By-File)

After Stage 2, perform a focused cleanup of core `.py` files one at a time.

1. Work in strict sequence: inspect one file, propose changes, wait for approval, then implement.
2. Do not batch-edit multiple Python files unless the user explicitly asks for it.
3. For each file, optimize for:
   - clear function structure
   - concise pipeline-style docstrings (`Args`, `Returns`, optional `Raises`)
   - removal of dead/legacy code and noisy comments
   - simple, efficient implementations without changing scientific intent
4. Keep behavior stable by default. If a proposed change can affect outputs, thresholds, fitting logic, selection logic, or numerical behavior, stop and request explicit approval.
5. Prefer local variable-name consistency inside the file (clear, stable naming), while preserving external interfaces unless approved otherwise.
6. Fail fast on invalid runtime prerequisites (missing required files/inputs) when silent fallback could bias scientific results.
7. After each file update, run lightweight validation (lint/static checks and syntax checks) and report results before moving to the next file.

## Consultation Protocol

Consult the user before implementing any of the following:

- variable renaming affecting interfaces
- parameter default changes
- algorithmic modifications
- statistical or scientific interpretation changes
- output format/schema changes
- dependency additions/removals that affect runtime behavior

When consulting, provide:
1. Proposed change
2. Rationale (speed/simplicity/cleanliness/generalization)
3. Risk level
4. Backward-compatibility impact
5. Minimal diff plan

## Output Expectations

For each engagement, output:

1. Stage 1 findings (issues and evidence)
2. Proposed Stage 1 safe fixes (no behavior change)
3. Explicit approval gate for sensitive changes
4. Stage 2 organization plan
5. Implemented structural updates and validation results

## References

- [audit-checklist.md](references/audit-checklist.md): issue taxonomy, review order, and consultation template.
