# FeasibilityChecker

## Goal

Add evaluator-aligned hard-feasibility checks for final placement acceptance, fallback routing, and soft-move acceptance.

## Inputs

- `doc/proposal.md`: Final hard check must cover overlaps, soft-block area, fixed-shape dimensions, and preplaced position/dimensions.
- `doc/high-level-design.md`: Checker determines whether optimized output is returned or fallback is invoked.
- `doc/detailed-design.md`: Module contract for `check(positions, problem) -> FeasibilityReport`, overlap threshold `1e-6`, fixed/preplaced tolerance `1e-4`, and soft area tolerance `1%`.
- `doc/test-plan.md`: Planned checker tests for touching rectangles, positive overlap, area tolerance, fixed/preplaced mismatch, and malformed tuples.

## Write Scope

Edit private checker helpers and final acceptance orchestration inside `my_optimizer.py`; do not edit `iccad2026_evaluate.py` or contest data.

## Read Scope

Inspect `iccad2026_evaluate.py` hard-constraint checks and `my_optimizer.py` output construction to align tuple validation and tolerances.

## Dependencies

Depends on `HardConstraintNormalizer` metadata. `SoftConstraintImprover`, `FeasibleFallbackPacker`, and final `solve()` orchestration depend on this checker.

## Tasks

- [x] Define private `FeasibilityReport` or equivalent result with feasible flag, violation counts, and diagnostic messages.
- [x] Validate placement length, finite numeric fields, and positive widths/heights.
- [x] Implement pairwise rectangle overlap checks that allow touching edges and reject positive overlap on both axes greater than `1e-6`.
- [x] Check soft-block realized area against `area_targets` with relative error no greater than `0.01`.
- [x] Check fixed/preplaced dimensions and preplaced coordinates against normalized targets with tolerance `1e-4`.
- [x] Wire checker results into `solve()` so optimized output is returned only when hard-feasible and fallback is invoked otherwise.

## Tests and Quality Gates

- [x] Add or run approved synthetic checker cases for touching rectangles, positive overlap, area tolerance, fixed-shape mismatch, preplaced coordinate mismatch, and malformed tuples.
- [x] After implementation approval, run `python iccad2026_evaluate.py --validate my_optimizer.py`.
- [x] After implementation approval, run `python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0`.

## Done When

- [x] Checker semantics match the documented evaluator tolerances for all hard-constraint categories.
- [x] Optimized and soft-improved placements are rejected before return when any hard violation exists.
- [x] Checker diagnostics are usable for fallback routing and regression fixture creation.
