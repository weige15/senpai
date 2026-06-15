# SoftConstraintImprover

## Goal

Provide an optional post-legalization improvement stage for boundary, grouping, and MIB soft constraints that never accepts a move unless hard feasibility is preserved.

## Inputs

- `doc/proposal.md`: Boundary moves are soft optimizations and should be attempted only if they preserve non-overlap and all hard constraints.
- `doc/high-level-design.md`: Soft improvements consume a legal candidate and depend on `FeasibilityChecker` for move acceptance.
- `doc/detailed-design.md`: Module contract for `improve(problem, placement, checker) -> placement`, no-op first implementation, and boundary-only incremental path.
- `doc/test-plan.md`: Planned checks for safe/rejected boundary moves, preplaced immutability, no-op preservation, and missing grouping/MIB rules.

## Write Scope

Edit only private soft-improvement helpers and the `MyOptimizer.solve(...)` orchestration point in `my_optimizer.py`; do not change hard-constraint definitions or evaluator scoring.

## Read Scope

Inspect current boundary adjustment logic in `my_optimizer.py`, constraint-column interpretation, and evaluator handling of boundary, grouping, and MIB soft constraints.

## Dependencies

Depends on `HardConstraintNormalizer` soft annotations and `FeasibilityChecker`. Should run only after `AnchorAwareLegalizer` has produced a candidate placement.

## Tasks

- [x] Implement `improve(...)` as a hard-feasibility-preserving no-op first if boundary/grouping/MIB move rules are not yet clear.
- [x] If boundary improvement is implemented, generate translations for non-preplaced blocks with boundary masks using the current bounding box.
- [x] Reject any candidate move that changes dimensions, moves a preplaced block, or overlaps another block.
- [x] Run `FeasibilityChecker.check(...)` on the full placement before accepting any soft move.
- [x] Leave grouping and MIB transformations disabled or explicitly deferred until their exact legality-preserving move rules are defined.

## Tests and Quality Gates

- [x] Add or run an approved boundary-conflict case where a boundary move would overlap another block and must be rejected.
- [x] Add or run an approved check confirming a no-op implementation preserves the legalizer placement exactly.
- [x] After implementation approval, run `python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0`.

## Done When

- [x] Soft improvement cannot turn a hard-feasible placement into an infeasible placement.
- [x] Preplaced anchors remain unchanged through the soft-improvement stage.
- [x] Boundary/grouping/MIB behavior is either safely implemented with checker acceptance or explicitly no-op/deferred.
