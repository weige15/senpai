# FeasibleFallbackPacker

## Goal

Provide a deterministic conservative packing path that returns a hard-feasible placement when the optimized legalizer or soft improver fails final checks.

## Inputs

- `doc/proposal.md`: A conservative fallback is preferred over returning an invalid optimized result because infeasible layouts receive cost `10.0`.
- `doc/high-level-design.md`: Fallback consumes normalized metadata and immutable anchors, then receives final `FeasibilityChecker` acceptance.
- `doc/detailed-design.md`: Module contract for `pack(problem, guidance=None) -> list[PlacementTuple]`, row/strip strategy, and anchor-first placement.
- `doc/test-plan.md`: Planned fallback checks for soft-only packing, anchor preservation, missing guidance, final checker acceptance, and overlapping-anchor failure reporting.

## Write Scope

Edit private fallback packing helpers and fallback orchestration inside `my_optimizer.py`; do not modify training scripts, validation data, evaluator scoring, or checkpoint files.

## Read Scope

Inspect existing deterministic packing utilities in `my_optimizer.py`, current fallback behavior if any, and overlap helpers shared with the legalizer/checker.

## Dependencies

Depends on `HardConstraintNormalizer` for legal dimensions and anchors, and `FeasibilityChecker` for final acceptance. May consume `DiffusionGuidanceAdapter` order only when it cannot compromise feasibility.

## Tasks

- [x] Implement `pack(...)` to insert preplaced anchors exactly before placing any movable blocks.
- [x] Choose deterministic movable order from guidance order or a stable fallback order.
- [x] Implement a row/strip cursor strategy that advances around occupied rectangles and expands the layout instead of overlapping anchors.
- [x] Keep dimensions unchanged for every block and output placements in block-index order.
- [x] Route optimized-path failures through fallback and run `FeasibilityChecker.check(...)` before returning fallback output.
- [x] Report or preserve diagnostics when fallback cannot satisfy contradictory hard inputs such as overlapping immutable anchors.

## Tests and Quality Gates

- [x] Add or run approved synthetic fallback cases for soft-only blocks, one anchor plus movables, missing guidance, and overlapping immutable anchors.
- [x] After implementation approval, run `python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0`.
- [x] After implementation approval, run `python iccad2026_evaluate.py --evaluate my_optimizer.py` and target 100/100 feasible local validation cases before score tuning.

## Done When

- [x] Normal non-contradictory inputs have a conservative fallback path that passes `FeasibilityChecker`.
- [x] Fallback never moves anchors, changes dimensions, or relies on diffusion output for legality.
- [x] Full local validation is ready to run with approval once the optimizer implementation is complete.
