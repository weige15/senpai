# AnchorAwareLegalizer

## Goal

Replace post-pack hard-coordinate overwrites with an obstacle-aware placement stage that inserts immutable anchors first and packs movable blocks around them.

## Inputs

- `doc/proposal.md`: Preplaced blocks must be immutable anchors throughout legalization; final coordinates must come from hard-aware placement.
- `doc/high-level-design.md`: `AnchorAwareLegalizer` owns optimized hard-aware coordinates and consumes normalized metadata plus advisory guidance.
- `doc/detailed-design.md`: Module contract for `legalize(problem, guidance) -> PlacementState`, candidate generation, overlap filtering, and deterministic tie-breaking.
- `doc/test-plan.md`: Planned anchor, touching-rectangle, positive-overlap, missing-guidance, and deterministic tie-break checks.

## Write Scope

Edit only private legalizer helpers and `MyOptimizer.solve(...)` orchestration in `my_optimizer.py`; remove or bypass current hard-constraint-breaking post-pack preplaced/boundary overwrites only as needed for this module.

## Read Scope

Inspect `my_optimizer.py` around `BStarTreeLegalizer`, current B*-tree ordering, contour packing, final preplaced coordinate restoration, and boundary adjustment logic.

## Dependencies

Depends on `HardConstraintNormalizer` for dimensions and anchors, and `DiffusionGuidanceAdapter` for advisory order/preferences. Uses overlap semantics that must match `FeasibilityChecker`.

## Tasks

- [x] Define private `PlacementState` or equivalent state with block-indexed positions, occupied rectangles, and placement order.
- [x] Initialize legalizer state by inserting every preplaced block at exact normalized coordinates and dimensions.
- [x] Generate a bounded deterministic candidate set for each movable block using occupied-rectangle edges, bounding-box points, and optional predicted positions.
- [x] Reject candidates that overlap any occupied rectangle while allowing edge-touching rectangles.
- [x] Score feasible candidates using deterministic terms such as predicted-center distance, bounding-box growth, and optional local HPWL where already placed peers are available.
- [x] Return a failed or incomplete state when no candidate can be placed so the orchestrator can invoke fallback.

## Tests and Quality Gates

- [x] Add or run an approved synthetic case with one preplaced anchor plus one movable block and confirm the movable avoids the anchor.
- [x] Add or run approved checks that touching rectangles pass and positive-area intersections are rejected.
- [x] After implementation approval, run `python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0`.

## Done When

- [x] Preplaced anchors are never moved after normalization.
- [x] Movable blocks are placed only at non-overlapping candidate positions around anchors and previously placed blocks.
- [x] Single-case evaluator validation reports no hard-constraint violations after approved execution.
