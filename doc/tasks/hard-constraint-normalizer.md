# HardConstraintNormalizer

## Goal

Implement the first private normalization stage for `MyOptimizer.solve(...)`, deriving evaluator-safe block metadata before diffusion guidance or placement decisions are used.

## Inputs

- `doc/proposal.md`: Normalize dimensions before layout decisions; preserve fixed/preplaced dimensions and preplaced coordinates exactly.
- `doc/high-level-design.md`: `HardConstraintNormalizer` owns legal dimensions, immutability flags, anchors, and soft annotations.
- `doc/detailed-design.md`: Module contract for `normalize(...) -> NormalizedProblem`, `BlockSpec`, target-position interpretation, and `O(n)` normalization.
- `doc/test-plan.md`: Planned checks for soft area preservation, fixed-shape dimensions, preplaced metadata, and malformed target fields.

## Write Scope

Create or edit only private helper classes, dataclasses, functions, and narrowly related orchestration inside `my_optimizer.py`; add a small local test harness only if explicitly approved in the implementation phase.

## Read Scope

Inspect `my_optimizer.py` around `MyOptimizer.solve(...)`, existing dimension derivation, target-position use, and any current post-pack correction logic; inspect `iccad2026_evaluate.py` hard-constraint checks before matching tolerances.

## Dependencies

None for implementation ordering. Later module tasks depend on the normalized metadata contract from this task.

## Tasks

- [x] Define private `BlockSpec` and `NormalizedProblem` records or equivalent structures with the fields from `doc/detailed-design.md`.
- [x] Implement `normalize(...)` to classify fixed-shape, preplaced, and movable blocks from `constraints` and `target_positions`.
- [x] Preserve fixed/preplaced dimensions exactly from `target_positions[i, 2:4]` and preplaced coordinates exactly from `target_positions[i, 0:2]`.
- [x] Derive positive area-preserving dimensions for movable soft blocks, initially using `sqrt(area) x sqrt(area)`.
- [x] Preserve boundary, MIB, and grouping annotations as soft metadata without enforcing them as hard constraints.
- [x] Surface missing or malformed fixed/preplaced target metadata for later feasibility rejection rather than silently treating it as legal.

## Tests and Quality Gates

- [x] Add or run an approved synthetic check covering one soft block, one fixed-shape block, and one preplaced block.
- [ ] After implementation approval, run `python iccad2026_evaluate.py --validate my_optimizer.py`.

## Done When

- [x] Every block has normalized dimensions, immutability flags, and soft annotations in block-index order.
- [x] Fixed-shape and preplaced dimensions match target positions exactly, and preplaced coordinates are available as immutable anchor metadata.
- [x] Approved validation or equivalent synthetic checks confirm output tuples remain finite and correctly shaped.
