# DiffusionGuidanceAdapter

## Goal

Convert existing model output into deterministic advisory placement guidance while preventing diffusion predictions from changing legal dimensions, anchors, or hard feasibility.

## Inputs

- `doc/proposal.md`: Diffusion output should guide order and quality only, never serve as a source of legality.
- `doc/high-level-design.md`: Adapter outputs stable block order, predicted centers, and optional candidate preferences.
- `doc/detailed-design.md`: Module contract for `build(problem, model, device) -> Guidance` and fallback guidance behavior.
- `doc/test-plan.md`: Planned checks for stable finite predictions, non-finite fallback, missing checkpoint behavior, and advisory-only dimensions.

## Write Scope

Edit the model-output interpretation path in `my_optimizer.py` only; do not change model architecture, training code, checkpoint handling policy, or evaluator files.

## Read Scope

Inspect `my_optimizer.py` model initialization, checkpoint loading, current Edge-GNN plus DiT invocation, predicted coordinate handling, and any CUDA cleanup/error handling.

## Dependencies

Depends on `HardConstraintNormalizer` metadata. `AnchorAwareLegalizer` and `FeasibleFallbackPacker` consume this module's `Guidance`.

## Tasks

- [x] Define private `Guidance` record or equivalent structure with deterministic `order`, predicted positions, predicted centers, availability flag, and warnings.
- [x] Wrap existing model inference/output parsing so only predicted x/y or center-like values become advisory guidance.
- [x] Ignore predicted dimensions for legality and leave `BlockSpec.width` and `BlockSpec.height` unchanged.
- [x] Exclude preplaced anchors from movable placement order.
- [x] Implement deterministic fallback guidance for missing checkpoints, invalid shapes, exceptions, or non-finite predictions.
- [x] Sort finite predicted movables deterministically by predicted coordinates and block index, filling missing movables with the fallback order.

## Tests and Quality Gates

- [x] Add or run an approved synthetic guidance check with finite predictions and a second check with non-finite predictions.
- [ ] After implementation approval, run `python iccad2026_evaluate.py --validate my_optimizer.py` to confirm missing or weak checkpoints do not break the optimizer API.

## Done When

- [x] Guidance is deterministic for fixed inputs and available even when model output is missing or unusable.
- [x] Model predictions cannot alter normalized dimensions or preplaced anchors.
- [x] Downstream legalizer inputs contain a complete movable order and optional predicted positions.
