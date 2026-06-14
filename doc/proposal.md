# Proposal: Hard-Constraint-Safe Legalizer for Diffusion Floorplanning

## Objective

Redesign the post-diffusion legalization path in `my_optimizer.py` so the optimizer preserves all contest hard constraints after the diffusion model produces an initial layout.

The user-provided implementation objective is concrete: the current legalizer violates hard constraints after the diffusion model. The lab-server training dataset is under `data_lite/worker_*`; this path is user-provided context and was not locally verified from this workspace. There is no separate ML validation split inside `data_lite` by default; the loader treats it as one large training dataset.

## Source Inputs

- `doc/problem-brief.md`: contest objective, optimizer API, hard/soft constraints, scoring, and validation expectations.
- `doc/repo-map.md`: current repository structure, main files, missing dependency/config context, and current optimizer/evaluator roles.
- `doc/quality-gates.md`: discovered validation/evaluator commands and missing lint/unit/type-check gates.
- `my_optimizer.py`: narrowly inspected around the diffusion model, B*-tree legalizer, and `solve()` flow.
- `iccad2026_evaluate.py`: narrowly inspected around the optimizer API, evaluator call path, hard-constraint checks, and validation command behavior.
- User-provided dataset clarification: `data_lite` is the lab-server training dataset directory, while `LiteTensorDataTest/` is the public local validation/evaluation set of 100 cases despite the README calling it validation and the folder/code using "Test".

## Current Project State

The optimizer currently uses a hybrid ML-plus-legalizer flow in `my_optimizer.py`:

- `MyOptimizer.solve(...)` derives legal widths/heights first: fixed/preplaced blocks use `target_positions` dimensions, and soft blocks use square dimensions from `sqrt(area_target)`.
- The Edge-GNN + DiT model predicts offsets from an initial all-zero placement.
- `BStarTreeLegalizer` builds a B*-tree order from diffusion-predicted coordinates and packs blocks with a contour method.
- After packing, a final correction step overwrites preplaced coordinates and moves boundary-constrained blocks to the current bounding-box edge.

The main risk in that flow is ordering: post-pack coordinate overwrites can invalidate the earlier non-overlap guarantee. Preplaced blocks and boundary-adjusted blocks can be moved into already packed blocks after legalization. Fixed/preplaced dimensions are mostly handled before packing, but preplaced locations are not treated as immovable obstacles during packing.

Dataset state is split across two roles:

- `data_lite/worker_*` on the lab server is the training corpus. By default, it is loaded as one large training dataset and should not be assumed to contain a separate ML validation split.
- `LiteTensorDataTest/` in this workspace is the public local validation/evaluation set with 100 cases. The naming is confusing because the contest README calls this set "Validation", while the folder and code use "Test".

## Problem Summary

The contest makes hard feasibility more important than local quality improvements because any hard-constraint violation assigns cost `10.0`. Hard constraints are:

- no overlaps;
- soft-block area within 1% of target;
- fixed-shape block dimensions exactly preserved;
- preplaced block position and dimensions exactly preserved.

Boundary, grouping, and MIB constraints are soft constraints. They should influence optimization where practical, but they must not be enforced in a way that reintroduces hard infeasibility.

## Constraints

- Preserve the `FloorplanOptimizer.solve(...)` API and return one `(x, y, width, height)` tuple per block.
- Do not edit contest validation data.
- Do not change `iccad2026_evaluate.py` scoring behavior unless explicitly requested.
- Keep changes focused on `my_optimizer.py` and any narrowly needed documentation.
- Avoid requiring new dependencies; no dependency manifest is present locally.
- Evaluation and training commands require explicit approval before execution.
- The lab-server dataset path `data_lite` is available by user report but not verified in this workspace.
- Do not evaluate implementation quality by splitting `data_lite` unless a separate split policy is explicitly designed later; use `LiteTensorDataTest/` for public local validation/evaluation.

## Proposed Approach

Make legalization feasibility-first and anchor-aware:

1. Normalize dimensions before any layout decision.
   - Fixed and preplaced blocks keep exact target dimensions.
   - Soft blocks use dimensions whose product is exactly the target area, with a small tolerance margin only if needed for floating-point stability.

2. Treat preplaced blocks as immutable anchors throughout legalization.
   - Preplaced blocks should be inserted into the placement first at exact target coordinates.
   - Other blocks must be packed around them, not packed first and then overwritten.

3. Use diffusion output only as a quality guide, not as a source of legality.
   - The diffusion model should influence block order, orientation/aspect choices if later allowed, and local placement preferences.
   - Final coordinates must come from a hard-constraint-aware legalizer and repair pass.

4. Replace post-pack hard moves with checked transformations.
   - Preplaced coordinates should never be changed after anchor insertion.
   - Boundary moves are soft-constraint optimizations and should be attempted only if the move preserves non-overlap and all hard constraints.

5. Add a deterministic final feasibility repair/check stage.
   - Before returning, run overlap detection, area/dimension checks, and preplaced checks using the same semantics as the evaluator.
   - If any hard violation remains, fall back to a conservative feasible packing strategy rather than returning an infeasible layout.

## Algorithm Strategy

Baseline method:

- A deterministic feasible packer that ignores diffusion coordinates except for a stable block order.
- Place immutable preplaced blocks first.
- Place remaining blocks in rows/strips or contour positions that avoid all already placed rectangles.
- Preserve all fixed/preplaced dimensions and all soft-block areas.
- This baseline is expected to be lower quality but should avoid `10.0` infeasible costs.

Intended optimized method:

- Use diffusion-predicted coordinates to sort blocks and select candidate placement regions.
- Use an obstacle-aware contour or skyline packer where preplaced blocks are already occupied obstacles.
- For each non-preplaced block, choose the feasible candidate that balances:
  - distance to diffusion-predicted center;
  - HPWL contribution against connected blocks/pins already placed;
  - compact bounding-box growth;
  - optional soft-constraint gains for boundary/grouping/MIB when they do not threaten hard feasibility.
- After all blocks are placed, run local compaction and boundary/grouping improvements only through legality-preserving moves.
- If optimized placement fails a final hard check, fall back to the deterministic feasible packer for that case.

Correctness strategy:

- Maintain invariants during placement: exact immutable dimensions, exact preplaced coordinates, positive widths/heights, and no rectangle intersections.
- Never perform a coordinate overwrite after legalization unless it is followed by a hard-feasibility check and repair.
- Keep boundary enforcement subordinate to overlap-free feasibility because boundary is soft and overlap is hard.
- Use evaluator-equivalent checks as the acceptance oracle for returned layouts.

Performance strategy:

- Keep the legalizer deterministic and near-quadratic or better for 21-120 block validation/test cases.
- Prefer simple candidate generation over expensive global optimization because the final score heavily penalizes infeasibility and local runtime still matters on the official leaderboard.
- Use diffusion output for ordering/quality so the model remains useful, but avoid relying on it for constraint satisfaction.
- Prioritize larger cases because total score is weighted by `exp(n / 12)`.

## Alternatives Considered

- Keep current B*-tree packer and only add a final overlap repair: lowest implementation cost, but preplaced anchors and boundary post-moves can keep fighting the packer unless the repair becomes a second legalizer.
- Disable diffusion and use only a deterministic strip packer: strongest feasibility baseline, but likely poor HPWL and area gap.
- Use simulated annealing from the evaluator as the main solver: source-supported as a baseline concept, but likely slower and less predictable without a focused implementation plan.
- Treat boundary constraints as hard during legalization: not aligned with the contest scoring model; boundary violations are soft, while overlap and immutable preplaced/fixed constraints are hard.

## Module Candidates

- `HardConstraintNormalizer`: proposal-level role for deriving legal dimensions and immutable block metadata from `area_targets`, `constraints`, and `target_positions`.
- `AnchorAwareLegalizer`: proposal-level role for obstacle-aware placement around preplaced blocks.
- `FeasibleFallbackPacker`: proposal-level role for guaranteed conservative placement if optimized legalization fails.
- `FeasibilityChecker`: proposal-level role for evaluator-aligned hard checks before returning.
- `SoftConstraintImprover`: optional later role for legality-preserving boundary/grouping/MIB refinements.

These names are module candidates, not required final class names.

## Milestones

1. Document the exact hard-constraint failure mode with one approved evaluator run or saved failing case.
2. Add an internal feasibility-check plan aligned with `iccad2026_evaluate.py`.
3. Replace post-pack preplaced overwrites with preplaced-anchor-aware packing.
4. Make boundary adjustment legality-preserving or disable it when it would create overlap.
5. Add a conservative fallback packer for any case that fails final hard checks.
6. Validate on one case, then all 100 local validation cases.
7. Use lab-server `data_lite` for training experiments only after confirming the expected loader path and command-line `--data-path` behavior.

## Validation Plan

Use the discovered quality gates from `doc/quality-gates.md`, with approval before each command:

- `python iccad2026_evaluate.py --validate my_optimizer.py`
  - Expected: optimizer module loads, returns correct format, and handles dummy data.
- `python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0`
  - Expected: no hard violations on a single validation case; inspect feasible flag and score.
- `python iccad2026_evaluate.py --evaluate my_optimizer.py`
  - Expected: all 100 validation cases should be feasible before optimizing score.
- `python iccad2026_evaluate.py --evaluate my_optimizer.py --save-solutions`
  - Optional comparison artifact for saved placements.
- `python iccad2026_evaluate.py --score my_optimizer_solutions.json`
  - Optional re-score after saving solutions.

Validation should be interpreted against `LiteTensorDataTest/`, the public 100-case local validation/evaluation set. Training experiments may use `data_lite`, but that directory should not be treated as a held-out validation source unless a deliberate split is added later.

Additional proposal-level checks:

- Tiny synthetic placements with one preplaced obstacle and several soft blocks should verify that non-preplaced blocks avoid the anchor.
- A synthetic fixed-shape case should verify exact dimensions are preserved.
- A boundary-constrained case should verify boundary moves are skipped or repaired when they would overlap.

No unit-test command exists yet, so these synthetic checks would need either a small local harness or manual evaluator-compatible inspection in a later implementation/design step.

## Risks and Tradeoffs

- A conservative legalizer may increase bounding-box area and HPWL, but that is preferable to infeasible `10.0` costs.
- Obstacle-aware placement around arbitrary preplaced blocks is more complex than the current simple contour packer.
- The current workspace may be partial; evaluator imports reference original FloorSet helper modules that may need the full runtime environment.
- The lab-server `data_lite` structure is training-oriented and may need explicit loader/path handling; it should not be confused with the public `LiteTensorDataTest/` validation/evaluation set.
- If diffusion weights are missing, the model path falls back to initialized weights, so the legalizer must be robust without useful ML predictions.

## Assumptions

- The correct implementation target is `my_optimizer.py`.
- The evaluator's hard-constraint checks in `iccad2026_evaluate.py` are authoritative for local validation.
- `target_positions` is available during evaluation for fixed/preplaced blocks, as shown in the evaluator call path.
- Boundary constraints should remain soft constraints, not hard feasibility requirements.
- The first implementation success criterion is feasibility across validation cases, then score improvement.
- The lab-server dataset under `data_lite` is accessible in the user's lab environment, but not necessarily in this local workspace.
- `LiteTensorDataTest/` is the public local 100-case validation/evaluation set, even though the contest naming alternates between "Validation" and "Test".

## Open Questions

- What exact hard violation is currently most common: overlap, area tolerance, fixed/preplaced dimension mismatch, or preplaced position mismatch?
- What exact loader path or `--data-path` should be used for lab-server `data_lite` training runs?
- Should the first implementation prioritize a guaranteed feasible fallback even if HPWL/area score worsens?
- Are trained checkpoints expected to be present during final local evaluation, or must the optimizer perform well without checkpoints?
- Is it acceptable to add a small internal validation harness later, given no unit-test framework is currently discovered?
