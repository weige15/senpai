# Performance Diagnosis

## Purpose

Diagnose why the checkpoint-enabled optimizer remains above the requested target score of `2.0` or below, record the failed recursive-slicing optimization attempt, and identify the next small optimization attempt.

User-provided checkpoint context:

- `iccad2026contest/checkpoints/dit_gnn_step_43299_epoch_11_end.pth`
- File status: present locally, 128 MB.

No optimizer implementation code was changed during this diagnosis.

## Diagnosis Scope

Covered:

- Saved evaluator artifact: `/tmp/senpai_local_relax_full.json`.
- Failed implementation-loop artifact: `/tmp/senpai_slicing_full.json`.
- Comparison artifacts: `/tmp/senpai_aspect_full.json`, `/tmp/senpai_frame_compact_full.json`, `/tmp/senpai_connection_43299_full.json`.
- Current checkpoint inventory under `iccad2026contest/checkpoints/`.
- `iccad2026contest/my_optimizer.py` checkpoint loading, hard-constraint normalization, diffusion guidance, constructive candidate generation, boundary frame compaction, aspect-ratio candidates, candidate selection, and `solve()` orchestration.
- `iccad2026contest/iccad2026_evaluate.py` scoring formula, hard/soft constraint semantics, local runtime handling, and block-count weighting.
- Local validation constraints were read only to recompute soft-violation subcategories for saved positions.
- Planning and quality-gate documents under `doc/`.

Not covered:

- The optimizer was not rerun.
- Training was not run.
- The checkpoint tensor was not loaded or inspected.
- No profiler or memory measurement was run.
- Hidden-test runtime normalization was not measured.
- The trial local-relax and recursive-slicing implementations were both reverted after violating rollback conditions.

## Source Documents Read

- `doc/problem-brief.md`
- `doc/proposal.md`
- `doc/high-level-design.md`
- `doc/detailed-design.md`
- `doc/test-plan.md`
- `doc/quality-gates.md`
- `doc/tasks/progress.md`
- `doc/performance-log.md`
- Existing `doc/performance-diagnosis.md`
- `/tmp/senpai_local_relax_full.json`
- `/tmp/senpai_slicing_full.json`
- `/tmp/senpai_aspect_full.json`
- `/tmp/senpai_frame_compact_full.json`
- `/tmp/senpai_connection_43299_full.json`
- `iccad2026contest/my_optimizer.py`
- `iccad2026contest/iccad2026_evaluate.py`
- `litetestLoader.py`

## Current Correctness Status

The kept aspect-candidate artifact and the reverted local-relax/slicing artifacts all report stable hard feasibility:

- `/tmp/senpai_aspect_full.json`: 100/100 feasible.
- `/tmp/senpai_local_relax_full.json`: 100/100 feasible.
- `/tmp/senpai_slicing_full.json`: 100/100 feasible.
- Capped feasible cases: 0/100.
- The result JSONs have no per-case errors.
- Shapely-aligned read-only recomputation of soft subcategories matched every saved `violations_relative` value.

Correctness conclusion:

- Basic correctness is stable enough for optimization diagnosis.
- The current bottleneck is solution quality and generated layout topology, not hard feasibility.
- Any optimization attempt must preserve 100/100 local validation feasibility.

Code provenance caveat:

- `doc/performance-log.md` records the local-relax trial as reverted because it slightly improved score but worsened weighted area and runtime.
- `doc/performance-log.md` records the recursive-slicing trial as reverted because it worsened score, HPWL, area, and runtime.
- Therefore `/tmp/senpai_local_relax_full.json` and `/tmp/senpai_slicing_full.json` are useful diagnostic artifacts, but they do not represent the current checked-in optimizer behavior.

Checkpoint provenance caveat:

- `dit_gnn_step_43299_epoch_11_end.pth` exists under `iccad2026contest/checkpoints/`.
- `dit_gnn_step_141000_loss_1.05.pth` also exists under the same directory.
- `my_optimizer.py` prefers `MY_OPTIMIZER_CHECKPOINT` first; otherwise it sorts `checkpoints/*.pth` by parsed step and prefers `step_141000` over `step_43299`.
- Future comparisons should keep `MY_OPTIMIZER_CHECKPOINT=/home/kuotzuwei15/pda/senpai/iccad2026contest/checkpoints/dit_gnn_step_43299_epoch_11_end.pth` explicit.

## Current Performance Baseline

| Metric | Value | Command | Verified? |
|---|---:|---|---|
| Score | 4.498339808 | Kept-code saved artifact `/tmp/senpai_aspect_full.json`; compatible run command is `MY_OPTIMIZER_CHECKPOINT=/home/kuotzuwei15/pda/senpai/iccad2026contest/checkpoints/dit_gnn_step_43299_epoch_11_end.pth python iccad2026_evaluate.py --evaluate my_optimizer.py --output /tmp/senpai_aspect_full.json` | Saved artifact parsed in this session; the same checkpoint path was verified present |
| Runtime | 12.0550 s average; 50.4962 s weighted average | Same saved artifact | Parsed from saved artifact; local score neutralizes runtime |
| Memory | Unknown | Not measured | Missing |

Additional kept-artifact facts:

- Timestamp: `2026-06-16T19:04:35.112101`.
- Average cost: `4.388497655`.
- Weighted `hpwl_gap`: `2.022872854`.
- Weighted `area_gap`: `2.162657056`.
- Weighted `violations_relative`: `0.185802775`.
- Weighted soft numerator split from Shapely-aligned read-only recompute: boundary `7.3066`, grouping `3.3990`, MIB `0.1537`, total soft `10.8593`, denominator `58.9547`.
- The 101-120 block bucket contributes `81.13%` of the weighted score.
- The 116-120 block bucket contributes `34.08%` of the weighted score.

Rejected-attempt comparisons:

- Local relax improved score only `4.498339808 -> 4.480942306`, but weighted area worsened `2.162657056 -> 2.171124361` and weighted runtime worsened `50.4962 s -> 52.9001 s`; it was reverted.
- Recursive slicing regressed score `4.498339808 -> 4.537967565`, weighted HPWL `2.022872854 -> 2.099279829`, weighted area `2.162657056 -> 2.251372582`, and weighted runtime `50.4962 s -> 58.8886 s`; it was reverted.
- In the local-relax artifact, only one meaningful weighted case changed: case 92 improved cost `5.0876 -> 4.6978`; the top contributors 99, 98, 95, 94, and 97 were unchanged.

## Expected Performance Target

User target:

- Converge to total score `<= 2.0`.

Minimum constraints:

- Preserve 100/100 hard feasibility.
- Keep checkpoint provenance explicit.
- Compare before/after on the same evaluator command or saved artifact.
- Prioritize large cases because the score is dominated by high block counts.

## Gap Analysis

Current kept-code total score is `4.4983`, leaving a gap of about `2.4983` points to the target.

Counterfactual estimates from `/tmp/senpai_aspect_full.json`:

- If soft violations were eliminated while HPWL and area stayed unchanged, estimated score would still be about `3.0928`.
- If HPWL gap alone were eliminated, estimated score would be about `3.0276`.
- If area gap alone were eliminated, estimated score would be about `2.9305`.
- If both HPWL and area gaps were eliminated while soft violations stayed unchanged, estimated score would be about `1.4598`.

Uniform reduction estimates:

- With current soft violations, HPWL and area gaps must both shrink to about `17.78%` of current values to reach `<= 2.0`.
- If soft violations are cut in half, HPWL and area gaps still must shrink to about `31.53%` of current values.
- If soft violations are eliminated, HPWL and area gaps still must shrink to about `47.78%` of current values.

Large-case bucket view:

| Bucket | Weight Share | Score | HPWL Gap | Area Gap | V_rel | Weighted Runtime |
|---|---:|---:|---:|---:|---:|---:|
| 21-60 | 0.0065 | 4.1397 | 2.0856 | 1.5456 | 0.1910 | 1.4464 s |
| 61-100 | 0.1822 | 4.8097 | 2.2459 | 2.0956 | 0.2105 | 7.2453 s |
| 101-120 | 0.8113 | 4.4313 | 1.9723 | 2.1826 | 0.1802 | 60.6011 s |
| 116-120 | 0.3408 | 4.3683 | 2.0026 | 2.2966 | 0.1603 | 95.8482 s |

Top weighted contributors:

| Test ID | Blocks | Cost | Weight Share | HPWL Gap | Area Gap | V_rel |
|---:|---:|---:|---:|---:|---:|---:|
| 99 | 120 | 5.1355 | 0.0800 | 2.0849 | 3.0937 | 0.1791 |
| 98 | 119 | 4.5953 | 0.0736 | 1.8213 | 2.1988 | 0.2115 |
| 95 | 116 | 4.9735 | 0.0573 | 3.1104 | 2.2361 | 0.1515 |
| 94 | 115 | 5.0331 | 0.0527 | 3.2796 | 2.1205 | 0.1538 |
| 97 | 118 | 3.7433 | 0.0677 | 1.7504 | 1.6139 | 0.1667 |
| 92 | 113 | 4.6978 | 0.0446 | 1.8387 | 1.6601 | 0.2679 |
| 96 | 117 | 3.2378 | 0.0623 | 1.3657 | 2.1864 | 0.0769 |
| 93 | 114 | 3.8984 | 0.0485 | 1.6594 | 1.6347 | 0.1935 |

## Benchmark or Evaluator Details

The local evaluator computes feasible cost as:

```text
Cost = (1 + 0.5 * (max(0, HPWL_gap) + max(0, Area_gap))) * exp(2 * V_rel) * RuntimeAdjustment
```

Hard infeasibility costs exactly `10.0`, while feasible costs are capped below `10.0`.

The local full evaluation neutralizes runtime by using `RuntimeFactor = 1.0`, so runtime is recorded but does not explain the saved local score. Official hidden scoring may still penalize the current long large-case runtime.

The total score is an exponentially weighted average:

```text
Total Score = sum_i Cost[i] * exp(n_i / 12) / sum_j exp(n_j / 12)
```

Soft violations are:

```text
V_rel = (boundary + grouping + MIB) / N_soft
```

The saved artifact stores aggregate `violations_relative`; the boundary/grouping/MIB split above was recovered by read-only recomputation from saved positions and local validation data using evaluator-aligned Shapely grouping semantics.

## Observed Bottlenecks

- Weighted HPWL gap remains high at `2.0229`; the current placement has roughly triple the baseline wirelength on a weighted relative basis.
- Weighted area gap remains high at `2.1627`; the current bounding boxes are still roughly `3.16x` the baseline bounding-box area on a weighted relative basis.
- The local-relax attempt barely affected the weighted result: only case 92 changed materially, while the largest score contributors remained unchanged.
- The recursive-slicing attempt worsened score, HPWL, area, and runtime despite improving soft violations, so a naive broader topology is not enough.
- Large cases dominate: cases 101-120 contribute `81.13%` of weighted score and still have score `4.4099`.
- Runtime is already expensive on high-block cases: weighted runtime is `50.50 s`, and case 99 took `163.51 s` in the saved kept run. Local score ignores this, but official runtime risk is rising.
- Boundary violations still dominate the soft numerator at `7.3066`, with grouping at `3.3990` and MIB nearly solved at `0.1537`.
- Soft-only improvement cannot reach the target: even perfect soft constraints leave estimated score near `3.0928`.
- `CandidateSelector._proxy_key(...)` still selects candidates using raw `hpwl + 0.01 * bbox_area` plus a large soft penalty, while the official score depends on relative HPWL/area gaps and multiplicative soft penalties.
- The current constructive families are all variations of the same structural idea: boundary rails, a central packed interior, optional frame compaction, and optional soft-block aspect profiles. This creates feasible layouts but does not explore enough fundamentally different adjacency/topology choices.
- The diffusion checkpoint is not currently the limiting factor by itself. The trained model provides guidance, but final coordinates are dominated by deterministic candidate generation and selection.

## Likely Causes

### Cause 1: Constructive layout topology is too narrow

- Type: Algorithmic limitation / search-control limitation.
- Evidence: Boundary-frame, frame-compaction, aspect-ratio, local-relax, and recursive-slicing attempts all remain near the same large-case score pattern. The local-relax trial changed only case 92 and left cases 99, 98, 95, 94, and 97 unchanged. The recursive-slicing trial regressed score `4.4983 -> 4.5380`. These high-weight cases still have HPWL gaps around `1.75-3.28` and area gaps around `1.61-3.09`.
- Affected modules: `ConstructiveCandidateLegalizer._boundary_frame_candidate`, `_boundary_frame_compaction_candidates`, `_boundary_frame_compaction_candidate`, `_pack_connected_bottom_left_unit_origins`, `_place_boundary_rail_units`, `CandidateSelector.best_feasible`.
- Risk: High. A broader topology can improve score, but it can also increase runtime and destabilize boundary/grouping unless checker-gated.
- Confidence: High.

### Cause 2: Local relaxation was too small and not targeted at dominant cases

- Type: Algorithmic limitation / parameter tuning issue.
- Evidence: `/tmp/senpai_local_relax_full.json` improved score only `0.0174` weighted points versus `/tmp/senpai_aspect_full.json`, worsened area gap, and increased runtime. Per-case comparison shows the only meaningful weighted improvement was case 92; the top five weighted contributors were unchanged.
- Affected modules: reverted local-relax trial from `iccad2026contest/my_optimizer.py`; future work should not repeat the same local rip-up strategy without stronger target selection and area acceptance criteria.
- Risk: Medium. More local search can consume runtime without moving the metric.
- Confidence: High.

### Cause 3: Candidate selection proxy is mismatched to the official objective

- Type: Benchmark/evaluator mismatch / parameter tuning issue.
- Evidence: `CandidateSelector._proxy_key(...)` uses raw HPWL and raw bbox area. The evaluator uses relative gaps against per-case baselines, clamps only negative gaps, and multiplies by `exp(2 * V_rel)`. This mismatch can keep selecting candidates that are locally reasonable but fail the weighted target, especially when the same candidate family offers tradeoffs between area, HPWL, and soft violations.
- Affected modules: `CandidateSelector._proxy_key`, `ConstructiveCandidateLegalizer.PROXY_BBOX_WEIGHT`, `ConstructiveCandidateLegalizer.PROXY_SOFT_WEIGHT`.
- Risk: Medium. True evaluator baselines are not passed into `solve()`, so any in-solver normalization must be approximate.
- Confidence: Medium to high.

### Cause 4: Neural checkpoint loss is not the final legalized-score bottleneck

- Type: Training/evaluator mismatch.
- Evidence: The user identifies `dit_gnn_step_43299_epoch_11_end.pth` as relatively low loss, but the final kept legalized score remains `4.4983`. The training proxy does not model boundary/grouping/MIB exactly, candidate source selection, anchor-aware legalization, or runtime. The final result is mostly determined by handcrafted legalizer candidates.
- Affected modules: `training_example.py`, training-loss helpers in `iccad2026contest/iccad2026_evaluate.py`, `DiffusionGuidanceAdapter`, `ConstructiveCandidateLegalizer`.
- Risk: Low for diagnosis, high for a training-only fix.
- Confidence: Medium to high.

## Optimization Hypotheses

| Priority | Hypothesis | Expected Impact | Risk | Affected Files | Verification |
|---:|---|---|---|---|---|
| 1 | Replace `CandidateSelector._proxy_key(...)` with a scale-normalized proxy based on candidate-set min/max HPWL and bbox area per solve call, with a softer but multiplicative soft-violation term. | Medium. Could choose better among existing candidates without adding much runtime. | Medium. It cannot overcome a narrow candidate set; wrong normalization can regress accepted choices. | `iccad2026contest/my_optimizer.py` | Full eval before/after; rollback if total score, weighted HPWL, or weighted area regress. |
| 2 | Add optional candidate-source and component logging for high-weight cases only, then use it to tune topology generation and selector weights. | Medium for diagnosis quality, low direct score impact. | Low to medium. Logging must be optional and not affect runtime or submission behavior. | `iccad2026contest/my_optimizer.py`, optional `/tmp` analysis output only | Run one full eval with logging enabled; verify no score change and identify winning source per top contributor. |
| 3 | Design a narrower case-targeted topology change only after candidate-source evidence identifies which existing family wins and fails on cases 99, 98, 95, 94, and 97. | Medium to high, but evidence-dependent. | High. More search has already shown runtime and quality regression risk. | `iccad2026contest/my_optimizer.py` | Require candidate-source evidence first; then validate with explicit checkpoint and rollback on score/runtime regression. |

## Recommended First Optimization

Implement Hypothesis 1 next: a scale-normalized `CandidateSelector` proxy.

The recursive-slicing topology was attempted and reverted because it worsened score, HPWL, area, and runtime. The next smallest reversible step is to improve selection among the existing candidate families before adding more expensive search.

Concrete scope:

- Change candidate selection only; do not add new placement candidates in the first pass.
- Compute per-solve candidate metrics after `SoftConstraintImprover`: HPWL, bbox area, soft relative, and source order.
- Normalize HPWL and bbox area against the candidate set's own finite min/max or best finite values so raw scale does not dominate.
- Use a proxy shaped closer to the evaluator: `1 + 0.5 * (normalized_hpwl + normalized_area)` multiplied by `exp(2 * soft_relative)`, with deterministic tie-breakers.
- Keep `FeasibilityChecker` as the acceptance gate.
- Preserve current candidate generation, dimensions, checkpoint handling, and fallback behavior.
- Roll back unless full validation improves total score without regressing both weighted HPWL and weighted area.

Rollback condition:

- Revert if 100/100 feasibility is lost.
- Revert if total score does not improve over the current kept code path `4.498339808`.
- Revert if weighted HPWL and weighted area both regress, because the target `<= 2.0` cannot be reached through soft-constraint improvement alone.
- Revert if runtime grows materially; selector-only work should be close to runtime-neutral.

Answer to the architecture question:

- The whole neural architecture does not yet need to be replaced. The checkpoint is not the dominant blocker visible in this artifact.
- The overall optimizer design still needs reconsideration at the candidate-generation/legalizer level, but the failed slicing attempt shows that simply adding a broader topology is not enough. Before adding more search, the selector should be made more evaluator-aligned so existing candidate tradeoffs are chosen more reliably.

## Exact Prompt for Implementation Loop

```text
Use [$implementation-loop-manager] in optimization mode.

Read:
- doc/performance-diagnosis.md
- doc/performance-log.md if present
- doc/test-plan.md
- doc/quality-gates.md
- relevant source and test files

Implement only the first recommended optimization hypothesis.

Rules:
- Do not change unrelated modules.
- Preserve correctness.
- Add or update tests only if needed.
- Run correctness tests first.
- Run the benchmark or evaluator after the change.
- Compare before/after numbers.
- Keep the change only if the target metric improves without breaking correctness.
- If the result is worse, revert the optimization and record why.
- Update doc/performance-log.md.
- Report changed files, commands run, before/after metrics, and whether the change was kept.
```

## Stop Conditions

- Stop if any change causes hard infeasibility.
- Stop if the selector-proxy change does not improve total score.
- Stop if weighted HPWL and weighted area both regress.
- Stop if runtime grows materially; selector-only work should be close to neutral.
- Stop if the next measurement does not explicitly use `dit_gnn_step_43299_epoch_11_end.pth`.
- Stop if selection requires hidden validation labels or per-case baseline metrics not available to `solve()`.

## Risks and Warnings

- The target `<= 2.0` is aggressive. Current soft cleanup alone cannot reach it.
- The 101-120 block bucket dominates the score, so small-case gains should not steer optimization decisions.
- Official runtime may penalize the current approach more than local validation because local validation neutralizes `RuntimeFactor`.
- A low-loss checkpoint is not sufficient evidence of final quality because final placements are legalizer-selected and the training loss is only a proxy.
- A new topology candidate is higher risk than selector tuning; the recursive-slicing artifact shows that broader search can regress both quality and runtime.
- Any design that uses local validation labels or saved baselines inside `solve()` would be invalid for hidden tests; selector normalization must use only candidates generated from contest inputs.

## Open Questions

- Which candidate family currently wins on cases 99, 98, 95, 94, and 97?
- Can optional candidate-source logging be added without affecting normal submission runtime?
- How much extra runtime is acceptable on `n >= 116` if score improves locally?
- Which selector normalization best predicts the evaluator without access to true per-case baselines?
- Are there additional checkpoints whose guidance changes large-case partition quality, or is legalizer topology still dominant across checkpoints?
