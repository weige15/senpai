# Performance Diagnosis

## Purpose

Diagnose why `iccad2026contest/eval_full_after_training.json` is still suboptimal and answer whether `iccad2026contest/my_optimizer.py` legalizes like `noML_optimizer.py`.

Conclusion: `my_optimizer.py` does legalize, and it has a partial noML-style constructive candidate path, but it is not equivalent to `noML_optimizer.py`. The current trained-checkpoint artifact is hard-feasible but still score-limited by high soft violations and HPWL, especially on large weighted cases.

No optimizer code was changed during this diagnosis.

## Diagnosis Scope

Covered:

- `iccad2026contest/my_optimizer.py` normalization, diffusion guidance, anchor-aware legalizer, constructive candidate legalizer, soft improver, feasibility checker, fallback packer, candidate selector, and `solve()` orchestration.
- `noML_optimizer.py` dimension planning, placement units, constructive seed generation, candidate scoring, boundary/skyline/frame packing, and local-search layers.
- Saved result artifacts: `iccad2026contest/eval_full_after_training.json`, `iccad2026contest/my_optimizer_results.json`, `/tmp/senpai_post_boundary.json`, and `/tmp/senpai_constructive_bounded_full.json`.

Not covered:

- No evaluator rerun was performed.
- No training or checkpoint generation was performed.
- No hidden-test behavior was measured.
- No direct `noML_optimizer.py` evaluator result exists in this workspace.

## Source Documents Read

- `doc/problem-brief.md`
- `doc/proposal.md`
- `doc/high-level-design.md`
- `doc/detailed-design.md`
- `doc/test-plan.md`
- `doc/quality-gates.md`
- `doc/tasks/progress.md`
- `doc/performance-log.md`
- `iccad2026contest/my_optimizer.py`
- `noML_optimizer.py`
- `iccad2026contest/eval_full_after_training.json`
- `iccad2026contest/my_optimizer_results.json`
- `/tmp/senpai_post_boundary.json`
- `/tmp/senpai_constructive_bounded_full.json`

## Current Correctness Status

Saved evaluator artifacts show hard feasibility is stable:

- `iccad2026contest/eval_full_after_training.json`: 100/100 feasible.
- `/tmp/senpai_constructive_bounded_full.json`: 100/100 feasible.
- `/tmp/senpai_post_boundary.json`: 100/100 feasible.
- `iccad2026contest/my_optimizer_results.json`: 100/100 feasible.

The local workspace currently has no `iccad2026contest/checkpoints/` directory, so a fresh local evaluator run would not reproduce the checkpoint-enabled `eval_full_after_training.json` unless the checkpoint is restored.

## Current Performance Baseline

| Metric | Value | Command | Verified? |
|---|---:|---|---|
| Score | 9.3684306419 | `python iccad2026_evaluate.py --evaluate my_optimizer.py` producing `iccad2026contest/eval_full_after_training.json` | Saved artifact parsed in this session; command not rerun |
| Runtime | 2.4154678679 s average | Same saved artifact | Saved artifact parsed in this session; command not rerun |
| Memory | Unknown | Not measured | Missing |

Additional saved comparisons:

- `/tmp/senpai_constructive_bounded_full.json`: score `6.9820660669`, 100/100 feasible, average runtime `2.3177188778s`, 3 capped cases, weighted capped share `6.95%`.
- `/tmp/senpai_post_boundary.json`: score `9.7367167134`, 100/100 feasible, average runtime `0.6807339573s`, 64 capped cases, weighted capped share `65.70%`.
- `iccad2026contest/my_optimizer_results.json`: score `9.9564528621`, 100/100 feasible, average runtime `0.4535196400s`, 79 capped cases, weighted capped share `83.10%`.

## Expected Performance Target

Near-term target:

- Preserve 100/100 local validation feasibility.
- Make checkpoint-enabled evaluation at least recover the saved no-checkpoint constructive result quality, approximately score `< 7.0`.
- Reduce weighted soft violations from the checkpoint artifact's `0.7656` toward the no-checkpoint constructive artifact's `0.3077`.
- Avoid a large runtime increase on large cases because test IDs 80-99 dominate the weighted score.

## Gap Analysis

The checkpoint artifact is legal but still near the feasible cap:

- Total score: `9.3684`.
- Average cost: `9.0845`.
- Capped cases: `53/100`.
- Weighted capped share: `60.09%`.
- Weighted `hpwl_gap`: `2.8529`.
- Weighted `area_gap`: `0.2984`.
- Weighted `violations_relative`: `0.7656`.

For comparison, the saved no-checkpoint bounded constructive artifact has much lower score despite worse weighted HPWL and area:

- Total score: `6.9821`.
- Capped cases: `3/100`.
- Weighted capped share: `6.95%`.
- Weighted `hpwl_gap`: `3.3920`.
- Weighted `area_gap`: `2.2667`.
- Weighted `violations_relative`: `0.3077`.

This implies the first-order gap in `eval_full_after_training.json` is not hard feasibility or area. It is high soft violations, with HPWL also contributing. Large cases are especially important: test IDs 80-99 contribute about `7.64` points of the `9.37` weighted score in the checkpoint artifact.

## Benchmark or Evaluator Details

The evaluator cost is dominated by:

```text
Cost = (1 + 0.5 * (HPWL_gap + Area_gap)) * exp(2 * V_rel) * RuntimeFactor
```

The local evaluator caps feasible costs just below `10.0`. Hard infeasible cases cost `10.0`.

Relevant commands, not run in this diagnosis because they require approval and may write results:

- `python iccad2026_evaluate.py --validate my_optimizer.py`
- `python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0`
- `python iccad2026_evaluate.py --evaluate my_optimizer.py`

## Observed Bottlenecks

- `my_optimizer.py` does have anchor-aware legalization, hard feasibility checking, fallback packing, soft boundary improvement, constructive units, cluster macros, and candidate selection.
- `solve()` builds both an ML anchor-aware candidate and constructive candidates, then returns the best hard-feasible candidate by proxy score.
- The constructive path is only a subset of `noML_optimizer.py`:
  - current `my_optimizer.py` has 3 constructive order variants;
  - `noML_optimizer.py` has many seed orders, true boundary frame/skyline packers, connectivity-greedy order, local search, rip-up/repack, boundary snapping, frame compaction, aspect trials, and MIB sync trials;
  - current `my_optimizer.py` sets `connection_cost = 0.0` inside constructive unit-origin scoring, so the "connected" name is not yet a real connectivity-aware placement cost.
- The trained-checkpoint artifact performs worse than the saved no-checkpoint bounded constructive artifact. The likely mechanical reason is that, when predictions exist, current constructive units carry `predicted_center` and `_trim_candidates(...)` sorts/truncates origins around the prediction. That can discard noML-style boundary/cluster origins that were retained in the no-checkpoint run.
- Current `HardConstraintNormalizer` still uses square dimensions for soft blocks and does not implement `noML_optimizer.py` MIB-compatible dimension planning.
- Current local workspace lacks `iccad2026contest/checkpoints/`, so the checkpoint artifact cannot be reproduced locally without restoring weights.

## Likely Causes

### Cause 1: Trained guidance is over-constraining constructive legalization

- Type: Algorithmic limitation / parameter interaction.
- Evidence: The no-checkpoint bounded constructive artifact scores `6.9821`, while the checkpoint artifact scores `9.3684` with similar average runtime. In current code, predicted centers affect unit origin candidate generation and trimming. With no predictions, candidate trimming keeps deterministic noML-style origins; with predictions, it prioritizes predicted-near origins.
- Affected modules: `ConstructiveCandidateLegalizer`, `DiffusionGuidanceAdapter`, `MyOptimizer.solve`.
- Risk: Medium if fixed by disabling guidance globally; low to medium if fixed by adding a guidance-neutral candidate that competes with guided candidates.
- Confidence: High.

### Cause 2: Current noML-style path is only a bounded subset of `noML_optimizer.py`

- Type: Algorithmic limitation.
- Evidence: `noML_optimizer.py` includes MIB-compatible dimension planning, `PlacementUnitSet`, 9 constructive seed orders, boundary frame/skyline packing, connected bottom-left placement, local search families, rip-up/repack, boundary snapping, and frame compaction. `my_optimizer.py` has constructive units and cluster macros but only 3 order variants, no local search, no true connected placement cost, and no MIB dimension synchronization.
- Affected modules: `ConstructiveCandidateLegalizer`, `CandidateSelector`, `HardConstraintNormalizer`.
- Risk: Medium to high for broad porting; lower if added incrementally.
- Confidence: High.

### Cause 3: Proxy scoring and placement scoring do not fully match evaluator terms during construction

- Type: Heuristic mismatch.
- Evidence: Final `CandidateSelector` scores HPWL, bbox, and soft relative violations, but the per-unit constructive origin scorer mostly optimizes boundary miss, bbox growth, predicted distance, and coordinate bias. `connection_cost` is currently zero, so high-HPWL placements are not prevented while blocks are being placed.
- Affected modules: `ConstructiveCandidateLegalizer._score_unit_origin`, `CandidateSelector`.
- Risk: Medium, because more expensive scoring can increase runtime on large cases.
- Confidence: Medium.

## Optimization Hypotheses

| Priority | Hypothesis | Expected Impact | Risk | Affected Files | Verification |
|---:|---|---|---|---|---|
| 1 | Add a guidance-neutral constructive candidate path that ignores predicted centers for origin trimming/scoring while still letting the ML-guided candidates compete. | High. Directly targets the gap where no-checkpoint constructive scoring beats checkpoint-guided scoring: `6.9821` vs `9.3684`. | Low to medium. It adds candidate diversity without removing existing guidance. | `iccad2026contest/my_optimizer.py` | Validate, then compare full evaluator with and without checkpoint; success means 100/100 feasible and checkpoint-enabled score moves toward `< 7.0`. |
| 2 | Port a small noML local-search family after candidate selection, starting with boundary snap or frame compaction only. | Medium to high. Targets residual soft violations and large-case capped costs. | Medium to high. More runtime and more code surface. | `iccad2026contest/my_optimizer.py` | Full evaluator, capped-case count, weighted soft relative, runtime on test IDs 100+. |
| 3 | Add real connectivity cost to constructive unit-origin scoring using placed blocks and pins. | Medium. Can reduce HPWL before final candidate scoring. | Medium. May slow construction and trade off against soft constraints. | `iccad2026contest/my_optimizer.py` | Compare weighted HPWL, total score, and runtime; preserve soft violation gains. |

## Recommended First Optimization

Implement Hypothesis 1 first: add a guidance-neutral constructive candidate path.

Concrete implementation intent:

- Keep the existing ML-guided `AnchorAwareLegalizer` and guided constructive candidates.
- Add a second constructive build using the same `problem` but guidance with:
  - deterministic movable order, or the same order but no predicted centers;
  - empty `predicted_positions` and `predicted_centers`;
  - `available = False`.
- Prefix candidate sources clearly, for example `constructive_neutral:*`.
- Let `CandidateSelector.best_feasible(...)` choose among ML anchor-aware, guided constructive, and guidance-neutral constructive candidates.
- Do not disable checkpoint loading.
- Do not import `noML_optimizer.py`; keep submission self-contained.

Rollback condition:

- Revert if feasibility drops below 100/100, checkpoint-enabled score does not improve over `9.3684`, or average runtime grows substantially without capped-case reduction.

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

- Stop if any evaluator run reports hard infeasibility.
- Stop if the change disables checkpoint loading or removes the ML-guided path instead of adding a competing neutral candidate.
- Stop if the checkpoint artifact cannot be reproduced because weights are missing; record that verification is blocked until the checkpoint is restored.
- Stop before running training, installing dependencies, or downloading data without approval.

## Risks and Warnings

- `iccad2026contest/eval_full_after_training.json` is a saved artifact, not a rerun from this diagnosis.
- The local checkpoint directory is missing, so checkpoint-enabled behavior cannot be reproduced locally right now.
- `/tmp/senpai_constructive_bounded_full.json` is also a saved artifact; it is useful for comparison but should be rerun before declaring a final improvement.
- Full noML parity would require more than the current constructive path; porting everything at once would be risky.

## Open Questions

- Where is the checkpoint that produced `eval_full_after_training.json`, and should it be restored before the next implementation loop?
- Should the guidance-neutral candidate be enabled always, or only when `guidance.available` is true?
- What score does direct `noML_optimizer.py` achieve on the same validation set?
- Should MIB-compatible dimension planning be ported before or after local-search improvements?
