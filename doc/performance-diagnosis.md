# Performance Diagnosis

## Purpose

Diagnose the bottleneck preventing `iccad2026contest/eval_full_after_training.json` from converging to a score of `2.0` or below.

The current artifact is hard-feasible and much better than the earlier feasibility-only baselines, but the score is now limited by the quality of candidate placements after legalization, not by checkpoint loss alone. No optimizer code was changed during the diagnosis step; the follow-up implementation result is recorded in `doc/performance-log.md`.

## Diagnosis Scope

Covered:

- Current saved evaluator artifact: `iccad2026contest/eval_full_after_training.json`.
- Current checkpoint inventory under `iccad2026contest/checkpoints/`.
- `iccad2026contest/my_optimizer.py` normalization, guidance, constructive legalizer, soft improver, candidate selector, checkpoint loading, and `solve()` orchestration.
- `iccad2026contest/iccad2026_evaluate.py` scoring, soft-violation normalization, runtime handling, and weighted total score.
- `iccad2026contest/training_example.py` and README training-loss notes.
- `noML_optimizer.py` only as a local reference for missing constructive and local-search families.
- Read-only recomputation of evaluator component counts from saved positions and `LiteTensorDataTest/`.

Not covered:

- No evaluator command was rerun.
- No training command was run.
- No checkpoint was loaded for inspection.
- No hidden-test behavior was measured.

## Source Documents Read

- `doc/problem-brief.md`
- `doc/proposal.md`
- `doc/high-level-design.md`
- `doc/detailed-design.md`
- `doc/test-plan.md`
- `doc/quality-gates.md`
- `doc/tasks/progress.md`
- `doc/performance-log.md`
- `iccad2026contest/eval_full_after_training.json`
- `iccad2026contest/my_optimizer_results.json`
- `/tmp/senpai_neutral_constructive_full.json`
- `/tmp/senpai_constructive_bounded_full.json`
- `/tmp/senpai_post_boundary.json`
- `iccad2026contest/my_optimizer.py`
- `iccad2026contest/iccad2026_evaluate.py`
- `iccad2026contest/training_example.py`
- `README.md`
- `noML_optimizer.py`

## Current Correctness Status

The saved result artifact is hard-feasible:

- `iccad2026contest/eval_full_after_training.json`: 100/100 feasible.
- Read-only rescoring of saved positions against local validation data found zero overlap, area-tolerance, and hard dimension violations.
- Validation and evaluator commands were not rerun in this diagnosis because project rules require approval for evaluator runs.

The checkpoint typo has been corrected: the intended checkpoint is present locally under `iccad2026contest/checkpoints/`:

- `dit_gnn_step_141000_loss_1.05.pth` at about 128 MB.

## Current Performance Baseline

| Metric | Value | Command | Verified? |
|---|---:|---|---|
| Score | 6.982045607 | Saved `iccad2026contest/eval_full_after_training.json`; compatible command is `python iccad2026_evaluate.py --evaluate my_optimizer.py --output eval_full_after_training.json` | Saved artifact parsed and component-rescored in this session; evaluator command not rerun |
| Runtime | 5.7033 s average; 13.2846 s weighted average | Same saved artifact | Saved artifact parsed in this session; runtime is locally neutralized in cost |
| Memory | Unknown | Not measured | Missing |

Additional current artifact facts:

- Feasible cases: 100/100.
- Capped feasible cases: 3/100.
- Weighted capped share: 6.95%.
- Weighted `hpwl_gap`: 3.3919.
- Weighted `area_gap`: 2.2668.
- Weighted `violations_relative`: 0.3077.
- Weighted soft counts from read-only recomputation: boundary `15.6460`, grouping `1.7552`, MIB `0.5367`, total soft `17.9379`, max possible `58.9547`.

## Expected Performance Target

User target:

- Converge to total score `<= 2.0`.

Minimum constraints for any optimization attempt:

- Preserve 100/100 local validation hard feasibility.
- Do not rely on lower training loss as the only success signal.
- Measure before/after on the same evaluator artifact or a newly approved full evaluator run.
- Prioritize large cases because block counts 101-120 carry about 81.13% of the current weighted score.

## Gap Analysis

Current total score is `6.9820`, so the gap to the requested target is about `4.9820` points.

The remaining gap is not a hard-feasibility issue. It is a combined quality issue:

- If soft violations were magically set to zero while HPWL and area stayed unchanged, estimated score would still be about `3.8293`.
- If HPWL gap were magically set to zero while area and soft stayed unchanged, estimated score would be about `3.8653`.
- If area gap alone were set to zero while HPWL and soft stayed unchanged, estimated score would be about `5.0087`.
- If both HPWL and area gaps were set to zero while soft stayed unchanged, estimated score would be about `1.8688`.
- If all HPWL, area, and soft gaps were zero, score would be `1.0`.

Interpretation:

- Reaching `<= 2.0` needs both lower soft violations and much better HPWL/area quality.
- Boundary violations dominate the remaining soft term. Weighted boundary violations are about 87% of the weighted soft numerator.
- The largest weighted cases dominate. The 101-120 bucket has score `7.0986`, weighted `hpwl_gap` `3.4644`, weighted `area_gap` `2.3761`, and weighted `violations_relative` `0.3035`.

Top weighted contributors in the current artifact are all large cases:

| Test ID | Blocks | Cost | Weight Share | HPWL Gap | Area Gap | V_rel |
|---:|---:|---:|---:|---:|---:|---:|
| 99 | 120 | 7.4836 | 0.0800 | 3.223 | 4.342 | 0.224 |
| 98 | 119 | 7.8448 | 0.0736 | 4.028 | 3.488 | 0.250 |
| 97 | 118 | 8.4317 | 0.0677 | 4.574 | 3.319 | 0.267 |
| 95 | 116 | 7.8364 | 0.0573 | 4.392 | 2.971 | 0.258 |
| 91 | 112 | 9.999999 | 0.0411 | 6.692 | 0.369 | 0.414 |

## Benchmark or Evaluator Details

The evaluator computes feasible cost as:

```text
Cost = (1 + 0.5 * (max(0, HPWL_gap) + max(0, Area_gap))) * exp(2 * V_rel) * RuntimeAdjustment
```

Local full evaluation then neutralizes runtime with `RuntimeFactor = 1.0` before computing the final score. In the saved local artifact, runtime is recorded but is not the direct reason the score is `6.9820`.

The total score is an exponentially weighted average:

```text
Total Score = sum_i Cost[i] * exp(n_i / 12) / sum_j exp(n_j / 12)
```

Soft violations are:

```text
V_rel = (boundary + grouping + MIB) / N_soft
```

Hard infeasibility would cost exactly `10.0`, but the current artifact is hard-feasible.

## Observed Bottlenecks

- The checkpoint loss is no longer the main bottleneck. `eval_full_after_training.json` and `/tmp/senpai_neutral_constructive_full.json` have essentially the same score, so the final selected candidate is coming from the constructive search ceiling rather than from improved model guidance.
- The current constructive path has only three order variants and a bounded origin search. The local `noML_optimizer.py` reference has more seed orders, real boundary frame/skyline packing, connected bottom-left placement, local search, rip-up/repack, frame compaction, MIB sync, and aspect trials.
- `ConstructiveCandidateLegalizer._score_unit_origin(...)` still sets `connection_cost = 0.0`, so per-placement origin selection does not use real block-to-block or pin-to-block connectivity.
- `HardConstraintNormalizer._soft_dimensions(...)` uses square dimensions for all non-fixed soft blocks. It does not implement MIB-compatible dimension planning or aspect trials.
- Boundary constraints dominate the remaining soft violations. Current `SoftConstraintImprover` can perform checked boundary translations, but it cannot create a coherent boundary frame/rail layout for many boundary-constrained blocks.
- The candidate selector scores full candidates by HPWL, bbox area, and soft relative violations, but it can only choose among candidates that earlier construction actually generated.

## Likely Causes

### Cause 1: Candidate-set ceiling after legalization

- Type: Algorithmic limitation.
- Evidence: The current saved checkpoint artifact scores `6.982045607`, while `/tmp/senpai_neutral_constructive_full.json` scores `6.982066067` with the same 3 capped cases and nearly identical weighted metrics. The trained guidance is not producing a better selected final placement than the guidance-neutral constructive candidate.
- Affected modules: `ConstructiveCandidateLegalizer`, `CandidateSelector`, `MyOptimizer.solve`.
- Risk: Medium. Expanding candidate families can increase runtime, especially on 101-120 block cases.
- Confidence: High.

### Cause 2: Boundary constraints are not structurally packed

- Type: Algorithmic limitation / soft-constraint heuristic gap.
- Evidence: Read-only rescoring found weighted boundary violations `15.6460` out of weighted total soft violations `17.9379`. Grouping and MIB are much smaller at `1.7552` and `0.5367`. Current code has a candidate named `boundary_skyline_connected`, but in `my_optimizer.py` it is an order variant plus origin scoring, not the boundary frame/skyline packer present in `noML_optimizer.py`.
- Affected modules: `ConstructiveCandidateLegalizer`, `SoftConstraintImprover`, `CandidateSelector`.
- Risk: Medium. Boundary satisfaction can increase bbox area if implemented as naive snapping.
- Confidence: High.

### Cause 3: Construction-time scoring is not aligned with evaluator HPWL and area

- Type: Heuristic mismatch.
- Evidence: Weighted `hpwl_gap` is `3.3919` and weighted `area_gap` is `2.2668`. Eliminating soft violations alone would still leave score around `3.8293`. In construction, `connection_cost` is currently `0.0`, and final candidate selection cannot repair HPWL/area when all generated candidates are poor.
- Affected modules: `ConstructiveCandidateLegalizer._score_unit_origin`, `ConstructiveCandidateLegalizer._generate_unit_origins`, `CandidateSelector`.
- Risk: Medium. Adding connectivity scoring inside the inner placement loop can increase runtime.
- Confidence: High.

## Optimization Hypotheses

| Priority | Hypothesis | Expected Impact | Risk | Affected Files | Verification |
|---:|---|---|---|---|---|
| 1 | Add one noML-style boundary-frame/skyline constructive candidate family for boundary-constrained units, with compact skyline placement for interior units and existing `CandidateSelector` choosing whether to keep it. | High. Directly targets the dominant soft numerator and may also reduce large-case bbox area by replacing ad hoc boundary moves with a structured frame. | Medium. Must preserve hard feasibility and cap candidate growth on 101-120 block cases. | `iccad2026contest/my_optimizer.py` | Validate, full evaluator, and soft-component recompute. Keep only if score improves below `6.9820`, feasibility remains 100/100, and runtime remains practical. |
| 2 | Add real connection cost to constructive unit-origin scoring using already placed block centers and pin positions. | Medium to high. Targets weighted `hpwl_gap` `3.3919`, especially large cases. | Medium. Inner-loop scoring may slow large cases and can trade HPWL against boundary/area. | `iccad2026contest/my_optimizer.py` | Compare weighted HPWL, total score, capped cases, and runtime against `6.9820`. |
| 3 | Implement MIB-compatible dimension planning and a MIB-sync candidate, then let selector choose it. | Low to medium. MIB is only `0.5367` weighted soft count, but synchronized dimensions can reduce soft penalty without hard violations when areas allow. | Medium. Dimension changes must preserve area tolerance and fixed/preplaced immutability. | `iccad2026contest/my_optimizer.py` | Validate hard constraints, compare MIB violations and total score. |

## Recommended First Optimization

Implement Hypothesis 1 first: add one boundary-frame/skyline constructive candidate family.

Status: implemented and kept in performance-log attempt 7 as `constructive:boundary_frame_structured`. Do not repeat this exact first attempt; the next optimization loop should use the recorded attempt-7 metrics as its baseline.

Concrete first attempt:

- Keep existing ML-guided, guidance-neutral, anchor-aware, fallback, and selector paths.
- Add a new candidate source such as `constructive_boundary_frame` or `constructive_boundary_skyline`.
- Build it from current `ConstructiveUnit` objects.
- Place units with boundary masks on their requested bbox rails or corners when possible.
- Pack non-boundary interior units with a compact skyline or bottom-left routine inside the frame.
- Preserve hard feasibility with the existing checker and let `CandidateSelector.best_feasible(...)` decide whether the new candidate beats existing candidates.
- Do not add broad local search, MIB dimension changes, or training changes in this first attempt.

Rollback condition:

- Revert the change if 100/100 feasibility is lost, total score does not improve over `6.9820`, capped cases increase, or average runtime grows substantially without a score improvement.

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
- Stop if boundary-frame placement is implemented as unchecked snapping that overlaps blocks or violates preplaced anchors.
- Stop if the new candidate family makes full validation too slow for large cases without measurable score improvement.
- Stop before running training, installing dependencies, downloading data, or deleting checkpoints without approval.
- Stop if the current `dit_gnn_step_141000_loss_1.05.pth` checkpoint cannot be loaded or reproduced by the evaluator.

## Risks and Warnings

- The current score baseline is a saved artifact, not an evaluator rerun from this diagnosis.
- The checkpoint inventory now matches the corrected user-stated checkpoint: `dit_gnn_step_141000_loss_1.05.pth`.
- Local runtime is neutralized in the current evaluator, but official hidden-test runtime normalization may still penalize slow candidate expansion.
- The training loss omits boundary, grouping, and MIB soft constraints, so more training alone may not reduce the dominant remaining soft term.
- Boundary improvements can worsen area if they simply expand the bbox. The first attempt should generate a competing candidate, not overwrite the current best candidate unconditionally.

## Open Questions

- Did the existing `dit_gnn_step_141000_loss_1.05.pth` checkpoint produce the current `eval_full_after_training.json` exactly, or should the full evaluator be rerun to refresh the baseline?
- What score does the standalone `noML_optimizer.py` achieve on the same validation set when run under the current evaluator?
- Should future training include differentiable approximations for boundary, grouping, and MIB constraints after the candidate-search bottleneck is reduced?
