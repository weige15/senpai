# Performance Diagnosis

## Purpose

Diagnose why the trained `iccad2026contest/my_optimizer.py` run remains near the infeasible score cap even though the saved full validation run reports 100/100 feasible cases.

## Diagnosis Scope

This diagnosis covers the saved validation result `iccad2026contest/eval_full_after_training.json`, the local evaluator scoring formula, and the optimizer implementation paths that affect feasibility, HPWL, bounding-box area, and soft-constraint violations.

No optimizer code was changed during this diagnosis.

## Source Documents Read

- `AGENTS.md`
- `doc/problem-brief.md`
- `doc/proposal.md`
- `doc/high-level-design.md`
- `doc/detailed-design.md`
- `doc/test-plan.md`
- `doc/quality-gates.md`
- `doc/tasks/progress.md`
- `iccad2026contest/eval_full_after_training.json`
- `iccad2026contest/iccad2026_evaluate.py`
- `iccad2026contest/my_optimizer.py`

## Current Correctness Status

The user-provided lab-server run reports 100/100 feasible validation cases:

```text
python iccad2026_evaluate.py --evaluate my_optimizer.py --output eval_full_after_training.json
Feasible: 100
Total Score: 9.7460
Avg Cost: 9.6864
Avg Runtime: 1.13s
```

The saved JSON in this workspace matches that baseline. Hard feasibility is therefore stable enough to proceed with score optimization. Local evaluator reruns were not performed because this repository requires approval before evaluator commands. A read-only attempt to compute boundary/grouping/MIB subcounts from the local evaluator was blocked by a missing local dependency: `ModuleNotFoundError: No module named 'numpy'`.

## Current Performance Baseline

| Metric | Value | Command | Verified? |
|---|---:|---|---|
| Score | 9.7459558487 | `python iccad2026_evaluate.py --evaluate my_optimizer.py --output eval_full_after_training.json` | User-provided run; saved JSON parsed locally |
| Runtime | 1.1306540513 s average | Same command | User-provided run; saved JSON parsed locally |
| Memory | Unknown | Not measured | Missing |

Additional parsed baseline facts:

- Feasible cases: 100/100.
- Average cost: 9.6864105735.
- Cost-capped cases: 72/100 at `9.999999`.
- Weighted share of cost-capped cases: 80.01% of final score weight.
- Weighted average `hpwl_gap`: 3.0854.
- Weighted average `area_gap`: 0.2623.
- Weighted average `violations_relative`: 0.8135.
- Weighted average runtime: 2.6663 s, but local scoring resets runtime factor to 1.0 for cost computation.

## Expected Performance Target

No official numeric target is available in the local documents. The practical near-term target should be:

- keep feasibility at 100/100;
- reduce total score below the current 9.7460 baseline;
- reduce the number of capped feasible cases;
- prioritize larger cases because block counts 101-120 contribute about 81.13% of final score weight in this saved run.

## Gap Analysis

The bottleneck is not hard feasibility anymore. The bottleneck is quality under the feasible scoring formula.

Counterfactual term analysis from the saved JSON:

| Counterfactual | Weighted Total Score |
|---|---:|
| Original saved run | 9.7460 |
| Set `violations_relative = 0` only | 2.6738 |
| Set `hpwl_gap = 0` only | 5.7889 |
| Set `area_gap = 0` only | 9.5855 |
| Set `hpwl_gap = 0` and `area_gap = 0` | 5.1208 |

This makes the ranking clear:

1. Soft-constraint violations are the largest scoring bottleneck because they are inside `exp(2 * V_rel)`.
2. HPWL is the second major bottleneck.
3. Bounding-box area is currently a much smaller first-order lever.
4. Runtime is not affecting the saved local score because the evaluator applies neutral local runtime factor `1.0`.

## Benchmark or Evaluator Details

`iccad2026contest/iccad2026_evaluate.py` computes feasible cost as:

```text
Cost = (1 + 0.5 * (max(0, HPWL_gap) + max(0, Area_gap))) * exp(2 * V_rel) * runtime_adjustment
```

For local full evaluation, runtime is recomputed with neutral runtime factor `1.0`. Feasible costs are capped at `9.999999`, which explains why many cases look indistinguishable from infeasible cases even though they are feasible.

The total score is an exponentially weighted average using `exp(block_count / 12)`, so large block-count cases dominate. In the saved run:

- 101-110 contributes 24.58% weight, score 9.8636.
- 111-115 contributes 22.47% weight, score 9.999999.
- 116-120 contributes 34.08% weight, score 9.6769.

## Observed Bottlenecks

- Soft violation rate is very high: weighted `violations_relative = 0.8135`; many printed cases are around `0.6-0.9`.
- HPWL is also high: weighted `hpwl_gap = 3.0854`; case 0 has `hpwl_gap = 3.9396`.
- The first saved case shows a nearly one-column placement: most blocks have `x = 71.0`, which is consistent with poor HPWL and a legalizer dominated by packing legality rather than netlist quality.
- Area is not the first bottleneck: weighted `area_gap = 0.2623`, and zeroing area alone barely changes final score.
- The score cap hides partial progress: 72 cases are already capped, and capped cases carry about 80.01% of the final weighted score.

## Likely Causes

### Cause 1: SoftConstraintImprover is a no-op

- Type: Algorithmic limitation.
- Evidence: `iccad2026contest/my_optimizer.py` defines `SoftConstraintImprover.improve(...)` to return a copy of the placement unchanged. Boundary, grouping, and MIB moves are explicitly disabled. The evaluator's `violations_relative` term is the largest counterfactual lever.
- Affected modules: `SoftConstraintImprover`, `FeasibilityChecker`, `MyOptimizer.solve`.
- Risk: Low for checker-backed boundary moves; medium for grouping and MIB moves because they can disturb legality or area constraints.
- Confidence: High.

### Cause 2: AnchorAwareLegalizer scoring ignores HPWL and soft constraints

- Type: Algorithmic limitation / heuristic mismatch.
- Evidence: `AnchorAwareLegalizer._score_candidate(...)` ranks by area growth, predicted distance, span, coordinate bias, and coordinates. It does not use `b2b_connectivity`, `p2b_connectivity`, pins, boundary masks, grouping, or MIB metadata. Saved case 0 has a vertical-column layout and `hpwl_gap = 3.9396`.
- Affected modules: `AnchorAwareLegalizer`, `DiffusionGuidanceAdapter`, `MyOptimizer.solve`.
- Risk: Medium. HPWL-aware scoring can improve wirelength but may increase runtime or area if not bounded.
- Confidence: High.

### Cause 3: Soft-constraint subcomponents are not measured in the saved result JSON

- Type: Unknown / needs measurement, with likely algorithmic limitation.
- Evidence: `TestResult` stores only aggregate `violations_relative`, not boundary, grouping, MIB, numerator, or denominator. The local attempt to re-evaluate component counts was blocked by missing `numpy`. The implementation currently derives most movable soft dimensions independently as square blocks and does not abut groups or enforce boundary contact.
- Affected modules: `iccad2026_evaluate.py` reporting path, `SoftConstraintImprover`, `HardConstraintNormalizer`, `AnchorAwareLegalizer`.
- Risk: Medium. Optimizing the wrong soft subcomponent could produce little score movement.
- Confidence: Medium.

## Optimization Hypotheses

| Priority | Hypothesis | Expected Impact | Risk | Affected Files | Verification |
|---:|---|---|---|---|---|
| 1 | Implement checker-backed boundary-only soft-constraint improvement for movable blocks. Try translations to required current bbox edges and accept only if hard-feasible and not worse by evaluator-style proxy. | Medium to high if boundary violations are a material part of `V_rel`; directly targets the biggest score term. | Low to medium. Boundary is per-block and can be skipped when unsafe. | `iccad2026contest/my_optimizer.py` | `python iccad2026_evaluate.py --validate my_optimizer.py`; `python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0`; full evaluation |
| 2 | Add bounded HPWL-aware candidate scoring to `AnchorAwareLegalizer`, using already placed connected blocks and pins as an incremental term after hard-feasible filtering. | High if the one-column/strip behavior is causing most HPWL gap. | Medium. More scoring work per candidate and possible area tradeoff. | `iccad2026contest/my_optimizer.py` | Same evaluator commands; compare weighted `hpwl_gap`, score, runtime |
| 3 | Add grouping/MIB-aware local improvement: abut blocks in the same cluster when legal, and synchronize MIB shapes only when area-compatible. | Potentially high if soft violations are dominated by grouping or MIB. | Medium to high. More complex and can easily break area, fixed-shape, or non-overlap constraints. | `iccad2026contest/my_optimizer.py` | Same evaluator commands plus per-component soft diagnostics if dependencies are available |

## Recommended First Optimization

Implement only Hypothesis 1 first: a checker-backed, boundary-only `SoftConstraintImprover`.

Reasoning:

- The biggest measured bottleneck is `violations_relative`.
- The current soft-improvement stage is completely disabled.
- Boundary moves are the smallest soft-constraint improvement that can be made reversible and legality-preserving.
- The change can be accepted only when the full hard-feasibility checker passes, preserving the current 100/100 feasibility baseline.

Rollback condition:

- Revert the change if full evaluation feasibility drops below 100/100, total score is not lower than 9.7460, capped weighted share does not improve, or average runtime increases substantially without a score gain.

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

- Stop if validation feasibility drops below 100/100.
- Stop if boundary-only moves do not reduce total score or capped-case count.
- Stop if evaluator dependencies are unavailable on the target lab server.
- Stop before training or checkpoint generation unless explicitly approved.
- Stop before broader HPWL, grouping, or MIB changes until the boundary-only attempt is measured.

## Risks and Warnings

- Because 72/100 cases are capped, small improvements may not move the final score unless they are large enough to uncap weighted large cases.
- The saved result JSON does not expose soft subcomponents, so boundary-only optimization may underperform if grouping or MIB dominates `V_rel`.
- Official runtime normalization is not locally measurable; a slower local optimizer may be penalized differently on the leaderboard.
- The current local environment lacks `numpy`, blocking evaluator-based component re-scoring here. The lab server environment used for the saved run likely has the required dependencies.

## Open Questions

- What are the boundary, grouping, and MIB violation counts for `eval_full_after_training.json` on the lab server?
- Does boundary-only repair reduce enough weighted large cases below the score cap?
- How much of the HPWL gap is caused by candidate scoring versus weak diffusion guidance?
- Are MIB groups area-compatible, or do some groups have conflicting area targets that make exact identical dimensions impossible?
