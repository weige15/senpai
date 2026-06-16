# Performance Diagnosis

## Purpose

Diagnose why the current checkpoint-enabled optimizer result in `/tmp/senpai_aspect_full.json` remains far above the requested target score of `2.0` or below, and identify the next optimization attempt with the best expected leverage.

User-provided checkpoint context:

- `iccad2026contest/checkpoints/dit_gnn_step_43299_epoch_11_end.pth`
- File status: present locally, 128 MB.

No optimizer implementation code was changed during this diagnosis.

## Diagnosis Scope

Covered:

- Saved evaluator artifact: `/tmp/senpai_aspect_full.json`.
- Current checkpoint inventory under `iccad2026contest/checkpoints/`.
- `iccad2026contest/my_optimizer.py` checkpoint loading, normalization, constructive candidate generation, boundary frame compaction, aspect-ratio candidates, connection-aware placement, candidate selection, and `solve()` orchestration.
- `iccad2026contest/iccad2026_evaluate.py` cost formula, hard/soft constraint semantics, local runtime handling, differentiable training proxy, and block-count weighting.
- Local validation dataset constraints were read only to recompute soft-violation subcategories for the saved positions.
- Planning and quality-gate documents under `doc/`.

Not covered:

- The optimizer was not rerun.
- Training was not run.
- The checkpoint tensor was not loaded or inspected.
- No profiler or memory measurement was run.
- Hidden-test runtime normalization was not measured.

## Source Documents Read

- `doc/problem-brief.md`
- `doc/proposal.md`
- `doc/high-level-design.md`
- `doc/detailed-design.md`
- `doc/test-plan.md`
- `doc/quality-gates.md`
- `doc/tasks/progress.md`
- `doc/performance-log.md`
- `README.md`
- `/tmp/senpai_aspect_full.json`
- `iccad2026contest/my_optimizer.py`
- `iccad2026contest/iccad2026_evaluate.py`
- `iccad2026contest/training_example.py`
- `lite_dataset_test.py`
- `litetestLoader.py`

## Current Correctness Status

The saved artifact reports stable hard feasibility:

- `/tmp/senpai_aspect_full.json`: 100/100 feasible.
- Capped feasible cases: 0/100.
- Read-only recomputation from saved positions and local validation data also found weighted hard violations of zero for overlap, area tolerance, and fixed/preplaced dimensions.

Correctness conclusion:

- Basic correctness is stable enough for optimization diagnosis.
- The current bottleneck is solution quality and search geometry, not hard feasibility.
- Any optimization attempt must preserve 100/100 local validation feasibility.

Checkpoint provenance caveat:

- `dit_gnn_step_43299_epoch_11_end.pth` exists under `iccad2026contest/checkpoints/`.
- `dit_gnn_step_141000_loss_1.05.pth` also exists under the same directory.
- `my_optimizer.py` prefers `MY_OPTIMIZER_CHECKPOINT` first; otherwise it sorts `checkpoints/*.pth` by parsed step and prefers `step_141000` over `step_43299`.
- The current `/tmp/senpai_aspect_full.json` history records that the run used `MY_OPTIMIZER_CHECKPOINT=/home/kuotzuwei15/pda/senpai/iccad2026contest/checkpoints/dit_gnn_step_43299_epoch_11_end.pth`. Future comparisons should keep that environment variable explicit.

## Current Performance Baseline

| Metric | Value | Command | Verified? |
|---|---:|---|---|
| Score | 4.498339808 | Saved artifact `/tmp/senpai_aspect_full.json`; compatible run command is `MY_OPTIMIZER_CHECKPOINT=/home/kuotzuwei15/pda/senpai/iccad2026contest/checkpoints/dit_gnn_step_43299_epoch_11_end.pth python iccad2026_evaluate.py --evaluate my_optimizer.py --output /tmp/senpai_aspect_full.json` | Saved artifact parsed in this session; optimizer/evaluator command not rerun |
| Runtime | 12.0550 s average; 50.4962 s weighted average | Same saved artifact | Parsed from saved artifact; local score neutralizes runtime |
| Memory | Unknown | Not measured | Missing |

Additional saved-artifact facts:

- Timestamp: `2026-06-16T19:04:35.112101`.
- Average cost: `4.388497655`.
- Weighted `hpwl_gap`: `2.022872854`.
- Weighted `area_gap`: `2.162657056`.
- Weighted `violations_relative`: `0.185802775`.
- Weighted soft numerator split from read-only recompute: boundary `7.3066`, grouping `3.3990`, MIB `0.1537`, total soft `10.8593`, denominator `58.9547`.
- The 101-120 block bucket contributes `81.13%` of the weighted score.
- The 116-120 block bucket contributes `34.08%` of the weighted score.

## Expected Performance Target

User target:

- Converge to total score `<= 2.0`.

Minimum constraints:

- Preserve 100/100 hard feasibility.
- Keep checkpoint provenance explicit.
- Compare before/after on the same evaluator command or saved artifact.
- Prioritize large cases because the score is dominated by high block counts.

## Gap Analysis

Current total score is `4.4983`, leaving a gap of about `2.4983` points to the target.

The gap cannot be closed by soft-constraint cleanup alone:

- If `violations_relative` were set to zero while HPWL and area stayed unchanged, estimated score would still be about `3.0928`.
- If HPWL gap alone were set to zero, estimated score would be about `3.0276`.
- If area gap alone were set to zero, estimated score would be about `2.9305`.
- If both HPWL and area gaps were set to zero while soft violations stayed unchanged, estimated score would be about `1.4598`.

Uniform reduction estimates:

- With current soft violations, HPWL and area gaps must both shrink to about `17.78%` of current values to reach `<= 2.0`.
- If soft violations are cut in half, HPWL and area gaps still must shrink to about `31.53%` of current values.
- If soft violations are eliminated, HPWL and area gaps still must shrink to about `47.78%` of current values.

Large-case bucket view:

| Bucket | Weight Share | Score | HPWL Gap | Area Gap | V_rel | Weighted Runtime |
|---|---:|---:|---:|---:|---:|---:|
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

The saved artifact stores aggregate `violations_relative`; the boundary/grouping/MIB split above was recovered by read-only recomputation from saved positions and local validation data.

## Observed Bottlenecks

- Weighted HPWL gap remains high at `2.0229`; the current placement has roughly triple the baseline wirelength on a weighted relative basis.
- Weighted area gap remains high at `2.1627`; the current bounding boxes are still roughly `3.16x` the baseline bounding-box area on a weighted relative basis.
- Large cases dominate: cases 101-120 contribute `81.13%` of weighted score and still have score `4.4313`.
- The aspect-ratio attempt improved score only from `4.5867` to `4.4983`; HPWL improved only `2.0405 -> 2.0229`, area improved only `2.1637 -> 2.1627`, and V_rel improved `0.1943 -> 0.1858`.
- Runtime is now expensive on high-block cases: weighted runtime is `50.50 s`, and case 99 took `163.51 s` in the saved run. Local score ignores this, but official runtime risk is rising.
- Boundary violations still dominate the soft numerator: weighted boundary `7.3066`, grouping `3.3990`, MIB `0.1537`.
- `CandidateSelector._proxy_key(...)` still chooses among candidates using raw `hpwl + 0.01 * bbox_area` plus a large soft penalty, rather than normalizing around the official relative HPWL and area gaps.
- The current constructive families create a boundary frame, compact its central column, and optionally reshape blocks, but they do not perform post-selection geometry improvement on the selected high-weight large cases.
- The differentiable training proxy in `iccad2026_evaluate.py` models HPWL and area gaps plus overlap/area-tolerance proxy terms; it does not model boundary, grouping, MIB, fixed/preplaced post-legalization effects, candidate selection, or official runtime. A lower-loss checkpoint can improve guidance, but it cannot directly optimize the final legalized candidate.

## Likely Causes

### Cause 1: Candidate topology is structurally narrow after boundary-frame compaction

- Type: Algorithmic limitation / search-control limitation.
- Evidence: The aspect-ratio family added shape variation but delivered only a small score gain (`4.5867 -> 4.4983`) and almost no area-gap movement (`2.1637 -> 2.1627`). The selected layouts are still generated from rigid frame rails plus bottom-left interior packing. Current top contributors, especially cases 99, 95, and 94, retain large HPWL and area gaps despite full feasibility.
- Affected modules: `ConstructiveCandidateLegalizer._boundary_frame_candidate`, `_boundary_frame_compaction_candidates`, `_boundary_frame_compaction_candidate`, `_pack_connected_bottom_left_unit_origins`, `_place_boundary_rail_units`.
- Risk: Medium to high. Broader topology search can improve score but may increase runtime and destabilize soft constraints.
- Confidence: High.

### Cause 2: Candidate selection proxy is mismatched to the official objective

- Type: Benchmark/evaluator mismatch / parameter tuning issue.
- Evidence: `CandidateSelector._proxy_key(...)` uses raw HPWL plus `0.01 * bbox_area`, while the evaluator uses relative gaps against per-case baselines and multiplies by `exp(2 * V_rel)`. The proxy can prefer a candidate that is raw-HPWL smaller but poor on relative area, or overpay soft cleanup when HPWL and area dominate the path to `<= 2.0`.
- Affected modules: `CandidateSelector._proxy_key`, `ConstructiveCandidateLegalizer.PROXY_BBOX_WEIGHT`, `ConstructiveCandidateLegalizer.PROXY_SOFT_WEIGHT`.
- Risk: Medium. Better normalization may improve selection among existing candidates with small edit scope, but approximate baselines inside `solve()` are unavailable unless inferred from inputs or candidate scales.
- Confidence: High.

### Cause 3: ML checkpoint loss is not the final legalized-score bottleneck

- Type: Training/evaluator mismatch.
- Evidence: The user identifies `dit_gnn_step_43299_epoch_11_end.pth` as relatively low loss, but the final saved score remains `4.4983`. The training proxy omits boundary/grouping/MIB and does not represent the candidate selector or legalizer topology. The optimizer uses diffusion predictions as advisory order/centers, then legalizes through hand-built candidates; the legalizer geometry now dominates the output.
- Affected modules: `training_example.py`, `iccad2026contest/iccad2026_evaluate.py` training-loss helpers, `DiffusionGuidanceAdapter`, `ConstructiveCandidateLegalizer`.
- Risk: Low for diagnosis, high for any training-only fix. More training is unlikely to close the score gap unless the legalizer can exploit better predictions.
- Confidence: Medium to high.

## Optimization Hypotheses

| Priority | Hypothesis | Expected Impact | Risk | Affected Files | Verification |
|---:|---|---|---|---|---|
| 1 | Add a bounded large-case post-selection local repack/relaxation candidate: for `n >= 100`, keep anchors and boundary rail satisfaction fixed where possible, rip up a small set of high-cost interior or rail-adjacent units from the currently best constructive candidate, and reinsert them with connection-aware bottom-left candidates scored by normalized HPWL/area delta. | High. Directly targets the measured HPWL and area gaps in the weighted 101-120 cases, where soft-only work cannot reach `<= 2.0`. | High. More search can increase runtime and can regress grouping/boundary unless every candidate is checker-gated and candidate count is tightly bounded. | `iccad2026contest/my_optimizer.py` | Compile, validate, single-case eval, full eval with explicit checkpoint; compare score, HPWL gap, area gap, V_rel, runtime, and feasibility against `/tmp/senpai_aspect_full.json`. |
| 2 | Replace `CandidateSelector._proxy_key(...)` with a scale-normalized proxy that approximates official relative HPWL and area terms across candidates, and tune soft penalty so it cannot dominate when quality gaps are large. | Medium. Could select better candidates already being generated without adding much runtime. | Medium. Without true evaluator baselines in `solve()`, normalization is approximate and can regress if weights are wrong. | `iccad2026contest/my_optimizer.py` | Full eval before/after; compare candidate-selected score components and runtime. |
| 3 | Add a second bounded aspect/topology profile targeted at the current worst contributors: compact case-99-like area by increasing central-column width/height tradeoff diversity and rail ordering variants, not just block aspect ratios. | Medium. Targets the high area gaps on cases 99, 90, 83, 88, and 79. | Medium to high. Additional candidate families increase runtime and may duplicate the narrow topology problem. | `iccad2026contest/my_optimizer.py` | Full eval with per-case component comparison; rollback if runtime rises without area and HPWL improvement. |

## Recommended First Optimization

Implement Hypothesis 1 first: a bounded large-case post-selection local repack/relaxation candidate.

Concrete scope:

- Run only for large cases, initially `block_count >= 100`.
- Start from a hard-feasible constructive candidate, preferably the current selector winner or the best frame/aspect candidate.
- Keep fixed and preplaced blocks immutable.
- Preserve exact soft-block areas and existing MIB-shared dimensions.
- Keep boundary rail blocks fixed in the first version unless moving one preserves its required edge touch.
- Identify a small rip-up set using local evidence: high-connectivity interior units, units with large connection cost to already placed neighbors/pins, and units adjacent to large bounding-box expansion.
- Reinsert the rip-up set using existing connection-aware bottom-left candidate generation, but score reinsertion by a normalized local proxy for HPWL plus bbox area rather than raw coordinates.
- Gate every generated candidate through `FeasibilityChecker` and existing candidate selection.
- Cap the number of rip-up units and retries so case-99 runtime does not grow without bound.

Rollback condition:

- Revert if 100/100 feasibility is lost, total score does not improve over `4.498339808`, weighted HPWL gap does not improve, weighted area gap does not improve, or weighted runtime grows substantially without score improvement.

This is the smallest next step that attacks the measured HPWL plus area bottleneck without betting on another training run. The current result is not blocked by hard feasibility, and soft cleanup alone cannot reach the requested score.

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
- Stop if the local repack pass worsens total score, weighted HPWL gap, or weighted area gap after full evaluation.
- Stop if large-case runtime grows substantially without score improvement.
- Stop if the next measurement shows the current artifact did not actually use `dit_gnn_step_43299_epoch_11_end.pth`.

## Risks and Warnings

- The current score target is aggressive. At current soft violations, HPWL and area gaps need to fall to about `17.78%` of their current values to reach `<= 2.0`.
- The 101-120 block bucket dominates the score, so small-case improvements are mostly noise.
- Official runtime may penalize the current approach more than local validation does, because local validation neutralizes `RuntimeFactor`.
- The low-loss checkpoint is not sufficient evidence of final quality because final placements are legalizer-selected and the training loss is only a proxy.
- Candidate-source attribution is not stored in `/tmp/senpai_aspect_full.json`; exact winner-source diagnosis would require instrumentation or rerunning with added logging, which was not done in this pass.

## Open Questions

- Which candidate family wins on the worst weighted cases after the aspect-ratio change?
- Can candidate-source logging be added or enabled without excessive output to make future diagnosis more direct?
- Is it acceptable to spend more runtime on cases `n >= 116` if score improves, given hidden official runtime normalization may differ?
- Should the next implementation add a small result-analysis script so component and soft-subcategory recomputes are repeatable without ad hoc commands?
