# Performance Diagnosis

## Purpose

Diagnose why the current checkpoint-enabled optimizer result in `iccad2026contest/eval_full_after_training.json` is still far above the requested target score of `2.0` or below.

User-provided checkpoint context:

- `iccad2026contest/checkpoints/dit_gnn_step_43299_epoch_11_end.pth`
- File status: present locally, 128 MB.

No optimizer implementation code was changed during this diagnosis.

## Diagnosis Scope

Covered:

- Saved evaluator artifact: `iccad2026contest/eval_full_after_training.json`.
- Current checkpoint inventory under `iccad2026contest/checkpoints/`.
- `iccad2026contest/my_optimizer.py` checkpoint loading, square soft-block dimension normalization, constructive candidate generation, boundary-frame compaction, connection-aware placement, candidate selection, and `solve()` orchestration.
- `iccad2026contest/iccad2026_evaluate.py` cost formula, hard/soft constraint semantics, runtime handling, and block-count weighting.
- Planning and quality-gate documents under `doc/`.

Not covered:

- The evaluator was not rerun.
- Training was not run.
- The checkpoint tensor was not loaded or inspected.
- No profiler or memory measurement was run.
- Soft violation subcategories were not recomputed from the dataset in this pass because the saved JSON stores only aggregate `violations_relative`.

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
- `iccad2026contest/eval_full_after_training.json`
- `iccad2026contest/my_optimizer.py`
- `iccad2026contest/iccad2026_evaluate.py`
- `iccad2026contest/training_example.py`

## Current Correctness Status

The saved artifact reports stable hard feasibility:

- `iccad2026contest/eval_full_after_training.json`: 100/100 feasible.
- Capped feasible cases: 0/100.
- No evaluator command was rerun in this diagnosis, so hard-violation subcounts were not independently reverified.

Correctness conclusion:

- Basic correctness is stable enough for optimization diagnosis.
- The current bottleneck is solution quality, not hard feasibility.
- Any optimization attempt must preserve 100/100 local validation feasibility.

Checkpoint provenance caveat:

- `dit_gnn_step_43299_epoch_11_end.pth` exists under `iccad2026contest/checkpoints/`.
- `dit_gnn_step_141000_loss_1.05.pth` also exists under the same directory.
- `my_optimizer.py` prefers `MY_OPTIMIZER_CHECKPOINT` first; otherwise it sorts `checkpoints/*.pth` by parsed step and prefers `step_141000` over `step_43299`.
- The saved JSON does not record which checkpoint was loaded. Runs intended to test `43299` should set `MY_OPTIMIZER_CHECKPOINT=/home/kuotzuwei15/pda/senpai/iccad2026contest/checkpoints/dit_gnn_step_43299_epoch_11_end.pth`.

## Current Performance Baseline

| Metric | Value | Command | Verified? |
|---|---:|---|---|
| Score | 4.586686865 | Saved `iccad2026contest/eval_full_after_training.json`; compatible run command is `MY_OPTIMIZER_CHECKPOINT=/home/kuotzuwei15/pda/senpai/iccad2026contest/checkpoints/dit_gnn_step_43299_epoch_11_end.pth python iccad2026_evaluate.py --evaluate my_optimizer.py --output eval_full_after_training.json` | Saved artifact parsed in this session; evaluator command not rerun |
| Runtime | 15.3444 s average; 56.2639 s weighted average | Same saved artifact | Parsed from saved artifact; local score neutralizes runtime |
| Memory | Unknown | Not measured | Missing |

Additional saved-artifact facts:

- Timestamp: `2026-06-16T06:39:07.637929`.
- Average cost: `4.420442148`.
- Weighted `hpwl_gap`: `2.040539561`.
- Weighted `area_gap`: `2.163723306`.
- Weighted `violations_relative`: `0.194281466`.
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

Current total score is `4.5867`, leaving a gap of about `2.5867` points to the target.

The gap cannot be closed by soft-constraint cleanup alone:

- If `violations_relative` were set to zero while HPWL and area stayed unchanged, estimated score would still be about `3.1021`.
- If HPWL gap alone were set to zero, estimated score would be about `3.0786`.
- If area gap alone were set to zero, estimated score would be about `2.9926`.
- If both HPWL and area gaps were set to zero while soft violations stayed unchanged, estimated score would be about `1.4845`.

Uniform reduction estimates:

- With current soft violations, HPWL and area gaps must both shrink to about `16.62%` of current values to reach `<= 2.0`.
- If soft violations are cut in half, HPWL and area gaps still must shrink to about `30.73%` of current values.
- If soft violations are eliminated, HPWL and area gaps still must shrink to about `47.57%` of current values.

Large-case bucket view:

| Bucket | Weight Share | Score | HPWL Gap | Area Gap | V_rel | Weighted Runtime |
|---|---:|---:|---:|---:|---:|---:|
| 101-120 | 0.8113 | 4.5281 | 1.9949 | 2.1797 | 0.1897 | 66.1907 s |
| 116-120 | 0.3408 | 4.4221 | 1.9551 | 2.2225 | 0.1777 | 100.8844 s |

Top weighted contributors:

| Test ID | Blocks | Cost | Weight Share | HPWL Gap | Area Gap | V_rel |
|---:|---:|---:|---:|---:|---:|---:|
| 99 | 120 | 5.1355 | 0.0800 | 2.0849 | 3.0937 | 0.1791 |
| 98 | 119 | 4.5953 | 0.0736 | 1.8213 | 2.1988 | 0.2115 |
| 95 | 116 | 5.1020 | 0.0573 | 2.9493 | 2.1439 | 0.1818 |
| 94 | 115 | 5.1200 | 0.0527 | 3.2845 | 2.4786 | 0.1385 |
| 97 | 118 | 3.7725 | 0.0677 | 1.6715 | 1.3861 | 0.2000 |

## Benchmark or Evaluator Details

The local evaluator computes feasible cost as:

```text
Cost = (1 + 0.5 * (max(0, HPWL_gap) + max(0, Area_gap))) * exp(2 * V_rel) * RuntimeAdjustment
```

Hard infeasibility costs exactly `10.0`, while feasible costs are capped below `10.0`.

The local full evaluation neutralizes runtime by using `RuntimeFactor = 1.0`, so runtime is recorded but does not explain the saved local score.

The total score is an exponentially weighted average:

```text
Total Score = sum_i Cost[i] * exp(n_i / 12) / sum_j exp(n_j / 12)
```

Soft violations are:

```text
V_rel = (boundary + grouping + MIB) / N_soft
```

The saved artifact contains positions and aggregate gaps for all cases, but not the individual boundary/grouping/MIB counts.

## Observed Bottlenecks

- Weighted HPWL gap remains high at `2.0405`; the current placement has roughly triple the baseline wirelength on a weighted relative basis.
- Weighted area gap remains high at `2.1637`; the current bounding boxes are still roughly `3.16x` the baseline bounding-box area on a weighted relative basis.
- The high-block-count cases dominate the score: cases 101-120 contribute `81.13%` of the weighted score.
- Runtime is rising as candidate families are added: the saved artifact reports `15.3444 s` average and `56.2639 s` weighted runtime.
- `HardConstraintNormalizer._soft_dimensions(...)` still makes every non-fixed soft block square, even though the contest permits arbitrary soft-block aspect ratios as long as area is preserved.
- The current boundary-frame compaction still uses square unit dimensions and fixed central-width hints. Boundary rail lengths are therefore constrained by sums of square block widths or heights.
- `CandidateSelector._proxy_key(...)` chooses among candidates using raw HPWL plus `0.01 * bbox_area` plus a large soft penalty, rather than a normalized proxy for the official relative HPWL and area gaps.
- The training loss is a proxy that omits final placement constraints such as fixed/preplaced, MIB, cluster, and boundary constraints. A lower-loss checkpoint can improve guidance, but it does not directly optimize the final legalizer output.

## Likely Causes

### Cause 1: Square-only soft-block dimensions limit frame compactness

- Type: Algorithmic limitation / parameterization issue.
- Evidence: `HardConstraintNormalizer._soft_dimensions(...)` returns `(sqrt(area), sqrt(area))` for every soft block. The contest explicitly allows arbitrary aspect ratios for soft blocks. Current area gap is still `2.1637`, and top/bottom boundary rails require side-by-side width while left/right rails require stacked height, making square dimensions a direct frame-size bottleneck.
- Affected modules: `HardConstraintNormalizer._soft_dimensions`, `ConstructiveCandidateLegalizer._build_units`, `_boundary_frame_candidate`, `_boundary_frame_compaction_candidate`, `FeasibilityChecker`.
- Risk: Medium to high. Shape changes must preserve exact area, fixed/preplaced immutability, MIB consistency, grouping behavior, and non-overlap.
- Confidence: High.

### Cause 2: Boundary-frame search is still structurally narrow on large cases

- Type: Algorithmic limitation / search-control limitation.
- Evidence: The frame-compaction candidate improved score from `4.9125` to `4.5867`, but 101-120 block cases still score `4.5281` with area gap `2.1797`. The current compaction tries bounded central-width hints and bottom-left interior repacking, but it cannot change unit shapes and still places rail/corner units in a rigid frame.
- Affected modules: `ConstructiveCandidateLegalizer._boundary_frame_compaction_candidates`, `_frame_compaction_central_width_hints`, `_pack_connected_bottom_left_unit_origins`, `_place_boundary_rail_units`.
- Risk: Medium. Broader search can improve quality but may further increase runtime.
- Confidence: High.

### Cause 3: Candidate selection and training optimize proxies, not the final score

- Type: Benchmark/evaluator mismatch / parameter tuning issue.
- Evidence: The selector proxy uses raw HPWL and raw bbox area with fixed constants, while the evaluator uses relative gaps against per-case baselines. The training proxy also omits final soft constraints. The saved result is feasible and no longer capped, so marginal candidate choice quality now matters more than basic feasibility.
- Affected modules: `CandidateSelector._proxy_key`, `training_example.py`, `DiffusionGuidanceAdapter`, `ConstructiveCandidateLegalizer`.
- Risk: Medium. Better normalization may select better existing candidates, but bad proxy weights can regress total score.
- Confidence: Medium.

## Optimization Hypotheses

| Priority | Hypothesis | Expected Impact | Risk | Affected Files | Verification |
|---:|---|---|---|---|---|
| 1 | Add a bounded boundary-aware soft-block aspect-ratio candidate family for non-fixed/non-preplaced soft blocks, preserving exact area and applying one shared shape per MIB group. Try a small set of profiles that make top/bottom rail blocks narrower, left/right rail blocks shorter, and interior blocks use compactness-oriented ratios. | High. Directly targets both area gap `2.1637` and HPWL gap `2.0405`, especially in 101-120 cases where boundary-frame geometry dominates. | Medium to high. Dimension changes can regress MIB, grouping, HPWL, or feasibility unless checker and selector gated. | `iccad2026contest/my_optimizer.py` | Compile, validate, single-case eval, full eval with explicit checkpoint; compare score, HPWL gap, area gap, V_rel, runtime, and feasibility. |
| 2 | Normalize `CandidateSelector._proxy_key(...)` around problem-local scale estimates so candidate choice better approximates the official relative HPWL and area terms instead of raw `hpwl + 0.01 * bbox_area`. | Medium. May improve selection among already generated candidates without increasing candidate count much. | Medium. Without true evaluator baselines in `solve()`, normalization is approximate and weight-sensitive. | `iccad2026contest/my_optimizer.py` | Full eval before/after; compare candidate-selected score components and runtime. |
| 3 | Add a bounded large-case local repack pass for the worst weighted cases: keep boundary rails fixed, rip up a small set of high-connectivity interior units, and reinsert them with connection-aware bottom-left candidates. | Medium to high. Targets HPWL after frame compaction. | High. More search increases runtime and can destabilize grouping compactness. | `iccad2026contest/my_optimizer.py` | Full eval with runtime tracking; rollback if runtime rises without score improvement. |

## Recommended First Optimization

Implement Hypothesis 1 first: a bounded boundary-aware soft-block aspect-ratio candidate family.

Concrete scope:

- Do not change fixed-shape or preplaced blocks.
- Preserve each soft block's exact area.
- For MIB groups, apply a shared width/height per group so MIB violations do not increase by construction.
- Keep grouping macros intact where possible; do not split existing cluster units in the first attempt.
- Generate a small number of shape profiles, for example:
  - top/bottom boundary blocks: narrower and taller;
  - left/right boundary blocks: wider and shorter;
  - corner blocks: balanced or two symmetric variants;
  - unconstrained interior blocks: one or two compactness-oriented ratios.
- Feed each shaped profile through the existing constructive frame/compaction path, then rely on `FeasibilityChecker` and `CandidateSelector` to accept only hard-feasible improvements.
- Do not add training changes, broad local search, or new dependencies in this first attempt.

Rollback condition:

- Revert if 100/100 feasibility is lost, total score does not improve over `4.586686865`, weighted area gap does not improve, or runtime grows substantially without score improvement.

This is the smallest next step that attacks the measured HPWL plus area bottleneck. Soft-constraint-only work cannot reach the target score, and more checkpoint training is unlikely to close the gap while final legalizer geometry remains square-only.

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
- Stop if the aspect-ratio candidate family worsens total score or weighted area gap after full evaluation.
- Stop if average or large-case runtime grows substantially without score improvement.
- Stop before running training, installing dependencies, downloading data, or deleting/moving checkpoints without approval.
- Stop if a run intended to use `dit_gnn_step_43299_epoch_11_end.pth` cannot prove it loaded that checkpoint.

## Risks and Warnings

- The current baseline is a saved artifact parsed in this session, not an evaluator rerun.
- Runtime is neutralized locally but matters on the official leaderboard; current saved runtime is already high on large cases.
- The default checkpoint loader prefers `dit_gnn_step_141000_loss_1.05.pth` unless `MY_OPTIMIZER_CHECKPOINT` is set.
- The score target `<= 2.0` requires large reductions in both HPWL and area gaps. Soft-constraint cleanup by itself is insufficient.
- Aspect-ratio changes are allowed by the contest but more invasive than coordinate-only packing changes; they must be checker-gated and easy to roll back.

## Open Questions

- Was `MY_OPTIMIZER_CHECKPOINT` set when `iccad2026contest/eval_full_after_training.json` was generated?
- Should the evaluator output or a sidecar note record the checkpoint path and selected candidate source for each run?
- Should candidate-source instrumentation be added before broader search, or only after the aspect-ratio attempt if results are ambiguous?
- How much official runtime penalty is acceptable for a local score improvement?
