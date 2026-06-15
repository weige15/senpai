# Performance Diagnosis

## Purpose

Diagnose why `iccad2026contest/eval_full_after_training.json` remains well above the requested target score of `2.0` or below, using the user-provided low-loss checkpoint context:

- Checkpoint path provided by user: `iccad2026contest/checkpoints/dit_gnn_step_43299_epoch_11_end.pth`
- File status: present locally at that path, about 128 MB

No optimizer implementation code was changed during this diagnosis.

## Diagnosis Scope

Covered:

- Saved evaluator artifact: `iccad2026contest/eval_full_after_training.json`.
- Checkpoint inventory under `iccad2026contest/checkpoints/`.
- `iccad2026contest/my_optimizer.py` checkpoint loading, hard normalizer, constructive legalizer, boundary-frame candidate, connection-aware origin scoring, candidate selector, and fallback path.
- `iccad2026contest/iccad2026_evaluate.py` score formula, hard-feasibility checks, soft-violation normalization, runtime handling, and block-count weighting.
- `iccad2026contest/training_example.py` and README training-loss notes.
- `noML_optimizer.py` as a read-only reference for unported candidate families.

Not covered:

- No optimizer/evaluator run was executed.
- No training command was run.
- No checkpoint tensor was loaded for inspection.
- No hidden-test behavior was measured.
- No profiler or memory measurement was run.

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
- `noML_optimizer.py`

## Current Correctness Status

The saved artifact is hard-feasible:

- `iccad2026contest/eval_full_after_training.json`: 100/100 feasible.
- Read-only recomputation of the saved positions with `ContestEvaluator` and `evaluate_solution(...)` found:
  - overlap violations: `0`
  - soft-block area violations: `0`
  - fixed/preplaced dimension or position violations: `0`

Correctness conclusion:

- Basic correctness is stable enough for optimization diagnosis.
- The current bottleneck is solution quality, not hard feasibility.
- Any optimization attempt must preserve 100/100 feasibility.

Checkpoint provenance caveat:

- `iccad2026contest/checkpoints/dit_gnn_step_43299_epoch_11_end.pth` exists.
- `iccad2026contest/checkpoints/dit_gnn_step_141000_loss_1.05.pth` also exists.
- `my_optimizer.py` loads an explicit `MY_OPTIMIZER_CHECKPOINT` first, otherwise sorts `checkpoints/*.pth` by parsed step and therefore prefers `step_141000` over `step_43299`.
- The saved JSON does not record which checkpoint was loaded. Future measurements intended to use `43299` should set `MY_OPTIMIZER_CHECKPOINT=/home/kuotzuwei15/pda/senpai/iccad2026contest/checkpoints/dit_gnn_step_43299_epoch_11_end.pth`.

## Current Performance Baseline

| Metric | Value | Command | Verified? |
|---|---:|---|---|
| Score | 4.912480450 | Saved `iccad2026contest/eval_full_after_training.json`; compatible run command is `MY_OPTIMIZER_CHECKPOINT=/home/kuotzuwei15/pda/senpai/iccad2026contest/checkpoints/dit_gnn_step_43299_epoch_11_end.pth python iccad2026_evaluate.py --evaluate my_optimizer.py --output eval_full_after_training.json` | Saved artifact parsed and recomputed in this session; optimizer/evaluator command not rerun |
| Runtime | 10.7188 s average in saved JSON; 31.6290 s weighted average | Same saved artifact | Parsed from saved artifact; local score neutralizes runtime |
| Memory | Unknown | Not measured | Missing |

Additional saved-artifact facts:

- Timestamp: `2026-06-16T05:29:57.775120`.
- Feasible cases: 100/100.
- Capped feasible cases: 0/100.
- Average cost: `4.494760179`.
- Weighted `hpwl_gap`: `2.261854903`.
- Weighted `area_gap`: `2.305821333`.
- Weighted `violations_relative`: `0.202297801`.
- Weighted soft counts from recomputation:
  - boundary: `7.954748777`
  - grouping: `3.325242990`
  - MIB: `0.536719077`
  - total soft: `11.816710844`
  - max soft denominator: `58.954724081`
- Weighted packing density of saved placements: `0.303169980`.
- Weighted packing density of ground-truth baseline placements: `0.965993286`.
- All 6,110 movable soft blocks in the saved solution are square.
- About 93.31% of corresponding movable soft blocks are non-square in ground truth.

## Expected Performance Target

User target:

- Converge to total score `<= 2.0`.

Minimum constraints for optimization:

- Preserve 100/100 local validation hard feasibility.
- Compare before/after on the same evaluator command or equivalent saved artifact.
- Keep checkpoint provenance explicit.
- Prioritize large cases because the 101-120 block bucket contributes `81.13%` of current weighted score.

## Gap Analysis

Current total score is `4.9125`, so the gap to target is about `2.9125` points.

This is not a feasibility problem:

- Hard violations are zero.
- There are no capped cases.

This is also not only a soft-constraint problem:

- If all soft violations were set to zero while HPWL and area stayed unchanged, estimated score would still be about `3.2838`.
- If boundary violations alone were set to zero, estimated score would be about `3.7629`.
- If grouping violations alone were set to zero, estimated score would be about `4.3787`.
- If MIB violations alone were set to zero, estimated score would be about `4.8258`.

The dominant bottleneck is joint HPWL and bounding-box quality:

- If HPWL gap alone were set to zero, estimated score would be about `3.2273`.
- If area gap alone were set to zero, estimated score would be about `3.1970`.
- If both HPWL and area gaps were set to zero while soft violations stayed unchanged, estimated score would be about `1.5118`.

Uniform reduction estimates:

- With current soft violations, HPWL and area gaps must both shrink to about `14.36%` of current values to reach `<= 2.0`.
- If soft violations are cut in half, HPWL and area gaps still must both shrink to about `27.79%` of current values.
- If soft violations are eliminated, HPWL and area gaps still must both shrink to about `43.79%` of current values.

Top weighted contributors:

| Test ID | Blocks | Cost | Weight Share | HPWL Gap | Area Gap | V_rel |
|---:|---:|---:|---:|---:|---:|---:|
| 99 | 120 | 5.2716 | 0.0800 | 2.4130 | 3.1792 | 0.1642 |
| 98 | 119 | 5.0467 | 0.0736 | 2.1632 | 2.1988 | 0.2308 |
| 95 | 116 | 5.9394 | 0.0573 | 3.4919 | 2.7657 | 0.1818 |
| 97 | 118 | 4.6236 | 0.0677 | 2.1438 | 1.8515 | 0.2167 |
| 94 | 115 | 5.4420 | 0.0527 | 3.6158 | 2.6355 | 0.1385 |

Large-case buckets:

| Bucket | Weight Share | Score | HPWL Gap | Area Gap | V_rel | Weighted Runtime |
|---|---:|---:|---:|---:|---:|---:|
| 101-120 | 0.8113 | 4.9223 | 2.2568 | 2.3455 | 0.2002 | 35.7785 s |
| 116-120 | 0.3408 | 4.9513 | 2.3689 | 2.5424 | 0.1788 | 54.8936 s |

## Benchmark or Evaluator Details

The local evaluator computes feasible cost as:

```text
Cost = (1 + 0.5 * (max(0, HPWL_gap) + max(0, Area_gap))) * exp(2 * V_rel) * RuntimeAdjustment
```

Hard infeasibility costs exactly `10.0`, while feasible costs are capped below `10.0`.

The local full evaluation neutralizes runtime when recomputing final costs with `RuntimeFactor = 1.0`, so runtime is recorded but does not explain the saved local score.

The total score is an exponentially weighted average:

```text
Total Score = sum_i Cost[i] * exp(n_i / 12) / sum_j exp(n_j / 12)
```

Soft violations are:

```text
V_rel = (boundary + grouping + MIB) / N_soft
```

The saved artifact has `positions` for all cases, enabling read-only component recomputation without rerunning the optimizer.

## Observed Bottlenecks

- The connection-aware origin scoring attempt improved score from about `5.1467` to `4.9125`, but HPWL and area gaps remain above `2.2` on a weighted basis.
- Packing density is extremely low: weighted saved-placement density is about `0.30`, while the baseline density is about `0.97`.
- The simple boundary-frame candidate in `my_optimizer.py` packs interior units with a shelf strategy and fixed central-width heuristic. It does not run a bounded frame compaction pass.
- The 101-120 bucket contributes `81.13%` of the weighted score and still scores about `4.9223`.
- The current connection-aware origin scoring now considers B2B/P2B distances during constructive placement, so the previous most obvious HPWL-blind construction bottleneck is no longer the first target.
- `HardConstraintNormalizer._soft_dimensions(...)` still uses square dimensions for all non-fixed soft blocks, which leaves no aspect-ratio search.
- The training proxy explicitly omits several final-evaluation constraints and cannot directly optimize the legalizer's final sparse packing behavior.

## Likely Causes

### Cause 1: Sparse boundary-frame and shelf packing on large cases

- Type: Algorithmic limitation / search-control limitation.
- Evidence: Weighted packing density is `0.3032` versus ground-truth baseline density `0.9660`; weighted area gap is `2.3058`. The current `ConstructiveCandidateLegalizer._boundary_frame_candidate(...)` computes a single frame and places interior units through `_pack_shelf_unit_origins(...)`; no large-case frame compaction is present. The 101-120 bucket accounts for `81.13%` of score and still scores `4.9223`.
- Affected modules: `ConstructiveCandidateLegalizer._boundary_frame_candidate`, `ConstructiveCandidateLegalizer._pack_shelf_unit_origins`, `CandidateSelector.best_feasible`, `MyOptimizer.solve`.
- Risk: Medium. More candidate generation can improve density but may increase runtime and can regress soft-boundary satisfaction if not checker/selector guarded.
- Confidence: High.

### Cause 2: Soft-block shape search is still too narrow

- Type: Algorithmic limitation / parameterization issue.
- Evidence: `HardConstraintNormalizer._soft_dimensions(...)` returns square dimensions for every soft movable block. In the saved solution, all 6,110 movable soft blocks are square, while about 93.31% of corresponding ground-truth movable soft blocks are non-square. The contest permits arbitrary soft-block aspect ratios as long as area tolerance is preserved.
- Affected modules: `HardConstraintNormalizer._soft_dimensions`, `ConstructiveCandidateLegalizer._build_units`, candidate generation, candidate selection, feasibility checking.
- Risk: Medium to high. Shape changes must preserve area, fixed/preplaced immutability, non-overlap, and MIB behavior; a shape search can easily trade one quality term against another.
- Confidence: High.

### Cause 3: Checkpoint loss is optimizing a proxy, not final post-legalization score

- Type: Benchmark/evaluator mismatch / model-guidance limitation.
- Evidence: README and `compute_training_loss(...)` document that training loss is a proxy and does not check all placement constraints, including fixed, MIB, cluster, and boundary. In `my_optimizer.py`, model output is advisory; final coordinates are produced by deterministic legalizers and candidate selection. The lower-loss `43299` checkpoint can improve guidance, but it cannot by itself fix sparse frame/shelf packing or missing shape/aspect search.
- Affected modules: `training_example.py`, `DiffusionGuidanceAdapter`, `ConstructiveCandidateLegalizer`, checkpoint-loading path in `MyOptimizer.__init__`.
- Risk: Low for diagnosis, medium for future training changes. Further training may produce diminishing returns unless the legalizer/search target is improved.
- Confidence: Medium.

## Optimization Hypotheses

| Priority | Hypothesis | Expected Impact | Risk | Affected Files | Verification |
|---:|---|---|---|---|---|
| 1 | Add one bounded large-case frame-compaction candidate family that reuses the current boundary-frame grouping, tries 2-4 central-width hints for `n >= 100`, repacks interior units with connection-aware bottom-left or shelf placement, and lets the existing checker/selector accept only feasible improvements. | High. Directly targets weighted area gap `2.3058`, low density `0.3032`, and the 101-120 bucket that contributes `81.13%` of score; can also reduce HPWL by compacting connected interiors. | Medium. Candidate generation can raise runtime; boundary/grouping soft counts may regress if the candidate selector proxy is weak. | `iccad2026contest/my_optimizer.py` | `python -B -m py_compile iccad2026contest/my_optimizer.py`; validate; single-case eval; full eval; compare total score, weighted area gap, HPWL, V_rel, density, and runtime. |
| 2 | Add bounded soft-block aspect-ratio trials for ordinary non-fixed, non-preplaced, non-MIB single-block units, preserving exact area and relying on full feasibility checks before selection. | Medium to high. Targets the all-square shape limitation and may improve both area and HPWL after compaction exists. | Medium to high. Shape changes interact with packing, MIB, grouping, and hard area tolerance. | `iccad2026contest/my_optimizer.py` | Validate hard feasibility; compare area gap, HPWL, MIB violations, total score, and runtime. |
| 3 | Add a normalized selector proxy that better balances HPWL, bbox area, and soft violations across candidate families, using problem-local scale estimates instead of raw `hpwl + 0.01 * bbox_area + 1000 * soft_relative`. | Medium. May let already generated candidates win when they improve the true score proxy. | Medium. Without official baselines inside `solve()`, normalization must be approximate; bad weights can regress score. | `iccad2026contest/my_optimizer.py` | Run full eval and compare candidate-family behavior through score components. |

## Recommended First Optimization

Implement Hypothesis 1 first: add one bounded large-case frame-compaction candidate family.

Concrete scope:

- Keep current dimensions, checkpoint handling, hard checker, fallback, connection-aware origin scoring, and candidate selector.
- Add a small candidate family inside `ConstructiveCandidateLegalizer` for `problem.block_count >= 100` and cases with boundary-constrained units.
- Start from the current boundary-frame grouping, then try a few central-width hints, such as scaled current width and aspect-derived widths.
- Repack interior units more compactly within the central area, preferably using the existing connection index when choosing origins.
- Place boundary rail/corner units on the frame edges without overlap.
- Generate only a small number of candidates, dedupe by position signature, and pass them through existing `FeasibilityChecker` and `CandidateSelector`.
- Do not add aspect-ratio changes, MIB sync, broad local search, rip-up/repack, training changes, or checkpoint changes in this first attempt.

Rollback condition:

- Revert if 100/100 feasibility is lost, total score does not improve over `4.912480450`, weighted area gap does not improve, or average/large-case runtime grows substantially without score improvement.

This is the smallest next step that targets the strongest measured bottleneck while avoiding the higher hard-constraint risk of changing block dimensions.

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
- Stop if frame-compaction candidates increase total score or weighted area gap after full evaluation.
- Stop if runtime grows substantially on 101-120 block cases without score improvement.
- Stop before running training, installing dependencies, downloading data, or deleting/moving checkpoints without approval.
- Stop if a run intended to use `dit_gnn_step_43299_epoch_11_end.pth` cannot prove it loaded that checkpoint.

## Risks and Warnings

- The current baseline is a saved artifact plus read-only recomputation, not an optimizer rerun during this diagnosis.
- Runtime is neutralized locally but may matter on the official leaderboard; current saved JSON reports `10.7188 s` average and `31.6290 s` weighted runtime.
- The default checkpoint loader will prefer `dit_gnn_step_141000_loss_1.05.pth` unless `MY_OPTIMIZER_CHECKPOINT` is set to the `43299` path.
- The score target `<= 2.0` is unlikely to be reached by soft-constraint cleanup alone. It requires large reductions in both HPWL and area gaps.
- Changing soft-block aspect ratios is promising, but it should follow a packing-density improvement because sparse layouts can dominate area gap even with better shapes.

## Open Questions

- Was `MY_OPTIMIZER_CHECKPOINT` set when `iccad2026contest/eval_full_after_training.json` was generated?
- Should future full runs always write the checkpoint path into a sidecar note or result filename to avoid provenance ambiguity?
- What score does `noML_optimizer.py` achieve under the same validation command and checkpoint-free conditions?
- Should future training include legalizer-aware or post-legalization losses after the constructive candidate set is less sparse?
