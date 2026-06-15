# Performance Diagnosis

## Purpose

Diagnose why `iccad2026contest/eval_full_after_training.json` is still far from the requested score target of `2.0` or below, using the user-provided low-loss checkpoint context:

- Claimed checkpoint path: `iccad2026contest/checkpoints/dit_gnn_step_43299_epoch_11_end.pth`
- Actual file found locally: `iccad2026contest/dit_gnn_step_43299_epoch_11_end.pth`

No optimizer implementation code was changed during this diagnosis.

## Diagnosis Scope

Covered:

- Saved evaluator artifact: `iccad2026contest/eval_full_after_training.json`.
- Checkpoint inventory under `iccad2026contest/` and `iccad2026contest/checkpoints/`.
- `iccad2026contest/my_optimizer.py` checkpoint loading, model-guidance adapter, constructive candidate generation, boundary-frame candidate, soft improver, and candidate selector.
- `iccad2026contest/iccad2026_evaluate.py` score formula, hard-feasibility checks, soft-violation normalization, runtime handling, and block-count weighting.
- `iccad2026contest/training_example.py` checkpoint naming and training-loss path.
- `noML_optimizer.py` as a read-only reference for candidate families not yet ported into `my_optimizer.py`.

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
- `README.md`
- `iccad2026contest/eval_full_after_training.json`
- `iccad2026contest/my_optimizer.py`
- `iccad2026contest/iccad2026_evaluate.py`
- `iccad2026contest/training_example.py`
- `noML_optimizer.py`

## Current Correctness Status

The saved artifact is hard-feasible:

- `iccad2026contest/eval_full_after_training.json`: 100/100 feasible.
- Read-only component recomputation over the saved positions and local validation data found:
  - overlap violations: `0`
  - soft-block area violations: `0`
  - fixed/preplaced dimension or position violations: `0`
- The evaluator command was not rerun because project rules require approval for evaluator runs.

Checkpoint provenance has an important caveat:

- `iccad2026contest/checkpoints/dit_gnn_step_43299_epoch_11_end.pth` was not found.
- `iccad2026contest/dit_gnn_step_43299_epoch_11_end.pth` was found, about 128 MB.
- `iccad2026contest/checkpoints/dit_gnn_step_141000_loss_1.05.pth` was also found, about 128 MB.
- `my_optimizer.py` loads from `Path(__file__).parent / "checkpoints"` unless `MY_OPTIMIZER_CHECKPOINT` is set, so the saved JSON does not prove which checkpoint produced it.

## Current Performance Baseline

| Metric | Value | Command | Verified? |
|---|---:|---|---|
| Score | 5.146680440 | Saved `iccad2026contest/eval_full_after_training.json`; compatible command is `python iccad2026_evaluate.py --evaluate my_optimizer.py --output eval_full_after_training.json` | Saved artifact parsed in this session; evaluator command not rerun |
| Runtime | 7.6864 s average; 20.5084 s weighted average | Same saved artifact | Saved artifact parsed in this session; runtime is locally neutralized in score |
| Memory | Unknown | Not measured | Missing |

Additional saved-artifact facts:

- Feasible cases: 100/100.
- Capped feasible cases: 0/100.
- Weighted `hpwl_gap`: `2.5651`.
- Weighted `area_gap`: `2.5026`.
- Weighted `violations_relative`: `0.1855`.
- Weighted soft counts from read-only recomputation:
  - boundary: `6.7767`
  - grouping: `3.5418`
  - MIB: `0.5367`
  - total soft: `10.8552`
  - max possible soft denominator: `58.9547`

## Expected Performance Target

User target:

- Converge to total score `<= 2.0`.

Minimum constraints for any optimization attempt:

- Preserve 100/100 local validation hard feasibility.
- Measure before/after on the same evaluator command or comparable saved artifact.
- Do not treat lower checkpoint loss as sufficient proof of score improvement.
- Prioritize large cases because the 101-120 block bucket carries `81.13%` of the current weighted score.

## Gap Analysis

Current total score is `5.1467`, so the gap to the requested target is about `3.1467` points.

This is no longer a feasibility problem, and it is not mainly a boundary-only soft-violation problem. The bottleneck is the combined HPWL and bounding-box quality gap:

- If all soft violations were set to zero while HPWL and area stayed unchanged, estimated score would still be about `3.5339`.
- If boundary violations alone were set to zero, estimated score would be about `4.0648`.
- If grouping violations alone were set to zero, estimated score would be about `4.5549`.
- If MIB violations alone were set to zero, estimated score would be about `5.0567`.
- If HPWL gap alone were set to zero, estimated score would be about `3.2719`.
- If area gap alone were set to zero, estimated score would be about `3.3304`.
- If both HPWL and area gaps were set to zero while soft violations stayed unchanged, estimated score would be about `1.4556`.

Uniform reduction estimates:

- With current soft violations, HPWL and area gaps must both shrink to about `14.75%` of their current values to reach `<= 2.0`.
- If soft violations are cut in half, HPWL and area gaps still must both shrink to about `26.02%` of current values.
- If soft violations are eliminated entirely, HPWL and area gaps still need to shrink to about `39%` of current values to be comfortably below `2.0`.

Top weighted contributors are large cases:

| Test ID | Blocks | Cost | Weight Share | HPWL Gap | Area Gap | V_rel |
|---:|---:|---:|---:|---:|---:|---:|
| 99 | 120 | 5.2716 | 0.0800 | 2.413 | 3.179 | 0.164 |
| 98 | 119 | 5.0467 | 0.0736 | 2.163 | 2.199 | 0.231 |
| 95 | 116 | 5.9394 | 0.0573 | 3.492 | 2.766 | 0.182 |
| 97 | 118 | 4.6236 | 0.0677 | 2.144 | 1.852 | 0.217 |
| 92 | 113 | 6.4332 | 0.0446 | 3.649 | 3.353 | 0.179 |

The 101-120 bucket has:

- Weighted score: `5.0993`.
- Weighted HPWL gap: `2.5246`.
- Weighted area gap: `2.5478`.
- Weighted `violations_relative`: `0.1811`.
- Weighted runtime: `22.8550 s`.

## Benchmark or Evaluator Details

The local evaluator computes feasible cost as:

```text
Cost = (1 + 0.5 * (max(0, HPWL_gap) + max(0, Area_gap))) * exp(2 * V_rel) * RuntimeAdjustment
```

Local full evaluation then neutralizes runtime with `RuntimeFactor = 1.0` before recomputing final costs, so runtime is recorded but is not the direct reason the saved score is `5.1467`.

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

- The boundary-frame attempt worked, but it did not get close to the target: score improved to about `5.15`, boundary violations dropped, and capped cases disappeared, yet weighted HPWL and area gaps remain about `2.5`.
- `ConstructiveCandidateLegalizer._score_unit_origin(...)` still sets `connection_cost = 0.0`, so origin selection does not use block-to-block or pin-to-block connectivity while placing units.
- The final `CandidateSelector` scores full candidates by HPWL, bbox area, and soft violations, but it can only choose among candidates already generated. It cannot fix connectivity-blind choices inside a candidate.
- `HardConstraintNormalizer._soft_dimensions(...)` still uses square dimensions for all non-fixed soft blocks. This leaves no aspect-ratio search to reduce bbox area or improve wirelength.
- The current `my_optimizer.py` has one structured boundary-frame candidate. The read-only `noML_optimizer.py` reference contains additional unported mechanisms: connected bottom-left placement, multiple skyline width hints, local swap/relocation/shelf-width trials, aspect trials, MIB sync, frame compaction, and rip-up/repack for large cases.
- Checkpoint provenance is ambiguous because the requested `43299` checkpoint is not in the default loader directory and the saved JSON does not record `MY_OPTIMIZER_CHECKPOINT`.

## Likely Causes

### Cause 1: Connectivity-blind construction inside candidate generation

- Type: Heuristic mismatch / algorithmic limitation.
- Evidence: Weighted `hpwl_gap` is `2.5651`. In `my_optimizer.py`, `_score_unit_origin(...)` sets `connection_cost = 0.0`, so candidate construction does not account for already placed connected blocks or pins. `noML_optimizer.py` has a reference `_unit_site_connection_cost_indexed(...)` that scores both B2B and P2B distances during placement.
- Affected modules: `ConstructiveCandidateLegalizer._score_unit_origin`, `ConstructiveCandidateLegalizer._choose_unit_origin`, `ConstructiveCandidateLegalizer._generate_unit_origins`.
- Risk: Medium. Adding connection scoring can trade HPWL against area/boundary quality and may slow large cases if implemented naively.
- Confidence: High.

### Cause 2: Shape search is too narrow for the area target

- Type: Algorithmic limitation / parameterization issue.
- Evidence: Weighted `area_gap` is `2.5026`, similar in magnitude to weighted HPWL gap. The current normalizer uses square soft-block dimensions only. The contest allows arbitrary aspect ratios for soft blocks, and `noML_optimizer.py` contains bounded aspect and MIB-sync trial machinery that is not present in `my_optimizer.py`.
- Affected modules: `HardConstraintNormalizer._soft_dimensions`, `ConstructiveCandidateLegalizer._build_units`, candidate generation, candidate selection.
- Risk: Medium. Any dimension trial must preserve soft-block area within 1%, fixed-shape dimensions, preplaced dimensions/positions, and non-overlap.
- Confidence: High.

### Cause 3: Large-case candidate-set ceiling after boundary-frame improvement

- Type: Algorithmic limitation / search-control limitation.
- Evidence: The 101-120 bucket contributes `81.13%` of score and still has score `5.0993`. Current `my_optimizer.py` has three order variants plus one boundary-frame candidate family, while the reference noML path has connected skyline placement, frame compaction, bounded local search, and rip-up/repack especially for larger cases. The current artifact has no capped cases, so the next gap is quality of the feasible candidate set rather than feasibility repair.
- Affected modules: `ConstructiveCandidateLegalizer`, `SoftConstraintImprover`, `CandidateSelector`, `MyOptimizer.solve`.
- Risk: Medium to high. Candidate expansion can raise runtime, and official hidden-test runtime normalization may penalize slow large-case search.
- Confidence: High.

## Optimization Hypotheses

| Priority | Hypothesis | Expected Impact | Risk | Affected Files | Verification |
|---:|---|---|---|---|---|
| 1 | Add bounded incremental B2B/P2B connection cost to constructive unit-origin scoring so placement sites are chosen with HPWL awareness before full candidates are finalized. | High. Directly targets weighted `hpwl_gap` `2.5651` and should help large cases where connected blocks are currently spread by area/boundary heuristics. | Medium. Must cache adjacency/pin indexes and keep runtime bounded; may worsen bbox if weighting is too strong. | `iccad2026contest/my_optimizer.py` | Validate, single-case eval, full eval, compare score, weighted HPWL, area gap, V_rel, runtime. |
| 2 | Add one bounded aspect-ratio candidate family for ordinary soft blocks and keep only selector-approved feasible candidates. | High. Directly targets weighted `area_gap` `2.5026` and can also reduce HPWL by making boundary/interior packing more compact. | Medium. Dimension changes must preserve area tolerance and cannot touch fixed/preplaced blocks; may interact with grouping/MIB macros. | `iccad2026contest/my_optimizer.py` | Validate hard feasibility; compare area gap, HPWL, MIB, total score, and runtime. |
| 3 | Port a small subset of noML-style large-case local search, starting with connected skyline or limited rip-up/repack for the 101-120 bucket only. | High but less isolated. Targets the dominant weighted bucket and can jointly improve HPWL/area/grouping after initial construction. | High. More code and runtime risk; needs strict trial budgets and rollback on full-eval slowdown or score regression. | `iccad2026contest/my_optimizer.py` | Full eval with bucket analysis; keep only if 101-120 weighted score improves and runtime remains practical. |

## Recommended First Optimization

Implement Hypothesis 1 first: add bounded incremental connection-aware unit-origin scoring.

Concrete scope:

- Keep the existing candidate families, boundary-frame candidate, checker, and selector.
- Add helper logic inside `ConstructiveCandidateLegalizer` to index B2B neighbors and P2B edges once per solve or candidate construction.
- When scoring a unit origin, compute incremental Manhattan distance from that unit's block centers to already placed connected block centers and to fixed pins.
- Use the connection cost as a bounded term in `_score_unit_origin(...)`, preferably after hard boundary feasibility preference but before weak tie-breakers such as predicted distance and coordinate bias.
- Do not add broad local search, aspect trials, MIB sync, or training changes in this first attempt.

Rollback condition:

- Revert if 100/100 feasibility is lost, total score does not improve over `5.146680440`, weighted HPWL does not improve, or average/large-case runtime grows substantially without score improvement.

This is the smallest next step that directly attacks the dominant quality bottleneck while preserving the existing legalizer and selector safety net.

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
- Stop if connection-aware scoring increases total score or weighted HPWL after full evaluation.
- Stop if runtime grows substantially on large cases without a score improvement.
- Stop before running training, installing dependencies, downloading data, or deleting/moving checkpoints without approval.
- Stop if the run intended to use `dit_gnn_step_43299_epoch_11_end.pth` cannot prove it actually loaded that checkpoint.

## Risks and Warnings

- The current baseline is a saved artifact, not an evaluator rerun from this diagnosis.
- Runtime is neutralized locally but may matter on the official leaderboard.
- The requested checkpoint path is inconsistent with the local loader directory. To avoid relative-path ambiguity, a future evaluator run that intends to use the `43299` checkpoint should set `MY_OPTIMIZER_CHECKPOINT=/home/kuotzuwei15/pda/senpai/iccad2026contest/dit_gnn_step_43299_epoch_11_end.pth`, or the file should be placed under `iccad2026contest/checkpoints/` with approval.
- Training loss omits boundary, grouping, and MIB constraints, and lower checkpoint loss alone will not solve the current HPWL/bbox candidate-quality bottleneck.
- The target `<= 2.0` likely needs multiple optimization loops: connection-aware placement plus shape/aspect or local-search improvements.

## Open Questions

- Was `MY_OPTIMIZER_CHECKPOINT` set when `iccad2026contest/eval_full_after_training.json` was generated?
- Should the `43299` checkpoint be copied or moved into `iccad2026contest/checkpoints/`, or should future runs use the environment variable explicitly?
- What score does `noML_optimizer.py` achieve under the current evaluator and checkpoint-free conditions?
- Should future training include differentiable proxies for boundary, grouping, MIB, and legalizer-aware HPWL after construction-time scoring is improved?
