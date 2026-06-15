# Performance Diagnosis

## Purpose

Diagnose the next optimization step for `iccad2026contest/my_optimizer.py`, with the user goal of keeping the GNN + diffusion model for coarse coordinate guidance while integrating the stronger legalization strategy from `noML_optimizer.py`.

The key question is no longer whether `my_optimizer.py` can be made hard-feasible. Saved results show hard feasibility is stable. The diagnosis focuses on why score remains near the feasible cap and which `noML_optimizer.py` legalization idea should be integrated first.

No optimizer implementation code was changed during this diagnosis.

## Diagnosis Scope

Covered:

- `iccad2026contest/my_optimizer.py` normalization, guidance, anchor-aware legalizer, fallback packer, soft improver, and `solve()` orchestration.
- `noML_optimizer.py` parsing, dimension planning, placement-unit abstraction, constructive candidates, candidate scoring, boundary-frame/skyline legalization, and local-search flow.
- `iccad2026contest/iccad2026_evaluate.py` hard feasibility checks and cost formula.
- Saved validation artifacts in `iccad2026contest/eval_full_after_training.json`, `iccad2026contest/my_optimizer_results.json`, `/tmp/senpai_pre_boundary.json`, and `/tmp/senpai_post_boundary.json`.

Not covered:

- No evaluator rerun was performed.
- No training or checkpoint generation was performed.
- No hidden-test behavior or official runtime normalization was measured.
- No full `noML_optimizer.py` benchmark artifact exists in this workspace.

## Source Documents Read

- `AGENTS.md`
- `README.md`
- `doc/problem-brief.md`
- `doc/proposal.md`
- `doc/high-level-design.md`
- `doc/detailed-design.md`
- `doc/test-plan.md`
- `doc/quality-gates.md`
- `doc/tasks/progress.md`
- `doc/performance-log.md`
- `iccad2026contest/iccad2026_evaluate.py`
- `iccad2026contest/my_optimizer.py`
- `noML_optimizer.py`
- `iccad2026contest/training_example.py`
- `iccad2026contest/eval_full_after_training.json`
- `iccad2026contest/eval_case0_after_training.json`
- `iccad2026contest/my_optimizer_results.json`
- `/tmp/senpai_pre_boundary.json`
- `/tmp/senpai_post_boundary.json`
- `/tmp/senpai_post_boundary_case0.json`

## Current Correctness Status

Saved evaluator results report hard feasibility as stable:

- `iccad2026contest/eval_full_after_training.json`: 100/100 feasible, total score `9.7460`, average runtime `1.13s`.
- `/tmp/senpai_post_boundary.json`: 100/100 feasible, total score `9.7367`, average runtime `0.68s`.
- `iccad2026contest/my_optimizer_results.json`: 100/100 feasible, total score `9.9565`, average runtime `0.45s`.

The current workspace has no `iccad2026contest/checkpoints/` directory, so a fresh local run would use deterministic no-checkpoint guidance unless checkpoints are restored. Evaluator commands were not run because repository rules require approval and evaluation writes output JSON by default.

Correctness is stable enough to proceed with optimization, provided the next implementation keeps the existing `FeasibilityChecker` acceptance gate and fallback path.

## Current Performance Baseline

| Metric | Value | Command | Verified? |
|---|---:|---|---|
| Score | 9.7367167134 | `python iccad2026_evaluate.py --evaluate my_optimizer.py --output /tmp/senpai_post_boundary.json` | Stale saved artifact parsed in this session; command not rerun |
| Runtime | 0.6807339573 s average | Same command | Stale saved artifact parsed in this session; command not rerun |
| Memory | Unknown | Not measured | Missing |

Additional baselines:

- Best saved trained-checkpoint run in the repo: score `9.7459558487`, 100/100 feasible, average runtime `1.1306540513s`, 72/100 cases capped at `9.999999`.
- Saved no-checkpoint pre-boundary run: score `9.9564528621`, 100/100 feasible, average runtime about `0.47s`, 79/100 cases capped.
- Saved no-checkpoint post-boundary run: score `9.7367167134`, 100/100 feasible, average runtime about `0.68s`, 64/100 cases capped.

For `/tmp/senpai_post_boundary.json`, weighted metrics are:

- weighted `hpwl_gap`: `3.1245`;
- weighted `area_gap`: `0.2027`;
- weighted `violations_relative`: `0.7853`;
- capped-case weighted share: `65.70%`.

## Expected Performance Target

No official numeric target is available in the local docs. The practical near-term target is:

- preserve 100/100 local validation feasibility;
- reduce score below the current best saved `9.7367`;
- reduce capped-case count and capped weighted share;
- reduce weighted `violations_relative`, especially boundary and grouping violations;
- avoid a large runtime increase on 101-120 block cases because those dominate the exponential total score.

## Gap Analysis

The score is still near the feasible cap because soft violations and HPWL remain high. Area is not the first-order bottleneck.

Counterfactuals from the saved post-boundary run:

| Counterfactual | Weighted Total Score |
|---|---:|
| Original post-boundary run | 9.7367 |
| Set `violations_relative = 0` only | 2.6636 |
| Set `hpwl_gap = 0` only | 5.3235 |
| Set `area_gap = 0` only | 9.5597 |
| Halve `violations_relative` | 5.7995 |
| Halve `hpwl_gap` | 8.6049 |

Read-only reconstruction of soft subcomponents was computed from saved positions plus `LiteTensorDataTest/` constraints using `noML_optimizer.py`'s `_soft_violations(...)` helper. This was not an evaluator rerun, but it closely matches the saved aggregate `violations_relative`.

For `/tmp/senpai_post_boundary.json`, weighted soft counts are approximately:

| Component | Weighted Count | Share of Weighted Soft Violations |
|---|---:|---:|
| Boundary | 24.56 | 53.17% |
| Grouping | 21.09 | 45.66% |
| MIB | 0.54 | 1.16% |

Component counterfactuals for the post-boundary run:

| Counterfactual | Weighted Total Score |
|---|---:|
| Remove boundary violations only | 5.5932 |
| Remove grouping violations only | 6.1813 |
| Remove MIB violations only | 9.6887 |
| Remove boundary and grouping violations | 2.7160 |

This makes the next target clear: the legalizer must address boundary and grouping together. MIB-only work is not a good first optimization because it explains about 1% of weighted soft violations.

## Benchmark or Evaluator Details

`iccad2026contest/iccad2026_evaluate.py` computes feasible cost as:

```text
Cost = (1 + 0.5 * (max(0, HPWL_gap) + max(0, Area_gap))) * exp(2 * V_rel) * RuntimeFactor
```

The local evaluator sets `RuntimeFactor = 1.0`, then caps feasible costs at `9.999999`. Hidden leaderboard runtime normalization may differ.

Hard infeasibility still costs `10.0`, so any integration from `noML_optimizer.py` must be gated by `FeasibilityChecker` and fallback routing. The local total score is exponentially weighted by `exp(block_count / 12)`, so larger cases matter most.

## Observed Bottlenecks

- Soft violations remain high after boundary-only improvement: weighted `violations_relative` is about `0.7853`.
- Boundary violations are still the largest reconstructed soft component, but grouping is nearly as large.
- HPWL remains high: weighted `hpwl_gap` is about `3.1245` in the post-boundary run.
- Current `AnchorAwareLegalizer._score_candidate(...)` scores area growth, distance to diffusion prediction, span, coordinate bias, and tie-break coordinates, but does not use connectivity, pins, grouping, MIB, or boundary metadata.
- Current `SoftConstraintImprover` can move individual boundary blocks, but it cannot build boundary-frame layouts or group macros.
- Current `HardConstraintNormalizer` uses square soft dimensions and does not synchronize MIB-compatible shapes, while `noML_optimizer.py` has `_plan_dimensions(...)` with MIB-compatible shape planning.
- Current `DiffusionGuidanceAdapter` uses model output only to sort blocks and seed candidate positions. With no local checkpoint directory, it falls back to block-index order.
- `noML_optimizer.py` has a richer legal candidate architecture: `PlacementUnit`, cluster macros, boundary-frame/skyline packers, connected bottom-left placement, candidate preflight, proxy scoring, and bounded local search.

## Likely Causes

### Cause 1: Current legalizer places individual blocks instead of soft-constraint-aware units

- Type: Algorithmic limitation.
- Evidence: `iccad2026contest/my_optimizer.py` places each movable block independently in `AnchorAwareLegalizer.legalize(...)`. `noML_optimizer.py` groups movable cluster members into `PlacementUnit` cluster macros via `_plan_soft_units(...)` and `_make_cluster_macro(...)`, making grouping abutment legal by construction for those units. Reconstructed post-boundary soft violations are about 45.66% grouping.
- Affected modules: `HardConstraintNormalizer`, `AnchorAwareLegalizer`, `SoftConstraintImprover`, new adapter code inside `iccad2026contest/my_optimizer.py`.
- Risk: Medium. Cluster macros can increase area or HPWL if used blindly, so they should be an alternative candidate, not a replacement path.
- Confidence: High.

### Cause 2: Candidate scoring is not aligned with evaluator quality terms

- Type: Heuristic mismatch.
- Evidence: `AnchorAwareLegalizer._score_candidate(...)` ignores `b2b_connectivity`, `p2b_connectivity`, `pins_pos`, and aggregate soft violations. `noML_optimizer.py` scores candidates with HPWL, bounding-box area, soft relative violations, and a hard-feasibility barrier in `_score_candidate(...)`, with `CandidateManager` selecting the best feasible candidate.
- Affected modules: `AnchorAwareLegalizer`, `SoftConstraintImprover`, `FeasibilityChecker`, possible candidate manager inside `my_optimizer.py`.
- Risk: Medium. More complete scoring can improve quality but may increase runtime if every candidate performs full HPWL and soft checks.
- Confidence: High.

### Cause 3: Diffusion guidance is too weakly coupled to legalization quality

- Type: Algorithmic limitation / environment issue.
- Evidence: `DiffusionGuidanceAdapter` currently turns model predictions into a placement order and predicted x/y preferences only. The local workspace has no checkpoint directory, so fresh runs degrade to deterministic fallback guidance. The saved trained run still has weighted `hpwl_gap` about `3.0854` and weighted `violations_relative` about `0.8135`, so model output alone is not enough without a stronger legalizer.
- Affected modules: `DiffusionGuidanceAdapter`, `AnchorAwareLegalizer`, `MyOptimizer.solve`.
- Risk: Low to medium. The model can remain advisory while legal candidates are generated by deterministic code.
- Confidence: Medium.

## Optimization Hypotheses

| Priority | Hypothesis | Expected Impact | Risk | Affected Files | Verification |
|---:|---|---|---|---|---|
| 1 | Add a noML-style constructive legalizer candidate inside `my_optimizer.py`: build placement units with movable cluster macros, preserve preplaced/fixed anchors, use diffusion order as one seed, generate a boundary-skyline-connected candidate, and choose the best hard-feasible result against the current ML-guided candidate using evaluator-style proxy scoring. | High. Targets boundary and grouping together while preserving hard feasibility; can also reduce HPWL through connected bottom-left placement. | Medium. Several helpers must be ported/adapted carefully, and runtime must stay bounded. | `iccad2026contest/my_optimizer.py` | Validate import/API, single-case evaluation, full evaluation; compare score, capped cases, weighted soft proxy, HPWL, runtime |
| 2 | Add incremental HPWL-aware scoring directly to `AnchorAwareLegalizer._score_candidate(...)`, using already placed connected blocks and pin edges after hard-feasible filtering. | Medium to high for HPWL; less direct for soft violations. | Medium. Candidate scoring becomes more expensive and may trade off against area/soft constraints. | `iccad2026contest/my_optimizer.py` | Same evaluator commands; compare weighted `hpwl_gap`, score, and runtime |
| 3 | Port MIB-compatible dimension planning from `noML_optimizer.py` into `HardConstraintNormalizer`, synchronizing soft MIB groups only when all members can share an area-valid shape. | Low to medium. MIB is only about 1% of reconstructed weighted soft violations, but this supports later unit-based packing. | Low to medium. Must not alter fixed/preplaced dimensions or violate soft-block area tolerance. | `iccad2026contest/my_optimizer.py` | Synthetic MIB cases plus evaluator; compare MIB violations and feasibility |

## Recommended First Optimization

Implement only Hypothesis 1 first: a noML-style constructive legalizer candidate as an alternative candidate path inside `iccad2026contest/my_optimizer.py`.

The first implementation should be deliberately bounded:

- Do not import `noML_optimizer.py` at runtime for contest submission; port or adapt only the needed helper logic into `my_optimizer.py`.
- Keep the existing ML-guided `AnchorAwareLegalizer` output.
- Build one additional candidate that borrows `noML_optimizer.py` concepts:
  - placement units;
  - cluster macros for movable grouping members;
  - exact immutable anchors;
  - optional MIB-compatible dimensions only when area-valid;
  - boundary-skyline-connected constructive packing;
  - evaluator-style hard preflight and proxy scoring.
- Run the existing `SoftConstraintImprover` and `FeasibilityChecker` on both the current ML candidate and the noML-style candidate.
- Return the best hard-feasible candidate by proxy score, falling back to `FeasibleFallbackPacker` when neither candidate passes.

Rollback condition:

- Revert the change if full evaluation feasibility drops below 100/100, total score does not improve below `9.7367`, capped weighted share does not improve, or average runtime grows substantially without score gain.

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

- Stop if a local or approved evaluator run reports any hard infeasible case.
- Stop if the noML-style candidate path replaces ML guidance instead of adding a candidate alternative.
- Stop if runtime increases sharply on 101-120 block cases before score improves.
- Stop if adapting the helper code requires broad rewrites outside `iccad2026contest/my_optimizer.py`.
- Stop before installing dependencies, running training, generating checkpoints, or downloading data without approval.

## Risks and Warnings

- `/tmp/senpai_post_boundary.json` is a saved artifact, not a rerun from this turn.
- Current local runs would not use trained diffusion weights because `iccad2026contest/checkpoints/` is absent.
- The reconstructed boundary/grouping/MIB split uses `noML_optimizer.py` helper logic and saved positions, not the official evaluator's Shapely path. The aggregate values are close enough for diagnosis, but exact counts should be confirmed in an approved evaluator/instrumented run if needed.
- A full port of `noML_optimizer.py` would be too risky as a first step; it would bypass the GNN + diffusion vision rather than integrating with it.
- The current docs disagree about `iccad2026contest/requirements.txt`: `AGENTS.md` says it was absent, but it is present in this workspace now. No install command was run.

## Open Questions

- Should the noML-style candidate be enabled only when no checkpoint is available, or always compete against the diffusion-guided candidate?
- Should implementation copy selected helpers into `my_optimizer.py` for submission self-containment, or is depending on `noML_optimizer.py` acceptable in the user's contest packaging?
- What score does `noML_optimizer.py` achieve on the same validation set when run directly?
- Will the trained checkpoint be restored before the next evaluation, and should the benchmark compare both no-checkpoint and checkpoint modes?
- Is it acceptable to add a small synthetic test harness for placement units and cluster macros, given no unit-test framework exists?
