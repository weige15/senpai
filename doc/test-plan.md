# Test Plan

## Purpose

This plan defines how to verify a feasibility-first redesign of `my_optimizer.py` before detailed implementation begins. Verification is done when hard-constraint behavior is covered by planned checks, evaluator commands are clearly identified as known-but-not-run, and the minimum done criteria are observable pass/fail gates instead of visual inspection or "looks correct" review.

## Source Requirements

Sources read:

- `doc/proposal.md`: objective, current failure mode, proposed anchor-aware legalizer, fallback strategy, validation plan, risks, and open questions.
- `doc/high-level-design.md`: logical modules, data flow, contracts, soft/hard constraint boundaries, and quality-gate alignment.
- `doc/problem-brief.md`: contest inputs, outputs, hard constraints, soft constraints, scoring, validation data, and environment facts.
- `doc/repo-map.md`: visible repository structure, missing tests, missing dependency manifest, main source files, and partial-workspace caveats.
- `doc/quality-gates.md`: discovered evaluator commands, missing unit/lint/type gates, and recommended minimum done criteria.

Requirements extracted:

- Return one `(x, y, width, height)` tuple per block from `MyOptimizer.solve(...)`.
- Preserve all hard constraints: no overlaps, soft-block area within 1%, fixed-shape dimensions unchanged, and preplaced position/dimensions unchanged.
- Treat boundary, grouping, and MIB constraints as soft constraints only.
- Use diffusion output only as advisory guidance for order or candidate preference.
- Insert preplaced blocks as immutable anchors before movable blocks are placed.
- Run an evaluator-aligned hard feasibility check before returning.
- Fall back to a conservative feasible packer if the optimized legalizer fails hard checks.
- Keep behavior deterministic enough for reproducible failures and bounded enough for 21-120 block validation cases.

Ambiguous or missing source material:

- `Missing`: no standalone test harness, unit test command, lint command, format command, type-check command, or CI configuration was discovered.
- `Unknown`: exact failing hard-constraint distribution in the current optimizer has not been measured.
- `Unknown`: behavior if both optimized legalization and fallback fail final hard-feasibility checks is unresolved in the HLD.
- `Unknown`: whether all original FloorSet helper modules required by `iccad2026_evaluate.py` are available in the execution environment.

## Test Scope

Covered by this plan:

- `MyOptimizer.solve(...)` input/output contract and orchestration.
- `HardConstraintNormalizer` dimension, area, fixed-shape, and preplaced metadata behavior.
- `DiffusionGuidanceAdapter` advisory-only use of model output.
- `AnchorAwareLegalizer` obstacle-aware placement around immutable anchors.
- `SoftConstraintImprover` legality-preserving treatment of boundary, grouping, and MIB moves.
- `FeasibilityChecker` evaluator-aligned rejection of hard violations.
- `FeasibleFallbackPacker` conservative recovery path.
- Public local validation/evaluation flow over `LiteTensorDataTest/`.
- Determinism and practical runtime for validation cases with 21-120 blocks.

## Non-Tested Scope

Out of scope for this phase:

- Training model checkpoints or changing `training_example.py`.
- Downloading datasets, installing dependencies, or repairing missing original FloorSet helper modules.
- Editing `LiteTensorDataTest/` or `.pth` files.
- Changing `iccad2026_evaluate.py` scoring behavior.
- Treating boundary, grouping, or MIB constraints as hard feasibility constraints.
- Final hidden-test leaderboard performance, because hidden data and official runtime normalization are not locally available.
- Final contest packaging/upload flow, because it is not specified by the visible local files.

## Smoke Tests

| Status | Check | Pass Criteria | Failure Examples |
| --- | --- | --- | --- |
| Planned | Import/load optimizer through evaluator validation path | `python iccad2026_evaluate.py --validate my_optimizer.py` exits successfully after approval | import error, missing `solve`, wrong return shape |
| Planned | Direct minimal `solve()` call through a small local harness if one is added later | returns one 4-tuple per block with finite numeric values | missing block result, non-finite coordinate, negative width |
| Planned | Single public validation case after implementation | approved single-case evaluator run reports feasible result for test case 0 | overlap, area mismatch, fixed/preplaced mismatch, cost `10.0` |
| Missing | Automated fast smoke harness committed to repo | a lightweight command exists and is documented | no command exists today |

## Unit Tests by Module

| HLD Module | Status | Planned Checks | Pass Criteria |
| --- | --- | --- | --- |
| `HardConstraintNormalizer` | Planned | Soft block dimension derivation, fixed-shape dimension preservation, preplaced coordinate/dimension metadata, malformed or missing target-position handling | soft areas are within 1%; fixed and preplaced dimensions exactly match targets; preplaced anchors are marked immutable |
| `DiffusionGuidanceAdapter` | Planned | Model output conversion to stable block order and candidate preferences; missing checkpoint behavior; no direct legality override | guidance is deterministic for fixed inputs; invalid/weak predictions do not alter legal dimensions or anchors |
| `AnchorAwareLegalizer` | Planned | Placement around one or more preplaced obstacles; touching allowed; overlap forbidden; deterministic tie-breaking | every movable block is non-overlapping with anchors and prior blocks; preplaced blocks are unchanged |
| `SoftConstraintImprover` | Planned | Boundary move accepted only when overlap-free; grouping/MIB improvement skipped or repaired when it would break hard constraints | accepted moves pass `FeasibilityChecker`; rejected moves leave the previous feasible placement unchanged |
| `FeasibilityChecker` | Planned | Overlap detection, soft-block area tolerance, fixed/preplaced dimension checks, preplaced coordinate checks, malformed tuple checks | violations are reported by category and hard-feasible placements are accepted |
| `FeasibleFallbackPacker` | Planned | Fallback placement with anchors, deterministic strip/row behavior, final check integration | fallback returns a hard-feasible placement or reports unrecoverable failure without silently returning invalid output |
| `MyOptimizer.solve(...)` orchestration | Planned | optimized-path success, optimized-path failure triggers fallback, fallback result is final-checked, output ordering by block index | returned list length equals `block_count`; all tuples are finite and pass hard checks |

## Integration Tests

| Status | Scenario | Pass Criteria |
| --- | --- | --- |
| Planned | End-to-end optimized path on a small synthetic case without preplaced blocks | `solve()` returns all blocks, dimensions satisfy hard constraints, no overlaps |
| Planned | End-to-end preplaced-anchor case | preplaced coordinates and dimensions are exact; all movable blocks avoid the anchor |
| Planned | Boundary-constrained case where boundary movement would overlap another block | boundary move is skipped or repaired; final placement stays hard-feasible |
| Planned | Forced optimized-path failure with fallback enabled | fallback is invoked and final output passes `FeasibilityChecker` |
| Planned | Approved evaluator validation on `LiteTensorDataTest/` test case 0 | evaluator reports feasible result and no hard violation categories |
| Planned | Approved full local validation over all 100 public cases | all cases are feasible before score optimization is considered |

## Golden Test Cases

These are hand-solvable fixtures to add in a future lightweight harness. Exact tuple outputs are required only where the module contract is deterministic; otherwise the oracle checks hard feasibility and immutable fields.

| Status | Case | Input Sketch | Expected Result |
| --- | --- | --- | --- |
| Planned | Single soft block | `block_count=1`, `area_targets=[100]`, no hard fixed/preplaced constraints | one tuple; `width * height` in `[99, 101]`; no overlap possible |
| Planned | Two touching soft blocks | two soft blocks with areas `4` and `9`, no anchors | two tuples; areas within tolerance; rectangles may touch but must not overlap |
| Planned | One preplaced anchor plus one movable block | anchor at `(0, 0, 2, 2)`, movable area `4` | anchor tuple exactly `(0, 0, 2, 2)`; movable rectangle does not intersect `[0,2] x [0,2]` |
| Planned | Fixed-shape block | one fixed-shape block with target dimensions `3 x 5` | returned dimensions exactly `3` and `5`; position may vary unless preplaced |
| Planned | Boundary move conflict | two blocks where moving one to the requested boundary would overlap the other | hard-feasible original placement is retained or repaired; no overlap is introduced |
| Planned | Fallback trigger | inject or simulate an optimized candidate with overlap | `FeasibilityChecker` rejects optimized candidate; fallback output is checked before return |

## Oracle or Reference Implementation Strategy

- `Existing`: `iccad2026_evaluate.py` contains the authoritative local evaluator semantics for hard constraints and scoring.
- `Planned`: use an internal `FeasibilityChecker` aligned with evaluator hard checks as the acceptance oracle for unit and integration tests.
- `Planned`: use brute-force rectangle intersection checks for small synthetic cases; touching edges must be accepted and positive-area intersections must be rejected.
- `Planned`: for small cases with no anchors, compare candidate output against a simple deterministic strip-packing reference for feasibility, not quality.
- `Planned`: for cases with anchors, use obstacle-set checks as the oracle rather than expecting unique coordinates.
- `Missing`: no committed reference implementation or local unit-test harness exists yet.

## Randomized or Property Tests

| Status | Property | Input Range | Determinism and Oracle |
| --- | --- | --- | --- |
| Planned | No overlaps for random soft-only cases | 1-12 blocks, positive areas in a bounded range | fixed seeds; oracle is pairwise rectangle intersection |
| Planned | Anchor immutability | 1-4 random preplaced anchors plus 1-12 movable blocks | fixed seeds; preplaced tuples must exactly match input targets |
| Planned | Area preservation | random soft-block area targets across several magnitudes | fixed seeds; relative error must be `<= 1%` |
| Planned | Fixed-shape preservation | random fixed width/height pairs | fixed seeds; dimensions must exactly match target dimensions |
| Planned | Fallback safety | random overlapped optimized candidates passed to fallback path | fallback result must pass hard checks or report explicit unrecoverable failure |

Shrinking expectation: when a randomized failure occurs, reduce to the smallest block count that reproduces the same hard-violation category, then save it as a regression fixture.

## Edge Cases

| Status | Edge Case | Pass Criteria |
| --- | --- | --- |
| Planned | Blocks that exactly touch edges | not counted as overlap |
| Planned | Very small positive area targets | positive finite dimensions; area within tolerance |
| Planned | Large area targets mixed with small targets | no numeric overflow or non-finite coordinates |
| Planned | Multiple preplaced anchors with gaps | movable blocks occupy legal free space or fallback expands layout |
| Planned | Preplaced anchors near or beyond current packed bounding box | anchor coordinates are preserved and bbox expands as needed |
| Planned | Boundary bitmask combinations for edge/corner requirements | treated as soft; hard feasibility is preserved |
| Planned | Missing or unusable checkpoints | optimizer still returns hard-feasible placement using initialized guidance or fallback |
| Planned | Malformed output tuples from internal candidate stages | checker rejects before return |
| Unknown | Contradictory hard inputs such as overlapping immutable preplaced blocks | expected behavior is unresolved and should be decided before implementation |

## Performance Benchmarks

| Status | Benchmark | Workload | Metrics | Threshold |
| --- | --- | --- | --- | --- |
| Planned | Single-case local evaluator | `python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0` after approval | feasible flag, cost, runtime | feasible result required; runtime recorded for comparison |
| Planned | Full public validation evaluator | `python iccad2026_evaluate.py --evaluate my_optimizer.py` after approval | feasible count, total score, average cost, average runtime | 100/100 feasible before score tuning |
| Planned | Synthetic scalability check | random or fixture cases up to 120 blocks if a harness is added | wall time and candidate count | practical runtime with no unbounded candidate explosion |
| Unknown | Official hidden-test runtime normalization | hidden 100-case set | official normalized runtime factor | not locally measurable |

Benchmark-only checks must not replace correctness checks. A faster run that returns any hard violation fails this plan.

## Evaluator or Grading Commands

No command in this section was run while creating this test plan.

| Status | Command | Purpose | Notes |
| --- | --- | --- | --- |
| Known, not run | `python iccad2026_evaluate.py --validate my_optimizer.py` | optimizer API and format validation | requires approval; may fail if original helper modules are missing |
| Known, not run | `python iccad2026_evaluate.py --validate my_optimizer.py --quick` | quick validation variant | requires approval; discovered in `doc/quality-gates.md` |
| Known, not run | `python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0` | single public validation case | requires approval; writes default results JSON |
| Known, not run | `python iccad2026_evaluate.py --evaluate my_optimizer.py` | all 100 local public validation cases | requires approval; may run for a while and writes default results JSON |
| Known, not run | `python iccad2026_evaluate.py --evaluate my_optimizer.py --save-solutions` | save evaluated placements for comparison | requires approval; writes solutions/results JSON |
| Known, not run | `python iccad2026_evaluate.py --score my_optimizer_solutions.json` | re-score saved solutions | requires approval and existing solutions JSON |
| Known, not run | `python iccad2026_evaluate.py --baseline` | generate baseline metrics | requires approval; writes `baseline_metrics.json` |
| Known, not run | `python iccad2026_evaluate.py --info` | lightweight evaluator information smoke test | requires approval under current workspace rules |
| Known, not run | `python iccad2026_evaluate.py --visualize --test-id 0` | visual inspection artifact | requires approval; writes `validation_case_0.png` |
| Known, not run | `python training_example.py` | training pipeline | not a minimum done gate; long-running and writes checkpoints |
| Missing | unit-test command | module-level tests | no test framework or command discovered |
| Missing | lint/format/type-check commands | static quality gates | no configuration discovered |

## Regression Tests

| Status | Regression Target | Trigger to Add Fixture | Expected Stable Check |
| --- | --- | --- | --- |
| Planned | Preplaced overwrite overlap | evaluator or synthetic case shows a movable block overlapped after preplaced coordinates are restored | anchor-aware legalizer avoids the anchor before final return |
| Planned | Boundary post-move overlap | boundary adjustment creates overlap after legal packing | soft move is rejected or repaired |
| Planned | Fixed/preplaced dimension drift | dimensions differ from target after normalization or fallback | checker rejects and implementation preserves exact dimensions |
| Planned | Soft area tolerance drift | realized soft area exceeds 1% relative error | checker rejects before return |
| Planned | Missing checkpoint behavior | no `.pth` file exists under `checkpoints/` | optimizer remains hard-feasible without trained weights |
| Planned | Large-case candidate explosion | runtime grows unexpectedly for 116-120 block cases | candidate generation is bounded or pruned |

## Manual Verification

- Review `doc/test-plan.md` against `doc/proposal.md` and `doc/high-level-design.md` for requirement-to-module traceability.
- Inspect any generated failing fixtures to confirm they encode the intended hard-constraint category.
- After approved evaluator runs, review feasible count, hard-violation messages, total score, average cost, and average runtime.
- Use visualization only as a secondary diagnostic for selected cases; visual inspection is not a pass criterion.
- Confirm no contest data, evaluator scoring logic, or training checkpoints were modified by verification work unless explicitly approved.

## Minimum Done Criteria

Before detailed implementation is considered ready to proceed:

- `doc/test-plan.md` exists with all required sections from the `test-plan-generator` skill.
- Every HLD module has at least one planned verification method or an explicit missing/unknown entry.
- Hard constraints have pass/fail checks: overlap, area tolerance, fixed-shape dimensions, and preplaced position/dimensions.
- Soft constraints are tested only as legality-preserving improvements, not as hard feasibility requirements.
- Evaluator and grading commands are listed as `Known, not run` unless they are later approved and actually executed.
- Missing unit, lint, format, type-check, and CI gates are recorded as `Missing`.

Before optimizer implementation is considered ready for contest-style local validation:

- Approved `python iccad2026_evaluate.py --validate my_optimizer.py` passes.
- Approved `python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0` reports a feasible result.
- Approved `python iccad2026_evaluate.py --evaluate my_optimizer.py` reports 100/100 feasible local validation cases.
- Any remaining infeasible case has a saved regression fixture or documented blocker with the exact hard-violation category.

## Open Questions

- What exact hard violation is most common in the current optimizer: overlap, area tolerance, fixed/preplaced dimension mismatch, or preplaced position mismatch?
- Should `FeasibilityChecker` expose violation categories through logs, internal return values, or both?
- What should `MyOptimizer.solve(...)` return or raise if both optimized legalization and fallback cannot satisfy hard constraints?
- Is adding a small internal test harness acceptable, given no test framework exists today?
- Are original FloorSet helper modules available in the intended evaluation environment despite being absent from this partial workspace?
- Are trained checkpoints expected during final evaluation, or must final quality assume no checkpoints?
- What exact `data_lite` loader path should be used for future training experiments, and should it remain separate from validation?
