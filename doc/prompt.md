# Implementation Prompt

## Objective

Implement the hard-constraint-safe legalization redesign for the ICCAD 2026 FloorSet Challenge optimizer in `/home/kuotzuwei15/pda/senpai`.

The implementation target is `my_optimizer.py`. Preserve the `FloorplanOptimizer.solve(...)` API and return one `(x, y, width, height)` tuple per block. The first success criterion is hard feasibility: no overlaps, soft-block area within 1%, fixed-shape dimensions unchanged, and preplaced position/dimensions unchanged. Score improvement comes after feasibility.

Confirmed failure mode: the current solver runs the diffusion model, builds a B*-tree order from the predicted coordinates, contour-packs the blocks, and then restores preplaced coordinates plus applies boundary moves after packing. Those post-pack edits can invalidate the packer's non-overlap guarantee and produce hard-infeasible layouts. The implementation must replace that final acceptance path with anchor-first legalization, evaluator-aligned checking, and fallback.

The planned architecture is a private in-file pipeline:

```text
MyOptimizer.solve(...)
  -> HardConstraintNormalizer
  -> DiffusionGuidanceAdapter
  -> AnchorAwareLegalizer
  -> SoftConstraintImprover
  -> FeasibilityChecker
      -> return optimized placement if feasible
      -> FeasibleFallbackPacker
      -> final FeasibilityChecker
      -> return fallback placement if feasible
```

Do not change contest data or evaluator scoring. Use diffusion output as advisory guidance only; hard feasibility must be owned by deterministic legalization, checking, and fallback.

## Inputs to Read First

Read these files before coding, in this order:

- `AGENTS.md`: local workflow rules, approval requirements, and repository constraints.
- `README.md`: contest API, data terminology, commands, scoring, and setup caveats.
- `doc/problem-brief.md`: present; source-grounded contest objective, inputs, outputs, hard constraints, scoring, data roles, and environment facts.
- `doc/repo-map.md`: present; current repository structure, entry points, missing dependencies/configs, and implementation status.
- `doc/quality-gates.md`: present; known commands, missing gates, and commands that require approval.
- `doc/proposal.md`: implementation objective, failure mode, feasibility-first approach, validation plan, and data-role clarification.
- `doc/high-level-design.md`: six-module architecture, data flow, contracts, and operational constraints.
- `doc/test-plan.md`: module-specific verification plan, golden cases, evaluator gates, and missing test infrastructure.
- `doc/detailed-design.md`: authoritative module boundaries, private data contracts, algorithms, tolerances, and open questions.
- `doc/tasks/progress.md`: progress tracker to update during implementation.
- `doc/tasks/hard-constraint-normalizer.md`
- `doc/tasks/diffusion-guidance-adapter.md`
- `doc/tasks/anchor-aware-legalizer.md`
- `doc/tasks/soft-constraint-improver.md`
- `doc/tasks/feasibility-checker.md`
- `doc/tasks/feasible-fallback-packer.md`

Also inspect these implementation files before editing:

- `my_optimizer.py`
- `iccad2026_evaluate.py`
- `training_example.py`

Optional source context is present as `FloorplanningContest_ICCAD_2026_v10.pdf`, but the Markdown docs above already summarize the relevant contest requirements.

## Current Implementation

Repository root: `/home/kuotzuwei15/pda/senpai`.

Visible top-level files include:

- `my_optimizer.py`: current optimizer implementation and primary edit target.
- `iccad2026_evaluate.py`: contest evaluator, optimizer base class, scoring helpers, dataloaders, CLI, and validation/evaluation logic.
- `training_example.py`: separate long-running training pipeline that writes checkpoints.
- `LiteTensorDataTest/`: public local validation/evaluation data with `config_21` through `config_120`.
- `FloorplanningContest_ICCAD_2026_v10.pdf`: contest specification.
- `doc/`: planning documents and module task files.

Dataset-role facts to preserve:

- Lab-server `data_lite/worker_*` is the user-reported training corpus, with entries such as `worker_0`, `worker_2`, and `worker_30...`, totaling roughly 1M samples. It was not verified in this local workspace.
- There is no separate ML validation split inside `data_lite` by default; the loader treats it as one large training dataset.
- `LiteTensorDataTest/` is the public local validation/evaluation set with 100 cases. The contest README calls it `Validation`, while the folder and evaluator class names use `Test`; do not let that naming mismatch change the data role.

Current `my_optimizer.py` facts:

- Imports `FloorplanOptimizer`, `calculate_hpwl_b2b`, `calculate_hpwl_p2b`, `calculate_bbox_area`, and `check_overlap` from `iccad2026_evaluate.py`.
- Defines `NetlistGNN`, `TimestepEmbedder`, `DiTBlock`, and `DiTSmallFloorplanBackbone`.
- Defines `BStarTreeLegalizer` with `build_from_ai_coordinates(...)` and `pack(...)`.
- Defines `MyOptimizer(FloorplanOptimizer)` with `solve(...)`.
- In `MyOptimizer.__init__`, loads the latest `.pth` from `checkpoints/` when present; otherwise runs with initialized model weights.
- In `solve(...)`, derives widths/heights first: fixed/preplaced dimensions come from `target_positions[i, 2:4]`; soft blocks use `sqrt(area)`.
- Runs the Edge-GNN plus DiT model from an all-zero initial placement and builds `ai_layout`.
- Uses `BStarTreeLegalizer` to build an order from AI coordinates and contour-pack blocks.
- After packing, directly overwrites preplaced coordinates and moves boundary-constrained blocks to bounding-box edges. This post-diffusion/post-pack correction can violate hard constraints and is the core failure mode to remove or bypass before returning.

Relevant evaluator facts from `iccad2026_evaluate.py`:

- `FloorplanOptimizer.solve(...)` signature includes `target_positions: Optional[torch.Tensor] = None`.
- The evaluator builds `target_positions` for the optimizer: all `-1` by default, fixed-shape blocks get width/height, and preplaced blocks get x/y/width/height.
- `check_overlap(...)` counts a violation only when both axis overlaps are greater than `1e-6`; touching edges are allowed.
- `check_area_tolerance(...)` uses 1% relative tolerance for soft blocks and skips fixed/preplaced blocks.
- `check_dimension_hard_constraints(...)` uses `1e-4` tolerance for fixed/preplaced dimensions and preplaced x/y.
- `evaluate_solution(...)` marks a case feasible only when overlap, area, and dimension hard-constraint violation counts are all zero.
- CLI commands include `--validate`, `--evaluate`, `--score`, `--baseline`, `--visualize`, `--training`, `--info`, `--data-path`, `--output`, `--test-id`, `--quick`, and `--save-solutions`.

Current quality infrastructure:

- No build command was discovered.
- No standalone unit-test command or test directory was discovered.
- No lint, format, type-check, static-analysis, or CI configuration was discovered.
- `README.md` references `iccad2026contest/requirements.txt`, but that file is not present in this workspace.
- `iccad2026_evaluate.py` imports helper modules from the original FloorSet layout that may be missing in this partial workspace.
- `training_example.py` imports `iccad2026contest.iccad2026_evaluate`, uses `get_training_dataloader`, auto-resumes from `checkpoints/`, and writes `.pth` checkpoints. Do not treat training as a correctness gate.

Git state at prompt creation:

- A usable Git repository was detected on branch `main`.
- Several planning docs and `doc/tasks/` appeared untracked. Re-check `git status --short --branch` before editing because this may have changed.

## Hard Constraints

Preserve these behavior and workflow constraints:

- Keep the external optimizer API: `MyOptimizer.solve(block_count, area_targets, b2b_connectivity, p2b_connectivity, pins_pos, constraints, target_positions=None)`.
- Return a list of exactly `block_count` `(x, y, width, height)` tuples.
- All returned tuple values must be finite, and width/height must be positive.
- No two blocks may overlap. Touching edges are allowed.
- Soft-block realized area must be within 1% relative error of `area_targets`.
- Fixed-shape blocks must preserve exact input width and height from `target_positions`, checked with evaluator tolerance `1e-4`.
- Preplaced blocks must preserve exact input x, y, width, and height from `target_positions`, checked with evaluator tolerance `1e-4`.
- Boundary, grouping, and MIB constraints are soft constraints only; never enforce them in a way that breaks hard feasibility.
- Diffusion output may influence block order and candidate preference only. It must never change legal dimensions, move anchors, or make an infeasible placement acceptable.
- Preplaced blocks are immutable anchors from the start of placement.
- Final output must be block-index ordered, not placement-order ordered.
- Use evaluator-aligned overlap threshold `1e-6`, fixed/preplaced tolerance `1e-4`, and soft area tolerance `0.01`.
- Keep legalizer and fallback deterministic for reproducible failures.
- Keep runtime practical for 21-120 block validation cases; avoid unbounded candidate growth.
- `LiteTensorDataTest/` is local public validation/evaluation data with 100 cases. Treat it as read-only. Although the README calls this set `Validation`, the folder/code name uses `Test`.
- Lab-server `data_lite/worker_*` is training data by user context, about 1M samples, and was not locally verified. The default loader treats it as one large training dataset, with no separate ML validation split by default. Do not treat it as validation unless a separate split policy is explicitly designed later.
- Do not edit `iccad2026_evaluate.py`, `LiteTensorDataTest/`, training data, labels, `.pth` datasets, or checkpoint files unless the user explicitly asks.
- Avoid new dependencies; no dependency manifest is present locally.
- Ask for approval before installs, downloads, evaluator runs, training, long-running commands, output-producing commands, or creating checkpoint/result artifacts when required by `AGENTS.md`.

## Non-Goals

- Do not optimize hidden-test packaging or official submission upload.
- Do not redesign the training pipeline.
- Do not require trained checkpoints for basic hard-feasible behavior.
- Do not use `training_example.py` as a minimum correctness gate.
- Do not split or repurpose `data_lite` as validation data.
- Do not judge the legalizer fix by carving an ad hoc validation split from `data_lite`; use synthetic checks and approved `LiteTensorDataTest/` evaluator runs for hard-feasibility validation.
- Do not make boundary, grouping, or MIB hard constraints.
- Do not broadly refactor model architecture unless it is strictly necessary to preserve existing inference behavior.
- Do not add a new package layout, service boundary, queue, storage layer, or external API.
- Do not change contest scoring logic or evaluator semantics.

## Execution Model

Work autonomously but incrementally:

1. Read all inputs listed above before editing.
2. Re-check repository status with `git status --short --branch`; preserve unrelated user or agent changes.
3. Keep edits focused on `my_optimizer.py` and `doc/tasks/progress.md` unless a small test harness or docs update is explicitly justified and allowed.
4. Implement one module or small workstream at a time.
5. Update `doc/tasks/progress.md` after starting, completing, blocking, or verifying each module.
6. Prefer small, reviewable patches. Do not perform broad refactors unrelated to feasibility.
7. Use private dataclasses, helper functions, or classes inside `my_optimizer.py` as described in `doc/detailed-design.md`.
8. Keep write scopes disjoint when using subagents. If several agents need `my_optimizer.py`, have only one writer at a time and let the main agent merge.
9. Run module-specific checks after each module where possible. If no configured command exists, use narrowly scoped synthetic checks or direct Python snippets only when safe and approved under local rules.
10. Run full quality gates at the end, requesting approval for evaluator commands as required.
11. Summarize command outputs in progress notes and final response. Include feasible count, score, hard-violation categories, and runtime when evaluator output is available.
12. Stop and ask only when blocked by missing requirements, conflicting docs, destructive choices, credentials, external services, unavailable dependencies, or quality gates that cannot be run.

If a command fails due to missing original FloorSet helper modules or sandbox/network restrictions, report the exact missing import or environment issue. Do not install, download, or rewrite around missing dependencies without approval.

## Module Workstreams

### Workstream 1: HardConstraintNormalizer

Task file: `doc/tasks/hard-constraint-normalizer.md`.

Expected writes:

- Private records and helper functions in `my_optimizer.py`.
- `doc/tasks/progress.md` status entries.

Implementation requirements:

- Define `BlockSpec` and `NormalizedProblem` or equivalent simple records.
- Implement `normalize(...)` using raw solve inputs.
- Interpret `constraints[:, 0]` as fixed-shape, `constraints[:, 1]` as preplaced, `constraints[:, 2]` as MIB annotation, `constraints[:, 3]` as grouping/cluster annotation, and `constraints[:, 4]` as boundary bitmask.
- Interpret `target_positions` sentinel `-1` exactly once during normalization.
- Use exact target dimensions for fixed/preplaced blocks.
- Use square `sqrt(area) x sqrt(area)` dimensions for soft blocks initially.
- Record malformed fixed/preplaced metadata for checker rejection.

Local verification:

- Synthetic soft/fixed/preplaced cases verifying normalized dimensions, anchors, and annotations.

### Workstream 2: FeasibilityChecker

Task file: `doc/tasks/feasibility-checker.md`.

Expected writes:

- Private `FeasibilityReport` and checker helpers in `my_optimizer.py`.
- `doc/tasks/progress.md` status entries.

Implementation requirements:

- Validate output length, finite values, and positive dimensions.
- Check pairwise overlap with evaluator threshold `1e-6`.
- Check soft-block area relative error `<= 0.01`.
- Check fixed/preplaced dimensions and preplaced x/y with tolerance `1e-4`.
- Return diagnostic counts and messages for fallback routing.

Local verification:

- Synthetic touching-rectangles pass.
- Synthetic positive-overlap fails.
- Area tolerance boundary cases pass/fail correctly.
- Fixed/preplaced mismatches fail.
- Malformed tuples fail.

### Workstream 3: DiffusionGuidanceAdapter

Task file: `doc/tasks/diffusion-guidance-adapter.md`.

Expected writes:

- Private `Guidance` record and model-output adapter logic in `my_optimizer.py`.
- `doc/tasks/progress.md` status entries.

Implementation requirements:

- Preserve existing model initialization and checkpoint behavior.
- Convert finite model x/y output into predicted positions/centers.
- Ignore model-predicted dimensions for legality.
- Exclude anchors from movable order.
- Return deterministic fallback guidance on missing checkpoints, model exceptions, invalid shapes, or non-finite predictions.

Local verification:

- Synthetic finite predictions produce deterministic order.
- Non-finite predictions fall back deterministically.
- Missing checkpoints do not prevent legal placement.

### Workstream 4: AnchorAwareLegalizer

Task file: `doc/tasks/anchor-aware-legalizer.md`.

Expected writes:

- Private `PlacementState`, candidate generation, scoring, and legalizer helpers in `my_optimizer.py`.
- Changes to `MyOptimizer.solve(...)` orchestration.
- `doc/tasks/progress.md` status entries.

Implementation requirements:

- Insert preplaced anchors exactly before movable placement.
- Generate bounded deterministic candidates from occupied rectangle edges, bounding-box points, origin/corner points, and optional predicted positions.
- Reject candidates that overlap occupied rectangles.
- Score feasible candidates with deterministic tie-breaks. Start with simple terms; HPWL-aware scoring may be deferred until feasibility is stable.
- Return failed/incomplete state when placement cannot be completed so fallback can run.
- Remove or bypass the current unsafe post-pack preplaced coordinate overwrite and boundary move path.

Local verification:

- One anchor plus one movable avoids anchor.
- Multiple anchors with gaps either fill legal spaces or expand layout.
- Missing guidance still places deterministically.
- Candidate tie-breaks are reproducible.

### Workstream 5: FeasibleFallbackPacker

Task file: `doc/tasks/feasible-fallback-packer.md`.

Expected writes:

- Private fallback packer helpers in `my_optimizer.py`.
- Fallback orchestration in `MyOptimizer.solve(...)`.
- `doc/tasks/progress.md` status entries.

Implementation requirements:

- Insert anchors exactly.
- Use deterministic movable order from guidance or stable fallback.
- Use a conservative row/strip cursor strategy that advances around collisions and expands layout rather than overlapping.
- Preserve all dimensions.
- Output placements in block-index order.
- Run `FeasibilityChecker` before returning fallback output.
- If anchors are contradictory or dimensions invalid, surface diagnostics; do not silently claim guaranteed feasibility for impossible inputs.

Local verification:

- Soft-only blocks pack without overlap.
- Preplaced anchors remain exact.
- Movables avoid anchors.
- Missing guidance works.
- Overlapping immutable anchors are reported infeasible.

### Workstream 6: SoftConstraintImprover

Task file: `doc/tasks/soft-constraint-improver.md`.

Expected writes:

- Private soft-improver helper in `my_optimizer.py`.
- Optional orchestration in `MyOptimizer.solve(...)`.
- `doc/tasks/progress.md` status entries.

Implementation requirements:

- A no-op implementation is acceptable for the first hard-feasibility pass.
- If boundary movement is implemented, do it only for non-preplaced blocks, keep dimensions unchanged, and accept moves only after `FeasibilityChecker` approves the full placement.
- Defer grouping and MIB transformations unless their exact move rules are clear and hard-feasibility-preserving.

Local verification:

- No-op preserves placement exactly.
- Boundary conflict move is rejected if implemented.
- Preplaced block with boundary mask is not moved.

### Workstream 7: Solve Orchestration and Cleanup

Expected writes:

- `MyOptimizer.solve(...)` in `my_optimizer.py`.
- `doc/tasks/progress.md` status entries.

Implementation requirements:

- Build the normalized problem first.
- Build advisory guidance second.
- Run anchor-aware legalizer.
- Run soft improver only on a complete candidate.
- Run checker before return.
- If optimized path fails, run fallback and check it.
- Return only block-index ordered tuples.
- Decide the least risky behavior for unresolved fallback failure. Prefer returning the best deterministic candidate with explicit internal diagnostics over raising inside evaluator loops only if the evaluator would otherwise mark the entire case as an exception. Document the choice in code comments and progress notes.

Local verification:

- Direct synthetic `solve()` calls for soft-only, fixed-shape, preplaced-anchor, boundary-conflict, and fallback-trigger cases.
- Evaluator validation commands after approval.

## Subagent Plan

Use subagents only when they reduce risk. Because production code is concentrated in `my_optimizer.py`, avoid simultaneous write access to that file.

Good subagent candidates:

- Read-only evaluator semantics audit: inspect `iccad2026_evaluate.py` and report exact hard-check tolerances and target-position construction. May not edit files.
- Read-only current optimizer audit: inspect `my_optimizer.py` and report current inference/legalizer/control-flow integration points. May not edit files.
- Test-case design subagent: draft synthetic cases for normalizer, checker, legalizer, fallback, and solve orchestration. May edit only a temporary notes file if explicitly approved; otherwise report snippets to the main agent.
- FeasibilityChecker implementation subagent: may edit only a clearly delimited checker block in `my_optimizer.py` and `doc/tasks/feasibility-checker.md` status if no other agent is editing `my_optimizer.py`.
- Fallback/Legalizer implementation subagent: may edit only a clearly delimited legalizer/fallback block in `my_optimizer.py` and matching task statuses if no other agent is editing `my_optimizer.py`.

Main-agent-only integration:

- `MyOptimizer.solve(...)` orchestration.
- Any change that removes or bypasses current unsafe post-pack correction logic.
- Any final merge when multiple subagents propose edits to `my_optimizer.py`.
- Final evaluator runs and final progress tracker update.

Shared or integration-only files:

- `my_optimizer.py`: shared production file. One writer at a time.
- `doc/tasks/progress.md`: update after each module; main agent owns final consistency.
- `iccad2026_evaluate.py`: read-only unless user explicitly requests evaluator changes.
- `training_example.py`: read-only for this objective.
- `LiteTensorDataTest/`: read-only.

If subagents are unavailable, implement workstreams sequentially in the order below.

## Implementation Order

1. Re-read `AGENTS.md`, planning docs, and task files; run `git status --short --branch`.
2. Inspect `my_optimizer.py` and evaluator hard checks in `iccad2026_evaluate.py`.
3. Implement `HardConstraintNormalizer`.
   - Local check: synthetic normalization for soft, fixed, and preplaced blocks.
   - Update `doc/tasks/progress.md`.
4. Implement `FeasibilityChecker`.
   - Local check: synthetic overlap, area, fixed/preplaced, and malformed cases.
   - Update `doc/tasks/progress.md`.
5. Implement `DiffusionGuidanceAdapter`.
   - Local check: finite and invalid guidance paths.
   - Update `doc/tasks/progress.md`.
6. Implement `FeasibleFallbackPacker`.
   - Local check: soft-only and anchor cases pass the checker.
   - Update `doc/tasks/progress.md`.
7. Implement `AnchorAwareLegalizer`.
   - Local check: anchors inserted first, movables avoid occupied rectangles, deterministic output.
   - Update `doc/tasks/progress.md`.
8. Implement `SoftConstraintImprover`.
   - Prefer no-op initially unless boundary movement can be made obviously safe.
   - Local check: no-op or boundary conflict case.
   - Update `doc/tasks/progress.md`.
9. Integrate `MyOptimizer.solve(...)`.
   - Remove or bypass unsafe post-pack preplaced/boundary overwrites.
   - Ensure fallback routing and final checker acceptance are wired.
   - Update `doc/tasks/progress.md`.
10. Run approved quality gates:
    - Start with `python iccad2026_evaluate.py --validate my_optimizer.py`.
    - Then run `python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0`.
    - Then run `python iccad2026_evaluate.py --evaluate my_optimizer.py` when single-case feasibility is stable.
11. If any evaluator case remains infeasible, record the exact hard-violation category in `doc/tasks/progress.md`, add or save a focused regression fixture only if appropriate and approved, and iterate on the responsible module.

## Testing and Quality Gates

Actual discovered commands:

```bash
python iccad2026_evaluate.py --validate my_optimizer.py
python iccad2026_evaluate.py --validate my_optimizer.py --quick
python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0
python iccad2026_evaluate.py --evaluate my_optimizer.py
python iccad2026_evaluate.py --evaluate my_optimizer.py --save-solutions
python iccad2026_evaluate.py --score my_optimizer_solutions.json
python iccad2026_evaluate.py --baseline
python iccad2026_evaluate.py --info
python iccad2026_evaluate.py --visualize --test-id 0
python iccad2026_evaluate.py --training
python training_example.py
```

Approval and output caveats:

- `--validate`, `--evaluate`, `--baseline`, `--score`, `--visualize`, `--training`, and `training_example.py` require approval under current workspace rules.
- Evaluation writes a results JSON by default.
- `--save-solutions` writes `my_optimizer_solutions.json`.
- `--baseline` writes `baseline_metrics.json` by default.
- `--visualize --test-id 0` writes `validation_case_0.png` by default.
- `python training_example.py` is long-running, uses training data, auto-resumes from `checkpoints/`, and writes `.pth` checkpoints. Do not run it as part of this implementation unless the user explicitly asks.

Missing gates:

- Build: no build command discovered.
- Unit tests: no command or test framework discovered.
- Lint: no command discovered.
- Format check: no command discovered.
- Type/static analysis: no command discovered.
- CI: no configuration discovered.

Minimum verification ladder:

- After helper implementation, run synthetic checks for module invariants where possible.
- Use `LiteTensorDataTest/` through the evaluator for public local validation/evaluation. Do not use lab-server `data_lite` as a validation gate unless the user first approves and defines a deliberate split policy.
- After any code change to `my_optimizer.py`, get approval and run `python iccad2026_evaluate.py --validate my_optimizer.py`.
- After validation passes, get approval and run `python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0`.
- Before declaring contest-style local readiness, get approval and run `python iccad2026_evaluate.py --evaluate my_optimizer.py`; target 100/100 feasible local public validation cases before score tuning.
- If saving comparison artifacts is useful, get approval and run `python iccad2026_evaluate.py --evaluate my_optimizer.py --save-solutions`, then `python iccad2026_evaluate.py --score my_optimizer_solutions.json`.

Record for every command:

- exact command;
- working directory;
- exit status;
- key output lines;
- feasible count, total score, average cost, and average runtime when available;
- hard-violation categories if any case fails.

## Progress Tracking

Maintain `doc/tasks/progress.md` throughout implementation.

Required progress behavior:

- Add timestamped or clearly labeled checkpoint entries under the relevant module or full-project gate when work starts, completes, is blocked, or is verified.
- Preserve existing checked boxes when the underlying task still applies.
- Mark module checkboxes only after implementation and module-specific verification are complete.
- Mark full-project gates only when the actual gate is configured and has passed, or note that it is missing/not configured.
- When an evaluator command is not run because approval was not granted, record it as not run, not passed.
- When a command fails because of missing imports, dependencies, or data paths, record the exact error and the next required user decision.
- Keep progress task-oriented; avoid long narrative logs in the tracker.

## Commit or Checkpoint Strategy

Do not create commits unless the user explicitly asks.

Before editing:

- Run `git status --short --branch`.
- Note whether changes in files you plan to edit are yours, user changes, or pre-existing untracked docs.
- Do not revert unrelated changes.

During implementation:

- Keep a clear diff grouped by module.
- Prefer completing and verifying one module before starting the next.
- If commits are requested, create logical commits by workstream, such as normalizer/checker, guidance, legalizer/fallback, solve integration, and tests/docs.
- Do not initialize a new repository, change branches, stash, rebase, merge, push, or rewrite history unless the user explicitly asks.

Generated artifacts:

- Do not leave evaluator result JSON, saved solutions, visualizations, checkpoints, caches, or temporary fixtures unless they were approved and are intentionally part of the deliverable.
- Do not delete user-created optimizer variants, checkpoints, contest data, or results without explicit approval.

## Acceptance Criteria

The implementation is complete when all applicable items are true:

- `my_optimizer.py` implements the six planned logical modules or equivalent private helpers.
- `MyOptimizer.solve(...)` uses the new pipeline and no longer relies on unsafe post-pack preplaced/boundary overwrites.
- The optimizer returns exactly one finite `(x, y, width, height)` tuple per block.
- Fixed-shape dimensions and preplaced x/y/width/height are preserved within evaluator tolerance.
- Soft-block areas are within 1% relative error.
- Returned placements are overlap-free under evaluator semantics.
- Boundary/grouping/MIB handling is either no-op or explicitly hard-feasibility-preserving.
- Missing checkpoints do not prevent hard-feasible output for normal non-contradictory inputs.
- `doc/tasks/progress.md` reflects completed, blocked, and verified work.
- Synthetic checks or an approved lightweight harness cover normalizer, checker, anchor legalizer, fallback, soft-improvement no-op/boundary safety, and solve fallback routing.
- Configured quality gates pass:
  - build passes if a build command becomes configured;
  - unit tests pass if added/configured;
  - lint passes if configured;
  - format check passes if configured;
  - type/static analysis passes if configured;
  - evaluator gates pass when approved.
- Required evaluator readiness gates pass after approval:
  - `python iccad2026_evaluate.py --validate my_optimizer.py`
  - `python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0`
  - `python iccad2026_evaluate.py --evaluate my_optimizer.py` with 100/100 feasible local public validation cases before score tuning is considered successful.
- Documentation is updated only where needed.
- No unrelated files are changed.
- Contest data, evaluator scoring, training data, labels, and checkpoints are untouched unless explicitly requested.

## Uncertainty Protocol

Make conservative, documented assumptions when safe:

- If fallback order is undecided, choose a deterministic order such as descending area with block-index tie-break or block-index order; document the choice.
- If legalizer candidate cap is undecided, choose a simple bounded cap, benchmark it before full validation, and document the tradeoff.
- If HPWL-aware scoring threatens feasibility or schedule, defer it until 100/100 feasibility is stable.
- If grouping or MIB move rules are unclear, keep `SoftConstraintImprover` no-op for those constraints.
- If both optimized and fallback placements fail due to contradictory immutable anchors or malformed metadata, preserve diagnostics and choose the least harmful evaluator-compatible behavior; document it.

Ask the user before proceeding when blocked by:

- missing requirements that change behavior;
- conflicts between docs, README, PDF, and evaluator code;
- destructive file operations;
- installing dependencies, downloading data, or using external services;
- credentials, tokens, or private data;
- evaluator/training commands requiring approval;
- missing helper modules or package layout that prevent validation;
- decisions that would require changing `iccad2026_evaluate.py`, contest data, checkpoints, or training pipeline.

Do not ask for clarification when the docs provide a safe implementation path. Continue with the conservative feasibility-first approach.

## Final Response Requirements

When finished, respond concisely with:

- implementation summary grouped by workstream;
- changed files grouped by workstream;
- tests and quality gates run, with exact commands, exit status, and key output summaries;
- evaluator feasible count, total score, average cost, and average runtime when available;
- known limitations and unresolved open questions;
- any follow-up required from the user, such as approving full validation, restoring missing helper modules, or deciding training/checkpoint strategy.

If a required gate could not be run, say exactly why. If no issues remain and all approved gates passed, state that clearly.
