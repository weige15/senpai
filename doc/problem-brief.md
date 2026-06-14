# Problem Brief

## Source Documents

- `AGENTS.md` - read. Repository operating rules and verified local workspace facts.
- `README.md` - read. Contest setup, task API, constraints, datasets, scoring, and evaluator commands.
- `FloorplanningContest_ICCAD_2026_v10.pdf` - read via `pdftotext`. Formal contest problem statement, inputs, outputs, objective function, datasets, and total-score definition.

## Assignment Objective

Develop a floorplanning optimizer for the ICCAD 2026 FloorSet Challenge. Given block areas, fixed terminals, weighted block-to-block connectivity, weighted block-to-terminal connectivity, and placement constraints, the optimizer must return block locations and dimensions that minimize wirelength and bounding-box area while satisfying all hard constraints.

The practical repository task is to implement the optimizer interface in `my_optimizer.py`, especially `solve()`, and verify it with the provided contest evaluator. The contest is performance-oriented and does not require a specific algorithmic paradigm; the PDF explicitly allows stochastic, data-driven, and hybrid approaches.

## Required Inputs

- Blocks `B = {b1, ..., bk}` with predetermined area targets.
- Terminals `T = {t1, ..., tr}` as fixed 2D points for external interfacing.
- Inter-module weighted connectivity between blocks.
- External weighted connectivity between blocks and terminals.
- Soft placement constraints:
  - Grouping constraints: grouped blocks should abut and form a single connected component.
  - Multi-Instantiation Block constraints: grouped blocks should share identical dimensions.
  - Boundary constraints: specified blocks should touch required bounding-box edges or corners.
- Hard placement constraints:
  - Area targets and dimensionality requirements.
  - Overlap-free rectangular block placement.
  - Fixed-shape immutability.
  - Preplaced location and dimension immutability.
- Repository API inputs from `README.md`: `block_count`, `area_targets`, `b2b_connectivity`, `p2b_connectivity`, `pins_pos`, and `constraints`.
- Boundary requirements are encoded in `constraints[:, 4]` as a bitmask: left = 1, right = 2, top = 4, bottom = 8.

## Required Outputs

- A placement for each block containing lower-left coordinates and dimensions.
- Repository API output from `solve()`: a list of `(x, y, width, height)` tuples, one per block.
- Coordinates may be floating point.
- The PDF defines block `bi` as occupying `[xi, xi + wi] x [yi, yi + hi]`.
- Saved evaluation solutions, when requested, are written by the evaluator as JSON via `--save-solutions`.

## Constraints

- No two blocks may overlap. Touching by shared edge is allowed.
- Soft-block realized area `w * h` must be within 1% relative error of its target area.
- Fixed-shape blocks must keep their exact input width and height.
- Preplaced blocks must keep their exact input location, width, and height.
- Any hard-constraint violation makes that test case infeasible and assigns cost `M = 10`.
- Aspect ratio is relaxed: any width/height ratio is allowed unless fixed-shape or preplaced dimensions apply.
- Coordinates are relaxed: integer coordinates are not required.
- Soft constraints affect the cost but do not by themselves make a solution infeasible.
- Submissions that reverse-engineer the dataset generator instead of developing genuine algorithmic solutions are disqualified by the PDF.
- `AGENTS.md` requires approval before installs, validation/evaluation runs, training, file generation, checkpoint generation, or other state-changing commands.

## Evaluation or Grading Criteria

- Lower cost is better.
- Per-test feasible cost is:

```text
Cost = (1 + 0.5 * (HPWL_gap + Area_gap)) * exp(2 * V_rel) * max(0.7, RuntimeFactor^0.3)
```

- Infeasible cost is exactly `10.0`.
- Feasible cost is capped below infeasible cost at `M - 1e-6`.
- `HPWL_gap` is the relative gap from baseline wirelength, clamped so beating the baseline gives no extra quality bonus.
- `Area_gap` is the relative gap from baseline bounding-box area, also clamped from below.
- `V_rel` is the normalized soft-constraint violation rate using grouping, boundary, and MIB violations only.
- Official runtime normalization is per hidden test case against the cross-submission median runtime. The local evaluator uses neutral runtime factor `1.0`.
- Final ranking uses an exponentially weighted average over 100 hidden test cases:

```text
Total Score = sum_i Cost[i] * exp(n_i / 12) / sum_j exp(n_j / 12)
```

- Larger block-count instances receive substantially higher weight; the README states the `116-120` bucket contributes about 34% of total score.
- Verified evaluator commands from `README.md` and `AGENTS.md` include:
  - `python iccad2026_evaluate.py --validate my_optimizer.py`
  - `python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0`
  - `python iccad2026_evaluate.py --evaluate my_optimizer.py`
  - `python iccad2026_evaluate.py --score my_optimizer_solutions.json`

## Required Deliverables

- An optimizer implementation compatible with the contest evaluator, normally in `my_optimizer.py`.
- The optimizer must expose the required `solve()` behavior and return one `(x, y, width, height)` tuple per block.
- Before final submission, the README instructs validating the optimizer with `python iccad2026_evaluate.py --validate my_optimizer.py`.
- The PDF says the final output format and submission instructions are specified in the FloorSet repository.
- No separate report, notebook, or written explanation is required by the provided sources.

## Relevant Methods From Papers

- None required by the provided sources.
- The PDF background mentions classical metaheuristics, hybrid learning frameworks, diffusion models, and ML-guided optimization as context, but it does not mandate any method.
- The proposal should not prematurely choose an algorithm solely because it is mentioned in related work.

## Data, Benchmarks, or Test Cases

- FloorSet-Lite training set: 1M samples, block counts from 21 to 120, available from Hugging Face according to README/PDF.
- Validation set: 100 samples, one per size from 21 to 120, available from Hugging Face and present locally as `LiteTensorDataTest/`.
- Hidden test set: 100 samples, same size range and format, used for final ranking.
- README states local validation data should be under `FloorSet/LiteTensorDataTest/` relative to the original repository root.
- The current workspace contains `LiteTensorDataTest/` directly in `/home/kuotzuwei15/pda/senpai`.
- Baseline HPWL and bounding-box area values are provided in the dataset.

## Implementation Environment

- Language/toolchain: Python 3 with PyTorch-based scripts.
- Contest framework file: `iccad2026_evaluate.py`.
- README setup references `pip install -r iccad2026contest/requirements.txt`, but that requirements file is not present in this workspace.
- README says the framework provides PyTorch DataLoaders `get_training_dataloader()` and `get_validation_dataloader()`.
- `training_example.py` may run long, use large data, and write checkpoints under `checkpoints/`.
- `my_optimizer.py` looks for `.pth` model weights in `checkpoints/`.
- `AGENTS.md` records that this workspace is not currently a usable Git repository even though an empty `.git/` directory exists.
- No verified lint, format, type-check, or unit-test command was found in the provided sources.

## Confirmed Facts

- The assignment is the ICCAD 2026 FloorSet Challenge.
- The repository task is to implement an optimizer compatible with `iccad2026_evaluate.py`.
- The optimizer output is a list of `(x, y, width, height)` tuples.
- Hard constraints are no overlap, soft-block area tolerance, fixed-shape immutability, and preplaced immutability.
- Soft constraints are grouping, MIB, and boundary constraints.
- The score combines HPWL gap, bounding-box area gap, soft-constraint violations, and runtime.
- Infeasible cases cost `10.0`.
- Larger test cases are weighted more heavily by `exp(n / 12)`.
- Training data has 1M samples; validation and hidden test each have 100 samples.
- The current workspace has validation data but no verified dependency manifest.

## Assumptions

- The requested brief should be written to `doc/problem-brief.md`, as required by the invoked skill.
- `README.md` and `FloorplanningContest_ICCAD_2026_v10.pdf` are authoritative for contest requirements unless later source documents conflict.
- `AGENTS.md` is authoritative for local workflow safety rules, not for contest scoring.
- The current workspace may be a partial copy of the original FloorSet repository.

## Open Questions

- Where is the missing dependency manifest referenced by README: `iccad2026contest/requirements.txt`?
- Are all modules imported by `iccad2026_evaluate.py` available in the actual runtime environment, given this workspace appears partial?
- What exact final submission packaging/upload process is required by the FloorSet repository or contest platform?
- Should future work target a quick feasible baseline, an ML-guided solver, a classical/hybrid optimizer, or another strategy?
- Are model checkpoints expected to exist locally before evaluation, or should the optimizer work well without them?

## Notes for Proposal Generation

- Start proposal work from the confirmed contest API, hard constraints, scoring formula, and local workspace limitations.
- Preserve the distinction between hard infeasibility constraints and soft score penalties.
- Prioritize feasibility because any hard-constraint violation assigns cost `10.0`.
- Account for large-instance weighting; scalability on 116-120 block cases matters strongly.
- Do not assume a required ML architecture; the sources allow multiple algorithmic paradigms.
- Resolve missing dependencies/imports and final submission packaging before committing to an implementation plan.
