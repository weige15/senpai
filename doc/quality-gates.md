# Quality Gates

## Environment Summary

- Repository root / working directory: `/home/kuotzuwei15/pda/senpai`.
- Language/toolchain discovered: Python 3 with PyTorch-based contest scripts.
- Primary evaluator: `iccad2026_evaluate.py`.
- Primary optimizer under development: `my_optimizer.py`.
- Local validation data directory discovered: `LiteTensorDataTest/` under the repository root, with `config_21` through `config_120`.
- User-provided external training data context: lab-server dataset appears under `data_lite/worker_*`; this path was not accessible or verified in this local workspace.
- README documents original FloorSet layout where commands run from `iccad2026contest/` and data lives in the parent FloorSet directory. In this partial workspace, data-path assumptions may need checking before execution.
- No `package.json`, `pyproject.toml`, `requirements*.txt`, `Makefile`, `tox.ini`, `noxfile.py`, `pytest.ini`, `setup.cfg`, CI config, `scripts/`, `bin/`, or `tools/` file was discovered.
- README references `iccad2026contest/requirements.txt`, but that file is not present in this workspace.
- Current Git discovery returned branch `main` with no porcelain changes before this document was created.

## Build Commands

- Missing: no repository build command was discovered.
- Discovered setup command from README, not a build gate and not runnable from available files without the missing manifest:

```bash
pip install -r iccad2026contest/requirements.txt
```

Working directory documented by README: original `FloorSet/` root before entering `iccad2026contest/`.

Notes: requires network/package installation and a requirements file not present in this workspace.

## Unit Test Commands

- Missing: no standalone unit test command, test directory, or test framework configuration was discovered.

## Integration Test Commands

- Discovered: validate optimizer submission format.

```bash
python iccad2026_evaluate.py --validate my_optimizer.py
```

Working directory: `/home/kuotzuwei15/pda/senpai`.

Notes: imports the optimizer and evaluator dependencies; may fail if original FloorSet helper modules are missing from the runtime.

- Discovered: quick validation variant from evaluator examples.

```bash
python iccad2026_evaluate.py --validate my_optimizer.py --quick
```

Working directory: `/home/kuotzuwei15/pda/senpai`.

Notes: same dependency caveats as full validation.

## Lint Commands

- Missing: no lint command or lint configuration was discovered.

## Format Commands

- Missing: no formatter command or formatter configuration was discovered.

## Type-Check Commands

- Missing: no type-check command or type-check configuration was discovered.

## Static Analysis Commands

- Missing: no static-analysis command or configuration was discovered.

## Benchmark or Evaluator Commands

- Discovered: single validation-case evaluator run for debugging.

```bash
python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0
```

Working directory: `/home/kuotzuwei15/pda/senpai`.

Notes: reads validation data, imports evaluator dependencies, runs optimizer, and writes a default results JSON unless `--output` changes the path.

- Discovered: full validation evaluator run.

```bash
python iccad2026_evaluate.py --evaluate my_optimizer.py
```

Working directory: `/home/kuotzuwei15/pda/senpai`.

Notes: reads all 100 validation cases, may run for a long time, imports evaluator dependencies, runs optimizer, and writes a default results JSON.

- Discovered: evaluator run with saved solutions.

```bash
python iccad2026_evaluate.py --evaluate my_optimizer.py --save-solutions
```

Working directory: `/home/kuotzuwei15/pda/senpai`.

Notes: reads all validation cases, may run for a long time, and writes both results JSON and `my_optimizer_solutions.json`.

- Discovered: score saved optimizer solutions.

```bash
python iccad2026_evaluate.py --score my_optimizer_solutions.json
```

Working directory: `/home/kuotzuwei15/pda/senpai`.

Notes: requires an existing saved solutions JSON and validation data; may write output if `--output` is supplied.

- Discovered: baseline metric generation.

```bash
python iccad2026_evaluate.py --baseline
```

Working directory: `/home/kuotzuwei15/pda/senpai`.

Notes: reads validation data and writes `baseline_metrics.json` by default.

- Discovered: explicit output path option for baseline or evaluation results.

```bash
python iccad2026_evaluate.py --baseline --output baselines.json
```

Working directory: `/home/kuotzuwei15/pda/senpai`.

Notes: writes the requested JSON output file.

- Discovered: training-data exploration.

```bash
python iccad2026_evaluate.py --training
```

Working directory: `/home/kuotzuwei15/pda/senpai`.

Notes: uses the 1M-sample training dataset path by default; may require large local data or auto-download depending on loader behavior.

- Discovered: training pipeline.

```bash
python training_example.py
```

Working directory: `/home/kuotzuwei15/pda/senpai`.

Notes: long-running training command; uses large training data, GPU if available, auto-resumes from `checkpoints/`, and writes `.pth` checkpoints under `checkpoints/`.

## Smoke Test Commands

- Discovered: show contest/evaluator information without optimizer execution.

```bash
python iccad2026_evaluate.py --info
```

Working directory: `/home/kuotzuwei15/pda/senpai`.

Notes: lightweight CLI smoke test, but still imports top-level evaluator dependencies before reaching the info path.

- Discovered: visualize one validation case.

```bash
python iccad2026_evaluate.py --visualize --test-id 0
```

Working directory: `/home/kuotzuwei15/pda/senpai`.

Notes: reads validation data and writes `validation_case_0.png` by default.

## Verified Commands

- None. No build, test, lint, format, type-check, static-analysis, benchmark, evaluator, smoke-test, package-manager, or training command was run during this discovery session.

## Commands Not Run

- `pip install -r iccad2026contest/requirements.txt` - not run because installation requires approval, network/package changes, and the referenced requirements file is not present locally.
- `python iccad2026_evaluate.py --validate my_optimizer.py` - not run because evaluator commands require explicit approval.
- `python iccad2026_evaluate.py --validate my_optimizer.py --quick` - not run because evaluator commands require explicit approval.
- `python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0` - not run because evaluator commands require explicit approval and this command writes a results JSON by default.
- `python iccad2026_evaluate.py --evaluate my_optimizer.py` - not run because evaluator commands require explicit approval, may be long-running, and writes a results JSON by default.
- `python iccad2026_evaluate.py --evaluate my_optimizer.py --save-solutions` - not run because evaluator commands require explicit approval, may be long-running, and writes solutions/results JSON.
- `python iccad2026_evaluate.py --score my_optimizer_solutions.json` - not run because evaluator commands require explicit approval and the required saved solution file was not checked for existence.
- `python iccad2026_evaluate.py --baseline` - not run because it requires explicit approval and writes `baseline_metrics.json`.
- `python iccad2026_evaluate.py --baseline --output baselines.json` - not run because it requires explicit approval and writes the requested output file.
- `python iccad2026_evaluate.py --training` - not run because it may use large training data or downloads.
- `python training_example.py` - not run because it is a long-running training script and writes checkpoints.
- `python iccad2026_evaluate.py --info` - not run because the skill requires approval before executing repo tooling, even for smoke tests.
- `python iccad2026_evaluate.py --visualize --test-id 0` - not run because it requires explicit approval and writes `validation_case_0.png`.

## Missing Quality Gates

- Build: no local build command was discovered.
- Unit tests: no unit test command was discovered.
- Lint: no lint command was discovered.
- Format: no format-check command was discovered.
- Type-check: no type-check command was discovered.
- Static analysis: no static-analysis command was discovered.
- CI: no CI configuration was discovered.
- Dependency lock/install verification: README references a requirements file, but no local requirements file was discovered.

## Recommended Minimum Done Criteria

- For documentation-only changes: review the changed Markdown and confirm no evaluator behavior is affected.
- For optimizer code changes in `my_optimizer.py`: run `python iccad2026_evaluate.py --validate my_optimizer.py` first.
- For behavioral optimizer changes: run `python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0` before any full validation run.
- Before considering an optimizer change ready for contest validation: run `python iccad2026_evaluate.py --evaluate my_optimizer.py` and record the total score, feasible count, average cost, and average runtime.
- When comparing optimizer variants or preserving results: run `python iccad2026_evaluate.py --evaluate my_optimizer.py --save-solutions`, then `python iccad2026_evaluate.py --score my_optimizer_solutions.json`.
- Do not treat training as a minimum done gate. `python training_example.py` is a separate long-running experiment path that should be approved and scheduled explicitly.
- Consider adding a lightweight unit-test or smoke-test harness later for hard-constraint helpers and optimizer output shape, but no such gate currently exists in the repository.
