# Repository Map

## Repository Summary

This workspace is a partial ICCAD 2026 FloorSet Challenge contest directory at `/home/kuotzuwei15/pda/senpai`.

The repository centers on developing and evaluating `my_optimizer.py` against the provided FloorSet validation data. The visible implementation is Python and PyTorch based. The main local files discovered are `README.md`, `AGENTS.md`, `iccad2026_evaluate.py`, `my_optimizer.py`, `training_example.py`, `FloorplanningContest_ICCAD_2026_v10.pdf`, `doc/problem-brief.md`, and `LiteTensorDataTest/`.

## Directory Structure

- `README.md`: contest overview, constraints, scoring, setup notes, and command reference.
- `AGENTS.md`: local workflow and safety rules for this workspace.
- `iccad2026_evaluate.py`: contest evaluator, scoring utilities, optimizer base class, baseline optimizers, dataloader helpers, validation, scoring, visualization, and CLI.
- `my_optimizer.py`: current submitted optimizer implementation.
- `training_example.py`: training pipeline for the model architecture used by `my_optimizer.py`.
- `FloorplanningContest_ICCAD_2026_v10.pdf`: contest specification PDF.
- `LiteTensorDataTest/`: validation dataset directory with config directories numbered from `config_21` through `config_120`; each discovered config contains `litedata_1.pth` and `litelabel_1.pth`.
- `doc/problem-brief.md`: existing source-grounded problem brief.
- `doc/repo-map.md`: this repository map.

No `src/`, `tests/`, `scripts/`, package directory, or dependency manifest was discovered by `rg --files`.

## Main Source Files

- `iccad2026_evaluate.py`
  - Defines contest dataclasses: `SolutionMetrics`, `TestResult`, and `EvaluationResult`.
  - Implements quality and constraint helpers including `calculate_hpwl_b2b`, `calculate_hpwl_p2b`, `calculate_bbox_area`, `check_overlap`, `check_area_tolerance`, `check_dimension_hard_constraints`, `compute_cost`, `evaluate_solution`, and `compute_total_score`.
  - Defines `FloorplanOptimizer`, `RandomOptimizer`, `SimulatedAnnealingOptimizer`, and `ContestEvaluator`.
  - Provides helpers for training, validation, visualization, saved-solution scoring, and dataset loading.
  - Imports project modules that are not visible in this workspace: `litetestLoader`, `liteLoader`, `lite_dataset`, `cost`, and `utils`.

- `my_optimizer.py`
  - Defines neural-network components `NetlistGNN`, `TimestepEmbedder`, `DiTBlock`, and `DiTSmallFloorplanBackbone`.
  - Defines `BStarTreeLegalizer` for contour-based non-overlap packing.
  - Defines `MyOptimizer(FloorplanOptimizer)` with `solve(...)`.
  - Loads the latest `.pth` file from `checkpoints/` if the directory exists; otherwise it runs with initialized model weights.
  - Uses the official evaluator helpers by importing from `iccad2026_evaluate.py`.

- `training_example.py`
  - Defines the same main model components as the optimizer training side.
  - Provides `main()` for high-frequency checkpoint training.
  - Uses `get_training_dataloader` and `compute_training_loss_differentiable`.
  - Saves checkpoints under `checkpoints/`.
  - Imports from `iccad2026contest.iccad2026_evaluate`, which assumes the original FloorSet package layout rather than only the files visible here.

## Existing Tests

No standalone unit test files or test directories were discovered.

The contest evaluator provides validation and evaluation workflows:

- `python iccad2026_evaluate.py --validate my_optimizer.py`
- `python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0`
- `python iccad2026_evaluate.py --evaluate my_optimizer.py`
- `python iccad2026_evaluate.py --score my_optimizer_solutions.json`

Per `AGENTS.md`, these commands require user approval before execution because they may run for a while or write outputs depending on options.

## Build System

No `pyproject.toml`, `setup.py`, `setup.cfg`, `requirements.txt`, `requirements*.txt`, `Pipfile`, `poetry.lock`, `package.json`, `Makefile`, or other build configuration was discovered.

`README.md` references `iccad2026contest/requirements.txt`, but that file is not present in this workspace.

## Runtime or CLI Entry Points

- `iccad2026_evaluate.py` is the main contest CLI. Its docstring lists `--evaluate`, `--validate`, `--baseline`, `--score`, `--visualize`, and `--info`.
- `training_example.py` has `if __name__ == '__main__': main()` and is a training entry point.
- `my_optimizer.py` is loaded by the evaluator as an optimizer module and is not primarily a standalone CLI.

Likely local commands from the README and evaluator:

- `python iccad2026_evaluate.py --validate my_optimizer.py`
- `python iccad2026_evaluate.py --evaluate my_optimizer.py`
- `python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0`
- `python iccad2026_evaluate.py --baseline`
- `python iccad2026_evaluate.py --visualize --test-id 0`
- `python training_example.py`

## Data and Assets

- `LiteTensorDataTest/` is present and contains validation `.pth` data and label files under config directories `config_21` through `config_120`.
- `FloorplanningContest_ICCAD_2026_v10.pdf` is present as the contest specification.
- No `LiteTensorData/` training data directory was discovered.
- No `checkpoints/` directory was discovered by `rg --files`; `my_optimizer.py` and `training_example.py` both reference it.
- No image, web, or other asset directories were discovered.

## Existing Documentation

- `README.md` documents contest terminology, constraints, scoring, dataset downloads, dataloader helpers, getting started steps, optimizer API, and command reference.
- `AGENTS.md` documents local repository rules, safety constraints, known commands, and current workspace caveats.
- `doc/problem-brief.md` summarizes the contest objective, inputs, outputs, constraints, scoring, data, environment, assumptions, and open questions.
- `FloorplanningContest_ICCAD_2026_v10.pdf` is the formal contest specification.

## Detected Dependencies

Detected from imports in visible Python files:

- Python standard library: `argparse`, `dataclasses`, `datetime`, `importlib.util`, `json`, `math`, `os`, `random`, `re`, `sys`, `time`, `pathlib`, and typing utilities.
- Third-party Python packages: `torch`, `torch.utils.data`, `numpy`, `tqdm`, and optional `shapely`.
- Local or original FloorSet modules referenced by `iccad2026_evaluate.py`: `litetestLoader`, `liteLoader`, `lite_dataset`, `cost`, and `utils`.
- Local contest module import in `my_optimizer.py`: `iccad2026_evaluate`.
- Original-layout package import in `training_example.py`: `iccad2026contest.iccad2026_evaluate`.

The local or original FloorSet modules above were not discovered by `rg --files`, so imports may depend on files outside this partial workspace or on the original repository layout.

## Important Scripts

- `iccad2026_evaluate.py`: official local evaluator and utility script.
- `my_optimizer.py`: optimizer implementation under development.
- `training_example.py`: long-running training script that can write checkpoints.

No separate shell scripts or automation scripts were discovered.

## Current Git State

Running `git status --short --branch` in `/home/kuotzuwei15/pda/senpai` returned:

```text
fatal: not a git repository (or any of the parent directories): .git
```

`AGENTS.md` states that a `.git/` directory exists but is empty. This workspace should be treated as not currently having usable Git metadata unless that changes.

## Missing or Ambiguous Areas

- Dependency installation source is ambiguous: `README.md` references `iccad2026contest/requirements.txt`, but no requirements file was discovered.
- Several modules imported by `iccad2026_evaluate.py` are not visible in this workspace.
- The workspace appears to be a partial copy of the original FloorSet repository.
- No standalone tests, lint command, formatter command, type-check command, or CI configuration was discovered.
- No local training dataset directory `LiteTensorData/` was discovered.
- No local `checkpoints/` directory or model weights were discovered, although the optimizer looks for `.pth` weights there.
- Final contest submission packaging is not described by the visible local files beyond the optimizer API and evaluator commands.

## Notes for Future Skills

- Preserve the required `FloorplanOptimizer.solve(...)` API and return one `(x, y, width, height)` tuple per block.
- Treat `LiteTensorDataTest/` as read-only validation input data.
- Ask before running evaluation, validation, training, baseline generation, visualization, install, download, or checkpoint-producing commands.
- Prioritize hard feasibility: no overlaps, soft-block area within 1%, fixed-shape dimensions unchanged, and preplaced position and dimensions unchanged.
- Do not assume missing original FloorSet modules are available until imports are tested or the full repository layout is restored.
- If implementation work follows, first resolve whether the current partial workspace can run `iccad2026_evaluate.py` without the missing local modules.
