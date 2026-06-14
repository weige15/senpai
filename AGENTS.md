# AGENTS.md

## Project Context
- This workspace is an ICCAD 2026 FloorSet Challenge contest directory.
- Primary task: develop and evaluate `my_optimizer.py` against the provided FloorSet validation data.
- Main files verified locally: `README.md`, `iccad2026_evaluate.py`, `my_optimizer.py`, `training_example.py`, `FloorplanningContest_ICCAD_2026_v10.pdf`, and `LiteTensorDataTest/`.
- Language/toolchain: Python 3 with PyTorch-based scripts.
- Package manager: Unknown in this workspace. `README.md` mentions `iccad2026contest/requirements.txt`, but no requirements file was found here.
- Git status: Unknown. A `.git/` directory exists but is empty; `git status --short --branch` reports this is not a git repository.

## Repository Rules
- Read project files before making assumptions; this workspace may be a partial copy of the original FloorSet layout.
- Keep changes focused on the requested optimizer, evaluation, training, or documentation task.
- Do not overwrite contest data, generated results, checkpoints, or user-created optimizer variants without explicit approval.
- Treat `LiteTensorDataTest/` as input data. Do not edit `.pth` dataset files unless the user explicitly requests data repair.
- Do not add secrets, tokens, private credentials, or copied environment values to files.

## Read-Only Discovery Commands
These commands are safe for future agents to run without approval:

```bash
pwd
ls -la
rg --files
rg -n "pattern" README.md iccad2026_evaluate.py my_optimizer.py training_example.py
sed -n '1,220p' README.md
sed -n '1,220p' iccad2026_evaluate.py
sed -n '1,220p' my_optimizer.py
sed -n '1,220p' training_example.py
```

Git discovery commands may be run, but currently fail because this workspace is not a valid Git repository:

```bash
git status --short --branch
git branch --show-current
git worktree list
```

## Commands Requiring Permission
Ask the user before running commands that install dependencies, download data, write files, use substantial compute, or may run for a long time.

Examples requiring approval:

```bash
pip install -r iccad2026contest/requirements.txt
python iccad2026_evaluate.py --validate my_optimizer.py
python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0
python iccad2026_evaluate.py --evaluate my_optimizer.py
python iccad2026_evaluate.py --evaluate my_optimizer.py --save-solutions
python iccad2026_evaluate.py --baseline
python iccad2026_evaluate.py --visualize --test-id 0
python iccad2026_evaluate.py --training
python training_example.py
```

Also ask before creating, deleting, moving, or renaming files; changing branches; committing; pushing; applying patches; formatting files; or running any training/checkpoint generation.

## Forbidden Commands
Do not run these unless the user explicitly requests the exact operation and understands the risk:

```bash
rm -rf
git reset --hard
git clean -fd
git checkout -- .
git restore .
git push --force
git push --force-with-lease
chmod -R
chown -R
sudo
```

## Build, Test, and Quality Gates
- Verified validation command from README/code: `python iccad2026_evaluate.py --validate my_optimizer.py`.
- Verified single-case evaluation command: `python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0`.
- Verified full validation evaluation command: `python iccad2026_evaluate.py --evaluate my_optimizer.py`.
- Verified scoring command for saved solutions: `python iccad2026_evaluate.py --score my_optimizer_solutions.json`.
- Verified optional output-producing commands: `--save-solutions`, `--baseline`, and `--output`.
- No verified lint, format, type-check, or unit-test command was found.
- `training_example.py` is a long-running training script that may download/use large training data and writes checkpoints under `checkpoints/`.
- `my_optimizer.py` looks for `.pth` model weights under `checkpoints/` and falls back when none exist.

## Documentation Rules
- Keep README changes aligned with the contest PDF and `iccad2026_evaluate.py`.
- Mark unverified setup, dependency, and path assumptions as unknown instead of inventing them.
- Prefer short operational notes in `AGENTS.md`; create `doc/workflow-rules.md` only if workflow details become too long for this file.

## Coding Rules
- Preserve the required optimizer API expected by `iccad2026_evaluate.py`.
- Keep hard contest constraints in mind: no overlaps, soft-block area within 1%, fixed-shape dimensions unchanged, and preplaced position/dimensions unchanged.
- Avoid changing validation data or contest scoring logic unless the user explicitly asks.
- Be careful with imports: current scripts reference modules from the original FloorSet layout that are not all present in this workspace.
- Use deterministic, narrowly scoped edits when possible; avoid broad refactors during optimization experiments.

## Git and Commit Rules
- This workspace is not currently a usable Git repository. Do not initialize a repository, create branches, commit, stash, merge, rebase, or push unless the user asks.
- If Git metadata becomes available later, check `git status --short --branch` before editing and preserve unrelated user changes.
- Never force-push or rewrite history unless the user explicitly requests it.

## Uncertainty Protocol
- If a dependency, dataset path, checkpoint, or module import is missing, report the exact missing item and ask before installing, downloading, or generating replacements.
- If a command may write output JSON, checkpoints, figures, caches, or downloaded data, ask for approval first.
- If contest rules in README, PDF, and code disagree, surface the disagreement with file references before changing behavior.
