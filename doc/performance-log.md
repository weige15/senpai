# Performance Log

| Attempt | Date | Change | Correctness Status | Score | Runtime | Memory | Kept? | Notes |
|---:|---|---|---|---:|---:|---:|---|---|
| 0 | 2026-06-15 | Feasibility-first legalizer/fallback baseline recorded in `doc/tasks/progress.md` | 100/100 feasible | 9.9565 | 0.45 s avg | Unknown | Baseline | Full evaluator wrote `my_optimizer_results.json`; this was before the `zz_final.pth` result discussed here. |
| 1 | 2026-06-15 | User-provided trained checkpoint run using `checkpoints/zz_final.pth` | 100/100 feasible | 9.7460 | 1.13 s avg | Unknown | Baseline | Saved as `iccad2026contest/eval_full_after_training.json`; 72/100 cases capped at feasible cost `9.999999`. |
| 2 | 2026-06-15 | Boundary-only `SoftConstraintImprover` with hard-checker acceptance and grouping/MIB-aware soft proxy | 100/100 feasible | 9.7367 | 0.68 s avg | Unknown | Yes | Full evaluator output saved to `/tmp/senpai_post_boundary.json`; capped cases dropped from 79 to 64 versus local no-checkpoint attempt 0. No local `iccad2026contest/checkpoints/` directory was present, so the trained checkpoint baseline was not rerun. |
