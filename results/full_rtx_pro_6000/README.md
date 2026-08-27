# Full-data RTX PRO 6000 results

This directory contains the completed five-seed full-data experiment used as
the study's primary result set. The run was executed by CURC Slurm job
`31632601` on August 25, 2026 with `QUICK_MODE=False`.

## Provenance and validation

| Field | Value |
| --- | --- |
| Result file | `experiment_results.csv` |
| Rows | 850 |
| Seeds | 42, 123, 456, 789, 2026 |
| Validation-only HPO rows | 720 |
| Test-evaluated rows | 130 |
| Device | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| PyTorch / CUDA | 2.11.0+cu130 / 13.0 |
| CUDA capability / compiled architecture | 12.0 / `sm_120` |
| Slurm allocation | 1 GPU, 8 CPUs, 64 GB RAM |
| Job result | Exit status 0 |

All rows record `quick_mode=False`, `compute_device=cuda`, and
`run_complete=True`. Hyperparameter-search candidate rows have null test
metrics and `evaluated_on_test=False`; only the validation-selected
configuration for each method was retrained and evaluated on the test split.

The full bundles contained 44,300/5,537/5,538 MovieLens train/validation/test
positive edges and 1,179,052/60,084/46,329 ogbl-collab positive edges.

## Headline results

Values are mean test ROC AUC +/- one sample standard deviation across five
seeds.

| Comparison | MovieLens 100K | ogbl-collab |
| --- | --- | --- |
| Best model family | DNN: 0.4627 +/- 0.0096 | PSL-DNN: 0.9477 +/- 0.0015 |
| Best HPO method | Grid: 0.6562 +/- 0.0183 | DE-style: 0.9178 +/- 0.0051 |
| Best optimizer | PSO: 0.5107 +/- 0.0288 | Adam: 0.8650 +/- 0.0243 |

The MovieLens search methods are close: grid 0.6562 +/- 0.0183, DE-style
0.6542 +/- 0.0108, and random 0.6528 +/- 0.0145. On ogbl-collab, DE-style
search ranks ahead of grid (0.9114 +/- 0.0078) and random search
(0.9047 +/- 0.0121). These are descriptive comparisons; no formal
significance test was performed.

## Timing

The CSV records 346.6 seconds of summed per-fit wall time. The complete Slurm
job took about 9.1 minutes, including unit tests, environment and hardware
capture, notebook startup, result validation, and file staging. For
ogbl-collab HPO candidates, grid recorded 48.8 seconds, random search 51.9
seconds, and DE-style search 64.6 seconds. These GPU timings must not be pooled
with the CPU quick-mode timings in `../experiment_results.csv`.

Diagnostic provenance is retained locally under
`artifacts/full_rtx_pro_6000/` and is ignored by Git because it includes the
executed notebook, package freeze, scheduler details, and hardware logs.
