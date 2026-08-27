# A Comparison of Neural Recommender Systems

## Manuscript

[Download the current full-results manuscript](A%20Comparison%20of%20Neural%20Recommender%20System%20Architectures%20-%20Draft%20Manuscript.pdf).

## Status

This manuscript is an unpublished working draft. It has received feedback from [Dr. Abel Iyasele](https://experts.colorado.edu/display/fisid_172310#background) from the University of Colorado Boulder. It has not been formally peer reviewed or published.

The findings should be interpreted as preliminary and are subject to revision.

## Current version

- Version date: August 25, 2026
- Status: Draft manuscript
- Experiment mode: `QUICK_MODE=False`
- Trial count: Five seeds per reported comparison
- Compute: NVIDIA RTX PRO 6000 Blackwell Server Edition, PyTorch 2.11.0+cu130, CUDA 13.0

## Relationship to this repository

The manuscript reports experiments implemented in this repository. The principal
materials are:

- [`../src/recommender_optimization_experiments.ipynb`](../src/recommender_optimization_experiments.ipynb):
  experiment execution, summaries, and visualizations
- [`../src/experiment_suite.py`](../src/experiment_suite.py):
  dataset preparation, models, training procedures, and search methods
- [`../results/full_rtx_pro_6000/experiment_results.csv`](../results/full_rtx_pro_6000/experiment_results.csv):
  primary full-data repeated-trial results
- [`../results/experiment_results.csv`](../results/experiment_results.csv):
  earlier quick-mode reproduction baseline
- [`../README.md`](../README.md):
  setup instructions, experimental protocol, and headline findings

## Scope and limitations

The current manuscript reports the completed full-data GPU run. The
model-family implementations remain lightweight research proxies and should not
be interpreted as exact reproductions of every referenced architecture. Full
mode loads the full dataset bundles, but each fit still uses short schedules
and sampled, balanced positive-negative mini-batches.

The results are limited to the included datasets, sampled edge-prediction
protocol, five random seeds, and the documented hardware and computational
budget.

## Citation

Because the manuscript has not been formally published, cite it as an
unpublished manuscript:

> Low, K. (2026). *A comparison of neural recommender systems*. Unpublished
> manuscript.

The citation information may change if the manuscript is submitted, archived as
a preprint, or formally published.

## Reproducibility

See the repository's [main README](../README.md) for environment setup and
execution instructions. The full-results CSV contains the observations used by
the current manuscript version.

## Review and feedback

Academic feedback is welcome. Please use the repository's issue tracker for
technical questions or reproducibility problems.

## Rights

Copyright © 2026 Kenneth Low. All rights reserved unless otherwise indicated by a
repository or manuscript license.
