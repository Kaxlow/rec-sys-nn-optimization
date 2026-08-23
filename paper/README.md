# A Comparison of Neural Recommender Systems

## Manuscript

[Download the current draft manuscript](A%20Comparison%20of%20Neural%20Recommender%20Systems%20-%20Draft%20Manuscript.pdf).

## Status

This manuscript is an unpublished working draft. It has received feedback from [Dr. Abel Iyasele](https://experts.colorado.edu/display/fisid_172310#background) from the University of Colorado Boulder. It has not been formally peer reviewed or published.

The findings should be interpreted as preliminary and are subject to revision.

## Current version

- Version date: August 22, 2026
- Status: Draft manuscript
- Experiment mode: `QUICK_MODE=True`
- Trial count: Five seeds per reported comparison

## Relationship to this repository

The manuscript reports experiments implemented in this repository. The principal
materials are:

- [`../src/recommender_optimization_experiments.ipynb`](../src/recommender_optimization_experiments.ipynb):
  experiment execution, summaries, and visualizations
- [`../src/experiment_suite.py`](../src/experiment_suite.py):
  dataset preparation, models, training procedures, and search methods
- [`../results/experiment_results.csv`](../results/experiment_results.csv):
  raw repeated-trial results
- [`../README.md`](../README.md):
  setup instructions, experimental protocol, and headline findings

## Scope and limitations

The reported experiments use quick-mode dataset sampling while retaining the
configured training schedules. The model-family implementations are lightweight
research proxies and should not be interpreted as exact reproductions of every
referenced architecture.

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

See the repository’s [main README](../README.md) for environment setup and
execution instructions. The committed CSV contains the results used by the
current manuscript version.

## Review and feedback

Academic feedback is welcome. Please use the repository’s issue tracker for
technical questions or reproducibility problems.

## Rights

Copyright © 2026 Kenneth Low. All rights reserved unless otherwise indicated by a
repository or manuscript license.