# rec-sys-nn-optimization

This project explores how different neural recommender architectures and
optimization strategies perform on graph-based recommendation tasks. It uses
implicit-feedback link prediction as the common framing, so user-item
interactions and graph edges can be evaluated with the same training and
scoring pipeline.

The core experiment harness lives in `src/experiment_suite.py`, with the
companion notebook in `src/recommender_optimization_experiments.ipynb` used for
running, comparing, and visualizing experiments. The code supports MovieLens
100K and optional OGB link-prediction data, then trains a set of lightweight
PyTorch recommender models including embedding MLP, GNN, RHMM-inspired,
PSL-DNN-inspired, GNN-BiLSTM, and reinforcement-learning bandit aggregation
variants.

The experiments compare both model families and optimization approaches:

- gradient optimizers for recommender training
- population-based search methods
- grid search, random search, and differential evolution for hyperparameter
  optimization
- validation and test metrics such as ROC AUC, average precision, and accuracy
- runtime and resource usage, including wall time, RAM, CPU, and GPU memory

Dataset artifacts are cached under `data/`, and experiment outputs are written
under `results/`. The implementation is intentionally compact and notebook
friendly: it is designed for side-by-side experimentation and analysis rather
than exact reproduction of every source architecture.

## Setup

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Then open and run:

```text
src/recommender_optimization_experiments.ipynb
```

The experiment suite will create local `data/` and `results/` directories as
needed.

## Evaluation protocol

Model parameters are learned from the training split. Hyperparameter-search
candidates are compared using mean validation AUC over five shared trial seeds;
their test metrics are left unset. After each search method selects the
configuration with the highest mean validation AUC, that configuration is
retrained and evaluated on the test split once per seed. Final tables report the
mean and standard deviation across those trials. This keeps the test set
independent of hyperparameter selection while measuring sensitivity to random
initialization and sampling.

The default trial seeds are `42`, `123`, `456`, `789`, and `2026`. Dataset
caches and result files are anchored to this repository's root-level `data/`
and `results/` directories regardless of the process's working directory.
