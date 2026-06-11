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
