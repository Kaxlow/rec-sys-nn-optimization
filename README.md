# A Comparison of Neural Recommender System Architectures

Independent study research project by Ken Low, University of Colorado Boulder.

This repository accompanies the manuscript, *A Comparison of
Neural Recommender System Architectures*. It investigates whether additional
architectural complexity improves recommendation-style edge prediction when
models are evaluated under a shared, intentionally lightweight training and
resource budget.

## Research questions

The study examines three questions:

1. How do embedding, graph, sequential, relational, and policy-guided neural
   recommender architectures compare under a common training protocol?
2. How do grid search, random search, and Differential Evolution-style search
   compare when tuning the same model and search space?
3. How do gradient-based optimizers compare with population-based,
   gradient-free neural-network training under a constrained compute budget?

The study examines whether architectural complexity, explicit
relational features, tuning strategy, and optimizer choice provide measurable
benefits relative to their computational cost.

## Experimental design

### Datasets

| Dataset | Role in the study | Transformation |
| --- | --- | --- |
| [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) | Small user-item recommendation benchmark | Ratings of 4 or 5 become positive edges in a bipartite graph; lower ratings are excluded rather than labeled as dislikes. |
| [ogbl-collab](https://ogb.stanford.edu/docs/linkprop/#ogbl-collab) | Larger graph-learning proxy | Author collaborations remain edges in a homogeneous graph using the OGB link-prediction split. |

MovieLens interactions are ordered by timestamp and divided into 80% training,
10% validation, and 10% test data. This approximates deployment: models learn
from earlier interactions and predict later ones. Negative examples are sampled
from unobserved node pairs. They represent missing retained interactions, not
explicit user dislikes.

`QUICK_MODE` controls dataset sampling only; it does not change epoch counts or
mini-batch schedules. With `QUICK_MODE=True`, MovieLens retains the earliest
40,000 positive interactions after filtering and chronological sorting, giving
32,000 training, 4,000 validation, and 4,000 test positives. For ogbl-collab,
quick mode preserves the official split structure while reproducibly retaining
120,000 training positives and 20,000 positive and 20,000 negative edges in
each validation and test partition.

### Model families

| Repository name | Study role |
| --- | --- |
| `dnn` | Embedding-based multilayer-perceptron baseline |
| `gnn` | Mean-aggregation graph-neural-network baseline |
| `rhmm` | GRU-based, latent-regime approximation inspired by RHMM-style sequential recommendation |
| `psl_dnn` | DNN augmented with degree and neighbor-overlap features inspired by Probabilistic Soft Logic |
| `gnn_bilstm` | GNN embeddings combined with a bidirectional LSTM history encoder |
| `rl_gnn` | GNN with a lightweight bandit policy over layer-aggregation strategies |

The hybrid models are compact experimental approximations. In particular,
`rhmm` is not a full hidden Markov model, `psl_dnn` does not run a PSL inference
engine, and `rl_gnn` is not a full long-horizon reinforcement-learning
recommender. These names describe the motivating inductive bias, not exact
reimplementations of the cited architectures.

### Comparisons and measurements

The harness runs three experiment suites:

- Model families: the six architectures above under common default settings.
- Hyperparameter search: grid search, random search, and DE-style search each
  receive 24 candidate evaluations over the same embedding-size, hidden-size,
  dropout, and learning-rate domain.
- Optimizer families: Adam and momentum SGD versus particle swarm optimization
  and an evolution-strategy-style optimizer. All four methods use a shared
  16-dimensional embedding, 32-unit hidden layer, and 0.1 dropout setting.

Each run records ROC AUC, average precision, binary accuracy, wall-clock time,
RAM, CPU usage, and GPU memory. AUC and average precision are the most useful
predictive measures here; binary accuracy depends on the sampled, balanced
positive-negative evaluation set.

## Evaluation protocol

The default trial seeds are `42`, `123`, `456`, `789`, and `2026`. Every
compared method uses the same seed set so initialization, negative sampling,
dropout, and population generation vary in a controlled and reproducible way.

Hyperparameter selection follows this sequence:

1. Fit each candidate on the training split for every seed.
2. Compare candidates using mean validation AUC only.
3. Select the configuration with the highest mean validation AUC.
4. Retrain that configuration once per seed and evaluate the test split.
5. Report the mean and sample standard deviation of the five test trials.

Validation-search rows deliberately contain `NaN` test metrics and
`evaluated_on_test=False`. This prevents the test set from influencing
hyperparameter selection.

## Current repeated-trial results

The following headline results come from the current
[`results/experiment_results.csv`](results/experiment_results.csv) with
`QUICK_MODE=True`. Values are mean test ROC AUC +/- one sample standard
deviation across five seeds.

| Comparison | MovieLens 100K | ogbl-collab |
| --- | --- | --- |
| Best model family | DNN: 0.5057 +/- 0.0054 | PSL-DNN: 0.9310 +/- 0.0013 |
| Best HPO method | Grid: 0.6476 +/- 0.0170 | Grid: 0.7864 +/- 0.0115 |
| Best optimizer | PSO: 0.5209 +/- 0.0236 | Adam: 0.7296 +/- 0.0215 |

The three MovieLens hyperparameter-search methods remain close after repeated
trials. Grid and random search are also close on ogbl-collab, although grid has
the highest mean AUC. Equal candidate counts do not imply equal compute cost:
the ogbl-collab grid protocol recorded 6,521.8 seconds in total, compared with
1,592.0 seconds for random search and 1,990.3 seconds for DE-style search.
Conclusions remain bounded to sampled edge prediction, short training
schedules, five seeds, and lightweight proxy implementations.

## Repository layout

```text
.
|-- .github/workflows/ci.yml
|-- data/                              # downloaded datasets; ignored by Git
|-- results/
|   `-- experiment_results.csv         # repeated-trial result rows
|-- scripts/
|   `-- build_raw_data_samples_notebook.py
|-- src/
|   |-- experiment_suite.py            # datasets, models, training, and search
|   `-- recommender_optimization_experiments.ipynb
|-- tests/
|-- LICENSE
|-- README.md
`-- requirements.txt
```

Paths are resolved from `experiment_suite.py`, so dataset caches and results
always use root-level `data/` and `results/` directories regardless of the
shell or notebook working directory.

## Reproducing the study

The code in this repository was ran with Python 3.11.15.

```bash
python -m venv .venv
```

Activate the environment on macOS or Linux:

```bash
source .venv/bin/activate
```

Or on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies and launch Jupyter:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab src/recommender_optimization_experiments.ipynb
```

Run the notebook from the first cell after restarting the kernel. The loaders
download datasets on first use, cache them under `data/`, and write flat result
rows to `results/experiment_results.csv`.

The latest five-seed quick-mode result file records approximately 3.18 hours of
summed run time on the original eight-core, 16 GB RAM CPU environment with
cached datasets. Runtime will vary by hardware. Full-data execution can take
substantially longer.

## Tests and continuous integration

The test suite covers deterministic seeding, graph preprocessing, negative-edge
sampling, metric calculation, stable repository paths, repeated-seed execution,
and validation-only hyperparameter selection. It does not download datasets or
run the full experiment matrix.

Run it locally with:

```bash
python -m unittest discover -s tests -v
```

GitHub Actions runs compilation, dependency checks, notebook-structure
validation, and all unit tests on pushes and pull requests.

## Limitations and responsible interpretation

- `ogbl-collab` is an author-collaboration benchmark, not a production
  user-item recommendation log.
- MovieLens ratings below 4 are discarded; sampled missing pairs are not proven
  dislikes.
- Quick mode restricts dataset sizes but retains the same training schedules;
  the sampled data can change method rankings.
- Runtime and resource measurements are machine-dependent.
- The current evaluation emphasizes sampled edge prediction. Ranking metrics
  such as NDCG@K, Recall@K, and Hit Rate@K remain important future work.
- Repeated seeds quantify some stochastic variation, but they do not address
  dataset shift, alternative splits, or external validity.

The project is intended for research and education, not deployment or decisions
about individual users.

## Selected references

- Bach, S. H., Broecheler, M., Huang, B., & Getoor, L. (2017).
  [Hinge-loss Markov random fields and probabilistic soft logic](https://jmlr.org/papers/v18/15-631.html).
- Bergstra, J., & Bengio, Y. (2012).
  [Random search for hyper-parameter optimization](https://www.jmlr.org/papers/v13/bergstra12a.html).
- He, X., et al. (2017).
  [Neural collaborative filtering](https://doi.org/10.1145/3038912.3052569).
- He, X., et al. (2020).
  [LightGCN](https://doi.org/10.1145/3397271.3401063).
- Kennedy, J., & Eberhart, R. (1995).
  [Particle swarm optimization](https://doi.org/10.1109/ICNN.1995.488968).
- Kingma, D. P., & Ba, J. (2014).
  [Adam](https://doi.org/10.48550/arXiv.1412.6980).
- Storn, R., & Price, K. (1997).
  [Differential evolution](https://doi.org/10.1023/A:1008202821328).
- Wang, X., et al. (2019).
  [Neural graph collaborative filtering](https://doi.org/10.1145/3331184.3331267).

## License

The source code is released under the [MIT License](LICENSE). Downloaded
datasets are not redistributed by this repository and remain subject to their
respective providers' terms.
