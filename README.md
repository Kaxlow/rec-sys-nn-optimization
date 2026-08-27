# A Comparison of Neural Recommender System Architectures

Independent study research project by Ken Low, University of Colorado Boulder.

This repository accompanies the manuscript, *A Comparison of
Neural Recommender System Architectures*. It investigates whether additional
architectural complexity improves recommendation-style edge prediction when
models are evaluated under a shared, lightweight training and
resource budget.

## Research questions

The study examines three questions in the context of a common training budget:

1. How do embedding, graph, sequential, relational, and policy-guided neural
   recommender architectures perform relative to one another?
2. How do grid search, random search, and Differential Evolution-style search
   compare in hyperparameter tuning for the same model and search space?
3. How do gradient-based optimizers compare with population-based,
   gradient-free neural-network learning?

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

`QUICK_MODE` was added to allow the experiments to be run on the local machine in the absence of sufficient GPU hardware. `QUICK_MODE` controls dataset sampling only; it does not change epoch counts or mini-batch schedules. Interactive notebook execution defaults to
`QUICK_MODE=True`; batch jobs can override it through `REC_SYS_QUICK_MODE`
without modifying the notebook. With quick mode enabled, MovieLens retains the earliest
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
RAM, CPU usage, and GPU memory where applicable. AUC and average precision are the most useful
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

The primary results come from the completed full-data GPU run in
[`results/full_rtx_pro_6000/experiment_results.csv`](results/full_rtx_pro_6000/experiment_results.csv).
The run used `QUICK_MODE=False`, all five seeds, and one NVIDIA RTX PRO 6000
Blackwell Server Edition GPU. Values below are mean test ROC AUC +/- one sample
standard deviation across the five independently trained trials.

| Comparison | MovieLens 100K | ogbl-collab |
| --- | --- | --- |
| Best model family | DNN: 0.4627 +/- 0.0096 | PSL-DNN: 0.9477 +/- 0.0015 |
| Best HPO method | Grid: 0.6562 +/- 0.0183 | DE-style: 0.9178 +/- 0.0051 |
| Best optimizer | PSO: 0.5107 +/- 0.0288 | Adam: 0.8650 +/- 0.0243 |

The three MovieLens search methods remain close, with mean AUCs from 0.6528 to
0.6562. On full ogbl-collab, DE-style search ranks first by mean test AUC, while
grid search has the lowest measured candidate-search time: 48.8 seconds versus
51.9 seconds for random search and 64.6 seconds for DE-style search. These are
descriptive repeated-trial comparisons, not formal significance tests.

The earlier CPU quick-mode results remain available in
[`results/experiment_results.csv`](results/experiment_results.csv) as a
lightweight reproduction baseline; they are not pooled with the full GPU
results. In that run, the corresponding winners were DNN and PSL-DNN for the
model families, grid search on both datasets, PSO on MovieLens, and Adam on
ogbl-collab. The changed ogbl-collab HPO ranking illustrates why run mode and
hardware provenance must accompany reported results.

## Repository layout

```text
.
|-- .github/workflows/ci.yml
|-- data/                              # downloaded datasets; ignored by Git
|-- paper/                             # manuscript and publication notes
|-- results/
|   |-- experiment_results.csv         # completed five-seed quick run
|   `-- full_rtx_pro_6000/             # separate full-data GPU results
|-- scripts/
|   |-- build_raw_data_samples_notebook.py
|   `-- run_full_rtx_pro_6000.sh       # CURC GPU batch job
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

The quick CPU run used Python 3.11.15. The full GPU run used Python 3.11.13,
PyTorch 2.11.0+cu130, and CUDA 13.0.

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

The notebook accepts the following environment variables for unattended runs:

| Variable | Default | Purpose |
| --- | --- | --- |
| `REC_SYS_QUICK_MODE` | `true` | Use sampled data when true and full data when false. |
| `REC_SYS_REQUIRE_CUDA` | `false` | Fail instead of silently falling back to CPU. |
| `REC_SYS_RESULTS_FILE` | `experiment_results.csv` | Result path relative to `results/`. |
| `REC_SYS_RUN_LABEL` | `quick` or `full` | Label stored in every result row. |
| `REC_SYS_EXPECTED_CUDA_DEVICE` | empty | Required substring in the CUDA device name. |
| `REC_SYS_REQUIRED_CUDA_ARCH` | empty | Required compiled PyTorch architecture, such as `sm_120`. |

The five-seed quick-mode result file records approximately 3.18 hours of summed
per-fit time on an eight-core, 16 GB RAM CPU environment. The full
GPU result file records 346.6 seconds of summed per-fit time on an evnvironment with eight allocated CPU cores, 64 GB RAM, and one NVIDIA RTX PRO 6000 Blackwell Server Edition GPU with compute capability 12.0, hosted by University of Colorado Research Computing (CURC).

### CURC full-data RTX PRO 6000 run

The batch workflow in
[`scripts/run_full_rtx_pro_6000.sh`](scripts/run_full_rtx_pro_6000.sh) runs the
same candidate budgets and five seeds on the full datasets. It requests one
RTX PRO 6000, verifies CUDA and `sm_120` support before training, stages a
job-specific copy under `/scratch/alpine`, and copies results and diagnostic
artifacts back to the submitted repository.

The latest run completed successfully on August 25, 2026. It produced 850 rows:
720 validation-only HPO candidate fits and 130 reportable test fits. The full
dataset bundles contained 44,300/5,537/5,538 MovieLens train/validation/test
positive edges and 1,179,052/60,084/46,329 ogbl-collab positive edges.

The notebook checkpoints completed suites to
`experiment_results.partial.csv`. A successful run replaces this with the
validated final CSV; if the job fails, the partial file is still copied back
for diagnosis and possible recovery but must not be reported as a complete run.

Use a separate GPU environment so changing the PyTorch CUDA build does not
alter an existing CPU experiment environment. On a CURC compute node:

```bash
module load uv

CPU_ENV="/projects/$USER/software/uv/envs/rec-sys-env"
GPU_ENV="/projects/$USER/software/uv/envs/rec-sys-gpu"

uv venv "$GPU_ENV" --python "$CPU_ENV/bin/python"
grep -vE '^torch==' requirements.txt > /tmp/rec-sys-requirements-no-torch.txt
uv pip install --python "$GPU_ENV/bin/python" \
  -r /tmp/rec-sys-requirements-no-torch.txt
uv pip install --python "$GPU_ENV/bin/python" \
  "torch==2.11.0" \
  --index-url https://download.pytorch.org/whl/cu130
```

Confirm the installed wheel and its bundled CUDA runtime:

```bash
"$GPU_ENV/bin/python" -c \
  "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_arch_list())"
```

On a CPU-only node, PyTorch 2.11 can report an empty architecture list because
no CUDA device is visible. The version checks should still report
`2.11.0+cu130` and CUDA `13.0`. The batch script performs the definitive
`sm_120` check after Slurm assigns an RTX PRO 6000 and fails before training if
the wheel and device are incompatible. Then, from the updated persistent
repository copy:

```bash
mkdir -p logs
sbatch scripts/run_full_rtx_pro_6000.sh
```

The script assumes the `ucb-general` account. Change its `--account` directive
if a different CURC allocation should be charged. Monitor the job with
`squeue --user="$USER"`; after completion, inspect:

```text
results/full_rtx_pro_6000/experiment_results.csv
artifacts/full_rtx_pro_6000/
logs/recsys-full-gpu-<job-id>.out
logs/recsys-full-gpu-<job-id>.err
```

GPU and CPU timings are hardware-specific and should not be pooled into a
single runtime comparison. Predictive metrics should also be labeled by run
mode and hardware, as recorded in the added result columns.

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
- The full GPU run still uses short training schedules and sampled balanced
  positive-negative mini-batches; "full" describes the dataset bundles, not
  exhaustive use of every training edge in every fit.
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
