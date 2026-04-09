"""Core experiment harness for neural recommender optimization comparisons.

This module keeps the notebook thin by housing the reusable pieces (functions, classes):

- dataset loading and preprocessing
- model definitions
- training and evaluation loops
- optimizer and hyperparameter search routines

The implementations are intentionally lightweight. They are meant to support
side-by-side experimentation in a single notebook rather than exact
paper-faithful reproductions of every hybrid recommender architecture.
"""

from __future__ import annotations

import itertools
import math
import os
import random
import time
import urllib.request
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch.nn.utils import parameters_to_vector, vector_to_parameters

try:
    import psutil
except ImportError:
    psutil = None

try:
    from ogb.linkproppred import LinkPropPredDataset

    HAS_OGB = True
    OGB_IMPORT_ERROR = None
except Exception as exc:
    LinkPropPredDataset = None
    HAS_OGB = False
    OGB_IMPORT_ERROR = exc


SEED = 42
ROOT = Path.cwd()
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable experiment runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class GraphBundle:
    """Container for one recommendation dataset represented as edge prediction.

    The same structure is used for MovieLens and OGB so the training loops can
    stay generic across bipartite user-item graphs and homogeneous graphs.
    """
    name: str
    task_type: str
    num_nodes: int
    train_edges: torch.Tensor
    train_pos: torch.Tensor
    val_pos: torch.Tensor
    val_neg: torch.Tensor
    test_pos: torch.Tensor
    test_neg: torch.Tensor
    edge_lookup: set
    history: dict[int, list[int]]
    neighbor_sets: dict[int, set[int]]
    degrees: torch.Tensor
    node_features: torch.Tensor | None = None
    source_nodes: int | None = None
    target_nodes: int | None = None
    target_offset: int = 0
    metadata: dict = field(default_factory=dict)


class ResourceTracker:
    """Track elapsed time and peak memory during a training or search run."""

    def __init__(self) -> None:
        self.start_time = 0.0
        self.peak_ram_mb = 0.0
        self.peak_gpu_mb = 0.0

    def tick(self) -> None:
        if psutil is not None:
            ram = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
            self.peak_ram_mb = max(self.peak_ram_mb, ram)
        if torch.cuda.is_available():
            gpu = torch.cuda.max_memory_allocated() / (1024**2)
            self.peak_gpu_mb = max(self.peak_gpu_mb, gpu)

    def __enter__(self) -> "ResourceTracker":
        self.start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.tick()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.tick()

    def summary(self) -> dict:
        return {
            "wall_time_sec": time.perf_counter() - self.start_time,
            "peak_ram_mb": self.peak_ram_mb,
            "peak_gpu_mb": self.peak_gpu_mb,
        }


def describe_bundle(bundle: GraphBundle) -> dict:
    """Return a compact tabular summary for notebook display."""
    return {
        "dataset": bundle.name,
        "task_type": bundle.task_type,
        "num_nodes": bundle.num_nodes,
        "train_edges": int(bundle.train_edges.shape[0]),
        "val_edges": int(bundle.val_pos.shape[0]),
        "test_edges": int(bundle.test_pos.shape[0]),
        **bundle.metadata,
    }


def move_bundle_to_device(bundle: GraphBundle, device: torch.device) -> GraphBundle:
    """Move tensor fields onto the active device while keeping Python metadata shared."""
    return GraphBundle(
        name=bundle.name,
        task_type=bundle.task_type,
        num_nodes=bundle.num_nodes,
        train_edges=bundle.train_edges.to(device),
        train_pos=bundle.train_pos.to(device),
        val_pos=bundle.val_pos.to(device),
        val_neg=bundle.val_neg.to(device),
        test_pos=bundle.test_pos.to(device),
        test_neg=bundle.test_neg.to(device),
        edge_lookup=bundle.edge_lookup,
        history=bundle.history,
        neighbor_sets=bundle.neighbor_sets,
        degrees=bundle.degrees.to(device),
        node_features=None if bundle.node_features is None else bundle.node_features.to(device),
        source_nodes=bundle.source_nodes,
        target_nodes=bundle.target_nodes,
        target_offset=bundle.target_offset,
        metadata=bundle.metadata,
    )


def build_neighbor_sets(edges: np.ndarray, num_nodes: int) -> dict[int, set[int]]:
    """Build adjacency lookups used by heuristic rule features."""
    neighbors = {i: set() for i in range(num_nodes)}
    for src, dst in edges:
        src_i = int(src)
        dst_i = int(dst)
        neighbors[src_i].add(dst_i)
        neighbors[dst_i].add(src_i)
    return neighbors


def build_history(edges: np.ndarray, timestamps: np.ndarray | None = None) -> dict[int, list[int]]:
    """Store interaction histories ordered by timestamp or original row order."""
    timestamps = np.arange(len(edges)) if timestamps is None else timestamps
    history: dict[int, list[int]] = {}
    for idx in np.argsort(timestamps):
        src, dst = edges[idx]
        history.setdefault(int(src), []).append(int(dst))
    return history


def compute_degrees(edges: np.ndarray, num_nodes: int) -> torch.Tensor:
    """Precompute simple degree statistics for graph-based hybrid features."""
    degrees = np.zeros(num_nodes, dtype=np.float32)
    for src, dst in edges:
        degrees[int(src)] += 1.0
        degrees[int(dst)] += 1.0
    return torch.tensor(degrees, dtype=torch.float32)


def sample_negative_edges(bundle: GraphBundle, n: int, rng: np.random.Generator) -> torch.Tensor:
    """Sample non-observed edges for implicit-feedback training and evaluation."""
    negatives = []
    while len(negatives) < n:
        size = max(256, n - len(negatives))
        if bundle.task_type == "bipartite":
            src = rng.integers(0, bundle.source_nodes, size=size)
            dst = rng.integers(0, bundle.target_nodes, size=size) + bundle.target_offset
        else:
            src = rng.integers(0, bundle.num_nodes, size=size)
            dst = rng.integers(0, bundle.num_nodes, size=size)
        for left, right in zip(src, dst):
            if left == right:
                continue
            pair = (int(left), int(right))
            rev = (int(right), int(left))
            if pair in bundle.edge_lookup or rev in bundle.edge_lookup:
                continue
            negatives.append(pair)
            if len(negatives) == n:
                break
    return torch.tensor(negatives, dtype=torch.long)


def _download(url: str, destination: Path) -> None:
    """Download a dataset artifact once and reuse the local cache after that."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        urllib.request.urlretrieve(url, destination)


def load_movielens_100k(quick_mode: bool = True, max_rows_quick: int = 40_000) -> GraphBundle:
    """Load MovieLens 100K as an implicit user-item link prediction problem."""
    set_seed()
    root = DATA_DIR / "movielens_100k"
    archive = root / "ml-100k.zip"
    extracted = root / "ml-100k"
    if not extracted.exists():
        _download("https://files.grouplens.org/datasets/movielens/ml-100k.zip", archive)
        with zipfile.ZipFile(archive, "r") as zip_ref:
            zip_ref.extractall(root)
    ratings = pd.read_csv(
        extracted / "u.data",
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )
    # The study treats recommendation as implicit feedback so positive edges are
    # "strong" ratings and the rest are left unobserved.
    ratings = ratings.loc[ratings["rating"] >= 4].copy()
    ratings["user_id"] -= 1
    ratings["item_id"] -= 1
    ratings = ratings.sort_values("timestamp").reset_index(drop=True)
    if quick_mode:
        ratings = ratings.iloc[:max_rows_quick].copy()
    num_users = int(ratings["user_id"].max() + 1)
    num_items = int(ratings["item_id"].max() + 1)
    # Item node ids are offset so users and items live in one joint node space.
    ratings["dst"] = ratings["item_id"] + num_users
    pairs = ratings[["user_id", "dst"]].to_numpy(dtype=np.int64)
    train_end = int(len(pairs) * 0.8)
    val_end = int(len(pairs) * 0.9)
    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]
    all_edges = set(map(tuple, pairs.tolist()))
    num_nodes = num_users + num_items
    # A temporary bundle gives the negative sampler the graph metadata it needs
    # before the final train/validation/test bundle is assembled.
    tmp = GraphBundle(
        name="tmp",
        task_type="bipartite",
        num_nodes=num_nodes,
        train_edges=torch.tensor(train_pairs, dtype=torch.long),
        train_pos=torch.tensor(train_pairs, dtype=torch.long),
        val_pos=torch.tensor(val_pairs, dtype=torch.long),
        val_neg=torch.empty((0, 2), dtype=torch.long),
        test_pos=torch.tensor(test_pairs, dtype=torch.long),
        test_neg=torch.empty((0, 2), dtype=torch.long),
        edge_lookup=all_edges,
        history={},
        neighbor_sets={},
        degrees=torch.zeros(num_nodes),
        source_nodes=num_users,
        target_nodes=num_items,
        target_offset=num_users,
    )
    val_neg = sample_negative_edges(tmp, len(val_pairs), np.random.default_rng(SEED + 1))
    test_neg = sample_negative_edges(tmp, len(test_pairs), np.random.default_rng(SEED + 2))
    return GraphBundle(
        name="movielens_100k",
        task_type="bipartite",
        num_nodes=num_nodes,
        train_edges=torch.tensor(train_pairs, dtype=torch.long),
        train_pos=torch.tensor(train_pairs, dtype=torch.long),
        val_pos=torch.tensor(val_pairs, dtype=torch.long),
        val_neg=val_neg,
        test_pos=torch.tensor(test_pairs, dtype=torch.long),
        test_neg=test_neg,
        edge_lookup=all_edges,
        history=build_history(train_pairs, ratings["timestamp"].to_numpy()[:train_end]),
        neighbor_sets=build_neighbor_sets(train_pairs, num_nodes),
        degrees=compute_degrees(train_pairs, num_nodes),
        source_nodes=num_users,
        target_nodes=num_items,
        target_offset=num_users,
        metadata={"quick_mode": quick_mode, "implicit_feedback": True},
    )


def _subsample_rows(tensor: torch.Tensor, max_rows: int, seed: int) -> torch.Tensor:
    """Randomly subsample rows while keeping runs reproducible."""
    if tensor.shape[0] <= max_rows:
        return tensor
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(tensor.shape[0], generator=gen)[:max_rows]
    return tensor[idx]


def _subsample_index_array(n_rows: int, max_rows: int, seed: int) -> np.ndarray | None:
    """Return reproducible row indices when a paired NumPy array must match a tensor."""
    if n_rows <= max_rows:
        return None
    gen = torch.Generator().manual_seed(seed)
    return torch.randperm(n_rows, generator=gen)[:max_rows].cpu().numpy()


@contextmanager
def _ogb_torch_load_compat(root: Path):
    """Temporarily relax PyTorch's weights-only loading for trusted OGB cache files."""
    original_torch_load = torch.load
    ogb_root = root.resolve()

    def compat_torch_load(f, *args, **kwargs):
        path = None
        if isinstance(f, (str, os.PathLike)):
            try:
                path = Path(f).resolve()
            except OSError:
                path = None
        if path is not None:
            try:
                path.relative_to(ogb_root)
                kwargs.setdefault("weights_only", False)
            except ValueError:
                pass
        return original_torch_load(f, *args, **kwargs)

    torch.load = compat_torch_load
    try:
        yield
    finally:
        torch.load = original_torch_load


def _load_ogb_edge_split(dataset: LinkPropPredDataset, split_type: str | None = None) -> dict[str, dict[str, np.ndarray]]:
    """Load an OGB split from either a packaged dict or the per-split cache files."""
    split_name = split_type or dataset.meta_info["split"]
    split_root = Path(dataset.root) / "split" / split_name
    split_dict_path = split_root / "split_dict.pt"
    if split_dict_path.exists():
        return torch.load(split_dict_path, weights_only=False)

    # OGB's cached split files are trusted local artifacts from the downloaded dataset,
    # and PyTorch 2.6 now requires an explicit opt-out for these NumPy-backed pickles.
    return {
        "train": torch.load(split_root / "train.pt", weights_only=False),
        "valid": torch.load(split_root / "valid.pt", weights_only=False),
        "test": torch.load(split_root / "test.pt", weights_only=False),
    }


def load_ogbl_collab(quick_mode: bool = True, max_train_quick: int = 120_000, max_eval_quick: int = 20_000) -> GraphBundle:
    """Load ogbl-collab as the larger graph-learning benchmark in the suite."""
    if not HAS_OGB:
        raise ImportError(f"OGB import failed: {OGB_IMPORT_ERROR}")
    set_seed()
    ogb_root = DATA_DIR / "ogb"
    with _ogb_torch_load_compat(ogb_root):
        dataset = LinkPropPredDataset(name="ogbl-collab", root=str(ogb_root))
        graph = dataset[0]
        split = _load_ogb_edge_split(dataset)
    years = np.asarray(split["train"]["year"]) if "year" in split["train"] else None
    train_pos = torch.as_tensor(split["train"]["edge"], dtype=torch.long)
    val_pos = torch.as_tensor(split["valid"]["edge"], dtype=torch.long)
    val_neg = torch.as_tensor(split["valid"]["edge_neg"], dtype=torch.long)
    test_pos = torch.as_tensor(split["test"]["edge"], dtype=torch.long)
    test_neg = torch.as_tensor(split["test"]["edge_neg"], dtype=torch.long)
    if quick_mode:
        # Quick mode keeps notebook turnaround practical on a local machine
        # without changing the rest of the experiment code.
        train_idx = _subsample_index_array(train_pos.shape[0], max_train_quick, SEED)
        if train_idx is not None:
            train_pos = train_pos[train_idx]
            if years is not None:
                years = years[train_idx]
        train_pos = _subsample_rows(train_pos, max_train_quick, SEED)
        val_pos = _subsample_rows(val_pos, max_eval_quick, SEED + 1)
        val_neg = _subsample_rows(val_neg, max_eval_quick, SEED + 2)
        test_pos = _subsample_rows(test_pos, max_eval_quick, SEED + 3)
        test_neg = _subsample_rows(test_neg, max_eval_quick, SEED + 4)
    lookup = set(map(tuple, torch.cat([train_pos, val_pos, test_pos], dim=0).cpu().numpy().tolist()))
    return GraphBundle(
        name="ogbl_collab",
        task_type="homogeneous",
        num_nodes=int(graph["num_nodes"]),
        train_edges=train_pos,
        train_pos=train_pos,
        val_pos=val_pos,
        val_neg=val_neg,
        test_pos=test_pos,
        test_neg=test_neg,
        edge_lookup=lookup,
        history=build_history(train_pos.cpu().numpy(), years),
        neighbor_sets=build_neighbor_sets(train_pos.cpu().numpy(), int(graph["num_nodes"])),
        degrees=compute_degrees(train_pos.cpu().numpy(), int(graph["num_nodes"])),
        node_features=None if "node_feat" not in graph else torch.as_tensor(graph["node_feat"], dtype=torch.float32),
        source_nodes=int(graph["num_nodes"]),
        target_nodes=int(graph["num_nodes"]),
        target_offset=0,
        metadata={"quick_mode": quick_mode, "ogb_proxy_task": "link_prediction"},
    )


def padded_histories(bundle: GraphBundle, nodes: torch.Tensor, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Create padded interaction sequences for recurrent hybrid models."""
    seqs, lengths = [], []
    for node in nodes.detach().cpu().tolist():
        seq = bundle.history.get(int(node), [])[-max_len:]
        seq = seq if seq else [int(node)]
        seqs.append(seq)
        lengths.append(len(seq))
    max_seen = max(lengths)
    padded = torch.zeros((len(seqs), max_seen), dtype=torch.long)
    for row, seq in enumerate(seqs):
        padded[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return padded.to(nodes.device), torch.tensor(lengths, dtype=torch.long, device=nodes.device)


def build_rule_features(bundle: GraphBundle, pairs: torch.Tensor) -> torch.Tensor:
    """Compute simple graph-heuristic features as a lightweight PSL-style proxy."""
    rows = []
    degrees = bundle.degrees.detach().cpu()
    for left, right in pairs.detach().cpu().tolist():
        left_neighbors = bundle.neighbor_sets.get(int(left), set())
        right_neighbors = bundle.neighbor_sets.get(int(right), set())
        common = len(left_neighbors & right_neighbors)
        union = max(1, len(left_neighbors | right_neighbors))
        src_deg = float(degrees[int(left)].item())
        dst_deg = float(degrees[int(right)].item())
        rows.append(
            [
                math.log1p(src_deg),
                math.log1p(dst_deg),
                math.log1p(src_deg * dst_deg + 1.0),
                common / union,
                common / max(1, min(len(left_neighbors), len(right_neighbors))),
            ]
        )
    return torch.tensor(rows, dtype=torch.float32, device=pairs.device)


def score_metrics(pos_logits: torch.Tensor, neg_logits: torch.Tensor) -> dict:
    """Evaluate positive vs. negative edge scores with unified classification metrics."""
    pos_scores = torch.sigmoid(pos_logits).detach().cpu().numpy()
    neg_scores = torch.sigmoid(neg_logits).detach().cpu().numpy()
    y_true = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    y_score = np.concatenate([pos_scores, neg_scores])
    return {
        "auc": roc_auc_score(y_true, y_score),
        "average_precision": average_precision_score(y_true, y_score),
        "binary_accuracy": accuracy_score(y_true, (y_score >= 0.5).astype(np.int32)),
    }


class NodeTable(nn.Module):
    """Shared node embedding table with optional projection of raw node features.

    Example:
        >>> table = NodeTable(num_nodes=5, embedding_dim=8, feature_dim=3)
        >>> node_features = torch.randn(5, 3)
        >>> node_repr = table(node_features)
        >>> node_repr.shape
        torch.Size([5, 8])
    """

    def __init__(self, num_nodes: int, embedding_dim: int, feature_dim: int | None = None) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.feature_proj = None if not feature_dim else nn.Linear(feature_dim, embedding_dim)

    def forward(self, node_features: torch.Tensor | None = None) -> torch.Tensor:
        node_repr = self.embedding.weight
        if node_features is not None and self.feature_proj is not None:
            node_repr = node_repr + self.feature_proj(node_features)
        return node_repr


def mean_graph_aggregate(x: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """Average one-hop neighbor messages for a simple GNN encoder."""
    src, dst = edges[:, 0], edges[:, 1]
    agg = torch.zeros_like(x)
    deg = torch.zeros(x.shape[0], device=x.device)
    ones = torch.ones(src.shape[0], device=x.device)
    agg.index_add_(0, src, x[dst])
    agg.index_add_(0, dst, x[src])
    deg.index_add_(0, src, ones)
    deg.index_add_(0, dst, ones)
    return agg / deg.clamp(min=1).unsqueeze(-1)


class GraphEncoder(nn.Module):
    """Small message-passing encoder used by the graph-based recommenders."""

    def __init__(self, num_nodes: int, embedding_dim: int, layers: int, dropout: float, feature_dim: int | None = None) -> None:
        super().__init__()
        self.node_table = NodeTable(num_nodes, embedding_dim, feature_dim)
        self.layers = nn.ModuleList(
            [nn.ModuleDict({"self": nn.Linear(embedding_dim, embedding_dim), "neigh": nn.Linear(embedding_dim, embedding_dim), "drop": nn.Dropout(dropout)}) for _ in range(layers)]
        )

    def forward(self, bundle: GraphBundle) -> list[torch.Tensor]:
        x = self.node_table(bundle.node_features)
        outs = [x]
        for layer in self.layers:
            agg = mean_graph_aggregate(x, bundle.train_edges)
            x = layer["drop"](F.relu(layer["self"](x) + layer["neigh"](agg)))
            outs.append(x)
        return outs


def combine_layers(outputs: list[torch.Tensor], mode: str) -> torch.Tensor:
    """Combine layer outputs with fixed heuristics instead of learned attention."""
    if mode == "last":
        return outputs[-1]
    if mode == "mean":
        return torch.stack(outputs, dim=0).mean(dim=0)
    weights = torch.linspace(1.0, 2.0, steps=len(outputs), device=outputs[0].device)
    stacked = torch.stack(outputs, dim=0)
    return (stacked * weights.view(-1, 1, 1)).sum(dim=0) / weights.sum()


class EmbeddingMLPRecommender(nn.Module):
    """Baseline DNN recommender over paired node embeddings."""

    def __init__(self, bundle: GraphBundle, embedding_dim: int = 32, hidden_dim: int = 64, dropout: float = 0.2, lr: float = 3e-3) -> None:
        super().__init__()
        self.lr = lr
        feat_dim = None if bundle.node_features is None else bundle.node_features.shape[1]
        self.table = NodeTable(bundle.num_nodes, embedding_dim, feat_dim)
        self.scorer = nn.Sequential(nn.Linear(embedding_dim * 4, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, bundle: GraphBundle, pairs: torch.Tensor) -> torch.Tensor:
        nodes = self.table(bundle.node_features)
        src, dst = nodes[pairs[:, 0]], nodes[pairs[:, 1]]
        return self.scorer(torch.cat([src, dst, src * dst, torch.abs(src - dst)], dim=-1)).squeeze(-1)


class GNNRecommender(nn.Module):
    """Standard graph-based recommender using shallow message passing."""

    def __init__(self, bundle: GraphBundle, embedding_dim: int = 32, hidden_dim: int = 64, dropout: float = 0.2, lr: float = 3e-3, layers: int = 2, combine_mode: str = "mean") -> None:
        super().__init__()
        self.lr = lr
        feat_dim = None if bundle.node_features is None else bundle.node_features.shape[1]
        self.encoder = GraphEncoder(bundle.num_nodes, embedding_dim, layers, dropout, feat_dim)
        self.combine_mode = combine_mode
        self.scorer = nn.Sequential(nn.Linear(embedding_dim * 4, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, bundle: GraphBundle, pairs: torch.Tensor) -> torch.Tensor:
        nodes = combine_layers(self.encoder(bundle), self.combine_mode)
        src, dst = nodes[pairs[:, 0]], nodes[pairs[:, 1]]
        return self.scorer(torch.cat([src, dst, src * dst, torch.abs(src - dst)], dim=-1)).squeeze(-1)


class RHMMInspiredRecommender(nn.Module):
    """Regime-aware recurrent hybrid inspired by RHMM-style sequential structure.

    This is not a full RHMM implementation. It uses a GRU plus a learned latent
    regime mixture to approximate hidden-state switching behavior.
    """

    def __init__(self, bundle: GraphBundle, embedding_dim: int = 32, hidden_dim: int = 64, dropout: float = 0.2, lr: float = 3e-3, num_regimes: int = 4) -> None:
        super().__init__()
        self.lr = lr
        feat_dim = None if bundle.node_features is None else bundle.node_features.shape[1]
        self.table = NodeTable(bundle.num_nodes, embedding_dim, feat_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.regime_head = nn.Linear(hidden_dim, num_regimes)
        self.regime_embeddings = nn.Parameter(torch.randn(num_regimes, hidden_dim) * 0.1)
        scorer_in_dim = hidden_dim * 2 + embedding_dim
        self.scorer = nn.Sequential(nn.Linear(scorer_in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, bundle: GraphBundle, pairs: torch.Tensor) -> torch.Tensor:
        nodes = self.table(bundle.node_features)
        histories, lengths = padded_histories(bundle, pairs[:, 0], 10)
        encoded, _ = self.gru(nodes[histories])
        last = encoded[torch.arange(encoded.shape[0], device=encoded.device), lengths - 1]
        regimes = F.softmax(self.regime_head(last), dim=-1) @ self.regime_embeddings
        dst = nodes[pairs[:, 1]]
        return self.scorer(torch.cat([last, regimes, dst], dim=-1)).squeeze(-1)


class PSLDNNRecommender(nn.Module):
    """Hybrid model that augments embeddings with rule-like graph heuristics."""

    def __init__(self, bundle: GraphBundle, embedding_dim: int = 32, hidden_dim: int = 64, dropout: float = 0.2, lr: float = 3e-3) -> None:
        super().__init__()
        self.lr = lr
        feat_dim = None if bundle.node_features is None else bundle.node_features.shape[1]
        self.table = NodeTable(bundle.num_nodes, embedding_dim, feat_dim)
        self.scorer = nn.Sequential(nn.Linear(embedding_dim * 4 + 5, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, bundle: GraphBundle, pairs: torch.Tensor) -> torch.Tensor:
        nodes = self.table(bundle.node_features)
        src, dst = nodes[pairs[:, 0]], nodes[pairs[:, 1]]
        feats = torch.cat([src, dst, src * dst, torch.abs(src - dst), build_rule_features(bundle, pairs)], dim=-1)
        return self.scorer(feats).squeeze(-1)


class GNNBiLSTMRecommender(nn.Module):
    """Hybrid model that mixes graph embeddings with bidirectional sequence encoding."""

    def __init__(self, bundle: GraphBundle, embedding_dim: int = 32, hidden_dim: int = 64, dropout: float = 0.2, lr: float = 3e-3, layers: int = 2) -> None:
        super().__init__()
        self.lr = lr
        feat_dim = None if bundle.node_features is None else bundle.node_features.shape[1]
        self.encoder = GraphEncoder(bundle.num_nodes, embedding_dim, layers, dropout, feat_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.scorer = nn.Sequential(nn.Linear(embedding_dim * 2 + hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, bundle: GraphBundle, pairs: torch.Tensor) -> torch.Tensor:
        nodes = combine_layers(self.encoder(bundle), "mean")
        histories, lengths = padded_histories(bundle, pairs[:, 0], 10)
        seq, _ = self.lstm(nodes[histories])
        pooled = seq[torch.arange(seq.shape[0], device=seq.device), lengths - 1]
        src, dst = nodes[pairs[:, 0]], nodes[pairs[:, 1]]
        return self.scorer(torch.cat([src, pooled, dst], dim=-1)).squeeze(-1)


class RLBanditAggregationRecommender(nn.Module):
    """GNN with a small bandit controller that chooses one aggregation strategy."""

    def __init__(self, bundle: GraphBundle, embedding_dim: int = 32, hidden_dim: int = 64, dropout: float = 0.2, lr: float = 3e-3, layers: int = 2) -> None:
        super().__init__()
        self.lr = lr
        feat_dim = None if bundle.node_features is None else bundle.node_features.shape[1]
        self.encoder = GraphEncoder(bundle.num_nodes, embedding_dim, layers, dropout, feat_dim)
        self.experts = ["last", "mean", "residual"]
        self.policy_logits = nn.Parameter(torch.zeros(len(self.experts)))
        self.scorer = nn.Sequential(nn.Linear(embedding_dim * 4, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def sample_expert(self) -> int:
        probs = torch.softmax(self.policy_logits.detach().cpu(), dim=0).numpy()
        return int(np.random.choice(len(self.experts), p=probs))

    def forward(self, bundle: GraphBundle, pairs: torch.Tensor, expert_idx: int | None = None) -> torch.Tensor:
        expert_idx = int(torch.argmax(self.policy_logits).item()) if expert_idx is None else expert_idx
        nodes = combine_layers(self.encoder(bundle), self.experts[expert_idx])
        src, dst = nodes[pairs[:, 0]], nodes[pairs[:, 1]]
        return self.scorer(torch.cat([src, dst, src * dst, torch.abs(src - dst)], dim=-1)).squeeze(-1)


MODEL_REGISTRY = {
    "dnn": EmbeddingMLPRecommender,
    "gnn": GNNRecommender,
    "rhmm": RHMMInspiredRecommender,
    "psl_dnn": PSLDNNRecommender,
    "gnn_bilstm": GNNBiLSTMRecommender,
    "rl_gnn": RLBanditAggregationRecommender,
}

# Shared experiment knobs keep the notebook readable and make quick/full runs
# easy to control from one place.
DEFAULT_CONFIG = {
    "history_len": 10,
    "batch_size": 512,
    "neg_ratio": 1,
    "optimizer_epochs_small": 5,
    "optimizer_epochs_large": 3,
    "optimizer_steps_small": 35,
    "optimizer_steps_large": 20,
    "family_epochs_small": 4,
    "family_epochs_large": 3,
    "family_steps_small": 30,
    "family_steps_large": 16,
    "hpo_epochs_small": 3,
    "hpo_epochs_large": 2,
    "hpo_steps_small": 20,
    "hpo_steps_large": 12,
}

HPO_SPACE = {
    "embedding_dim": [16, 32, 48],
    "hidden_dim": [32, 64, 96],
    "dropout": [0.0, 0.1, 0.2, 0.3],
    "lr": [1e-3, 3e-3, 1e-2],
}


def make_train_batch(bundle: GraphBundle, batch_size: int, neg_ratio: int, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a balanced mini-batch of observed and synthetic negative edges."""
    pos_idx = rng.integers(0, bundle.train_pos.shape[0], size=batch_size)
    pos_pairs = bundle.train_pos[pos_idx]
    neg_pairs = sample_negative_edges(bundle, batch_size * neg_ratio, rng)
    pairs = torch.cat([pos_pairs, neg_pairs], dim=0)
    labels = torch.cat([torch.ones(pos_pairs.shape[0]), torch.zeros(neg_pairs.shape[0])], dim=0).float()
    return pairs, labels


@torch.no_grad()
def evaluate_model(model: nn.Module, bundle_cpu: GraphBundle, bundle_device: GraphBundle, pos_pairs: torch.Tensor, neg_pairs: torch.Tensor, expert_idx: int | None = None) -> dict:
    """Score a model on held-out positive and negative edges in evaluation-sized batches."""
    model.eval()
    pos_logits, neg_logits = [], []
    for start in range(0, pos_pairs.shape[0], 2048):
        batch = pos_pairs[start : start + 2048].to(bundle_device.train_edges.device)
        pos_logits.append(model(bundle_device, batch, expert_idx=expert_idx) if isinstance(model, RLBanditAggregationRecommender) else model(bundle_device, batch))
    for start in range(0, neg_pairs.shape[0], 2048):
        batch = neg_pairs[start : start + 2048].to(bundle_device.train_edges.device)
        neg_logits.append(model(bundle_device, batch, expert_idx=expert_idx) if isinstance(model, RLBanditAggregationRecommender) else model(bundle_device, batch))
    return score_metrics(torch.cat(pos_logits), torch.cat(neg_logits))


def _train_gradient(bundle_cpu: GraphBundle, model_name: str, hparams: dict, optimizer_name: str, epochs: int, steps: int, suite: str, device: torch.device) -> dict:
    """Train one model with a gradient optimizer and return a flat result row."""
    rng = np.random.default_rng(SEED)
    bundle_device = move_bundle_to_device(bundle_cpu, device)
    model = MODEL_REGISTRY[model_name](bundle_device, **hparams).to(device)
    lr = hparams.get("lr", getattr(model, "lr", 3e-3))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) if optimizer_name == "adam" else torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    losses = []
    with ResourceTracker() as tracker:
        for _ in range(epochs):
            model.train()
            for _ in range(steps):
                pairs, labels = make_train_batch(bundle_cpu, DEFAULT_CONFIG["batch_size"], DEFAULT_CONFIG["neg_ratio"], rng)
                pairs, labels = pairs.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(bundle_device, pairs)
                loss = F.binary_cross_entropy_with_logits(logits, labels)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
                tracker.tick()
        val_metrics = evaluate_model(model, bundle_cpu, bundle_device, bundle_cpu.val_pos, bundle_cpu.val_neg)
        test_metrics = evaluate_model(model, bundle_cpu, bundle_device, bundle_cpu.test_pos, bundle_cpu.test_neg)
        tracker.tick()
        summary = tracker.summary()
    return {
        "suite": suite,
        "dataset": bundle_cpu.name,
        "model": model_name,
        "method": optimizer_name,
        "val_auc": val_metrics["auc"],
        "val_ap": val_metrics["average_precision"],
        "val_accuracy": val_metrics["binary_accuracy"],
        "test_auc": test_metrics["auc"],
        "test_ap": test_metrics["average_precision"],
        "test_accuracy": test_metrics["binary_accuracy"],
        "loss_tail": float(np.mean(losses[-10:])) if losses else float("nan"),
        **summary,
        "notes": "",
    }


def train_population_recommender(bundle_cpu: GraphBundle, method: str, device: torch.device) -> dict:
    """Optimize the same DNN's weights directly with PSO or evolution strategies."""
    bundle_device = move_bundle_to_device(bundle_cpu, device)
    model = EmbeddingMLPRecommender(bundle_device, embedding_dim=16, hidden_dim=32, dropout=0.1, lr=3e-3).to(device)
    pairs, labels = make_train_batch(bundle_cpu, 256, 1, np.random.default_rng(SEED))
    pairs, labels = pairs.to(device), labels.to(device)
    base = parameters_to_vector(model.parameters()).detach()
    dim = base.numel()

    def objective(vec: np.ndarray) -> float:
        # Population-based methods operate on flattened parameter vectors rather
        # than gradients, so we temporarily write candidate weights into the model.
        vector_to_parameters(torch.tensor(vec, dtype=base.dtype, device=device), model.parameters())
        with torch.no_grad():
            return float(F.binary_cross_entropy_with_logits(model(bundle_device, pairs), labels).item())

    with ResourceTracker() as tracker:
        if method == "pso":
            # Particle swarm tracks personal and global best candidates.
            pop, iters = 10, 20 if bundle_cpu.name == "movielens_100k" else 12
            rng = np.random.default_rng(SEED)
            pos = np.tile(base.cpu().numpy(), (pop, 1)) + 0.05 * rng.standard_normal((pop, dim))
            vel = np.zeros_like(pos)
            pbest = pos.copy()
            pscores = np.array([objective(x) for x in pos])
            gidx = int(np.argmin(pscores))
            gbest, gscore = pbest[gidx].copy(), float(pscores[gidx])
            for _ in range(iters):
                r1, r2 = rng.random((pop, dim)), rng.random((pop, dim))
                vel = 0.7 * vel + 1.4 * r1 * (pbest - pos) + 1.4 * r2 * (gbest - pos)
                pos = pos + vel
                scores = np.array([objective(x) for x in pos])
                mask = scores < pscores
                pbest[mask], pscores[mask] = pos[mask], scores[mask]
                gidx = int(np.argmin(pscores))
                if pscores[gidx] < gscore:
                    gbest, gscore = pbest[gidx].copy(), float(pscores[gidx])
                tracker.tick()
            best = gbest
        else:
            # This is a compact evolution-strategy loop that nudges the search
            # center toward better random perturbations.
            sigma, es_lr, pop, iters = 0.05, 0.03, 18, 18 if bundle_cpu.name == "movielens_100k" else 10
            rng = np.random.default_rng(SEED)
            center = base.cpu().numpy().copy()
            best, best_score = center.copy(), objective(center)
            for _ in range(iters):
                noise = rng.standard_normal((pop, dim))
                candidates = center + sigma * noise
                rewards = -np.array([objective(x) for x in candidates])
                norm = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
                center = center + (es_lr / (pop * sigma)) * (noise.T @ norm)
                score = objective(center)
                if score < best_score:
                    best, best_score = center.copy(), score
                tracker.tick()
        vector_to_parameters(torch.tensor(best, dtype=base.dtype, device=device), model.parameters())
        val_metrics = evaluate_model(model, bundle_cpu, bundle_device, bundle_cpu.val_pos, bundle_cpu.val_neg)
        test_metrics = evaluate_model(model, bundle_cpu, bundle_device, bundle_cpu.test_pos, bundle_cpu.test_neg)
        tracker.tick()
        summary = tracker.summary()
    return {
        "suite": "optimizer_training",
        "dataset": bundle_cpu.name,
        "model": "dnn",
        "method": method,
        "val_auc": val_metrics["auc"],
        "val_ap": val_metrics["average_precision"],
        "val_accuracy": val_metrics["binary_accuracy"],
        "test_auc": test_metrics["auc"],
        "test_ap": test_metrics["average_precision"],
        "test_accuracy": test_metrics["binary_accuracy"],
        "loss_tail": float("nan"),
        **summary,
        "notes": "Population-based training on the same neural recommender.",
    }


def train_rl_family_model(bundle_cpu: GraphBundle, hparams: dict, epochs: int, steps: int, device: torch.device) -> dict:
    """Train the RL-style hybrid by alternating supervised learning and bandit updates."""
    rng = np.random.default_rng(SEED)
    bundle_device = move_bundle_to_device(bundle_cpu, device)
    model = RLBanditAggregationRecommender(bundle_device, **hparams).to(device)
    optimizer = torch.optim.Adam([p for name, p in model.named_parameters() if name != "policy_logits"], lr=hparams.get("lr", 3e-3))
    baseline = 0.0
    with ResourceTracker() as tracker:
        for _ in range(epochs):
            expert_idx = model.sample_expert()
            for _ in range(steps):
                pairs, labels = make_train_batch(bundle_cpu, DEFAULT_CONFIG["batch_size"], DEFAULT_CONFIG["neg_ratio"], rng)
                pairs, labels = pairs.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = F.binary_cross_entropy_with_logits(model(bundle_device, pairs, expert_idx=expert_idx), labels)
                loss.backward()
                optimizer.step()
                tracker.tick()
            # Validation AUC serves as the reward signal for the aggregation policy.
            reward = evaluate_model(model, bundle_cpu, bundle_device, bundle_cpu.val_pos, bundle_cpu.val_neg, expert_idx=expert_idx)["auc"]
            baseline = 0.9 * baseline + 0.1 * reward
            with torch.no_grad():
                model.policy_logits[expert_idx] += 0.2 * (reward - baseline)
        best_idx = int(torch.argmax(model.policy_logits).item())
        val_metrics = evaluate_model(model, bundle_cpu, bundle_device, bundle_cpu.val_pos, bundle_cpu.val_neg, expert_idx=best_idx)
        test_metrics = evaluate_model(model, bundle_cpu, bundle_device, bundle_cpu.test_pos, bundle_cpu.test_neg, expert_idx=best_idx)
        tracker.tick()
        summary = tracker.summary()
    return {
        "suite": "model_family",
        "dataset": bundle_cpu.name,
        "model": "rl_gnn",
        "method": "bandit_policy",
        "val_auc": val_metrics["auc"],
        "val_ap": val_metrics["average_precision"],
        "val_accuracy": val_metrics["binary_accuracy"],
        "test_auc": test_metrics["auc"],
        "test_ap": test_metrics["average_precision"],
        "test_accuracy": test_metrics["binary_accuracy"],
        "loss_tail": float("nan"),
        **summary,
        "notes": f"selected_expert={model.experts[best_idx]}",
    }


def evaluate_hparams(bundle_cpu: GraphBundle, model_name: str, hparams: dict, device: torch.device) -> dict:
    """Train a short run for one hyperparameter configuration."""
    epochs = DEFAULT_CONFIG["hpo_epochs_small"] if bundle_cpu.name == "movielens_100k" else DEFAULT_CONFIG["hpo_epochs_large"]
    steps = DEFAULT_CONFIG["hpo_steps_small"] if bundle_cpu.name == "movielens_100k" else DEFAULT_CONFIG["hpo_steps_large"]
    return _train_gradient(bundle_cpu, model_name, hparams, "adam", epochs, steps, "hyperparameter_search", device)


def run_grid_search(bundle_cpu: GraphBundle, model_name: str, device: torch.device) -> list[dict]:
    """Enumerate a small structured grid of hyperparameters."""
    rows = []
    grid = itertools.product(HPO_SPACE["embedding_dim"][:2], HPO_SPACE["hidden_dim"][:2], HPO_SPACE["dropout"][:3], HPO_SPACE["lr"][:2])
    for emb, hid, drop, lr in grid:
        row = evaluate_hparams(bundle_cpu, model_name, {"embedding_dim": emb, "hidden_dim": hid, "dropout": drop, "lr": lr}, device)
        row["method"], row["notes"] = "grid_search", f"embedding_dim={emb}, hidden_dim={hid}, dropout={drop}, lr={lr}"
        rows.append(row)
    return rows


def run_random_search(bundle_cpu: GraphBundle, model_name: str, device: torch.device, trials: int = 8) -> list[dict]:
    """Sample hyperparameters independently from the predefined search space."""
    rows, rng = [], np.random.default_rng(SEED)
    for _ in range(trials):
        candidate = {k: (float(rng.choice(v)) if k in {"dropout", "lr"} else int(rng.choice(v))) for k, v in HPO_SPACE.items()}
        row = evaluate_hparams(bundle_cpu, model_name, candidate, device)
        row["method"], row["notes"] = "random_search", ", ".join(f"{k}={v}" for k, v in candidate.items())
        rows.append(row)
    return rows


def decode_de_vector(vector: np.ndarray) -> dict:
    """Map a continuous DE candidate vector into valid discrete model hyperparameters."""
    return {
        "embedding_dim": int(np.clip(np.round(vector[0] / 16) * 16, 16, 48)),
        "hidden_dim": int(np.clip(np.round(vector[1] / 32) * 32, 32, 96)),
        "dropout": float(np.clip(vector[2], 0.0, 0.35)),
        "lr": float(10 ** np.clip(vector[3], -3.2, -2.0)),
    }


def run_de_search(bundle_cpu: GraphBundle, model_name: str, device: torch.device) -> list[dict]:
    """Run a differential-evolution-style population search over hyperparameters."""
    rng, rows = np.random.default_rng(SEED), []
    pop, gens = 6, 4 if bundle_cpu.name == "movielens_100k" else 3
    bounds = np.array([[16, 48], [32, 96], [0.0, 0.35], [-3.2, -2.0]], dtype=float)
    population = np.column_stack([rng.uniform(bounds[i, 0], bounds[i, 1], size=pop) for i in range(bounds.shape[0])])
    scores = []
    for cand in population:
        row = evaluate_hparams(bundle_cpu, model_name, decode_de_vector(cand), device)
        rows.append(row)
        scores.append(row["val_auc"])
    for _ in range(gens):
        for idx in range(pop):
            pool = [i for i in range(pop) if i != idx]
            a, b, c = rng.choice(pool, size=3, replace=False)
            mutant = population[a] + 0.7 * (population[b] - population[c])
            mask = rng.random(4) < 0.7
            if not mask.any():
                mask[rng.integers(0, 4)] = True
            trial = np.clip(np.where(mask, mutant, population[idx]), bounds[:, 0], bounds[:, 1])
            row = evaluate_hparams(bundle_cpu, model_name, decode_de_vector(trial), device)
            if row["val_auc"] >= scores[idx]:
                population[idx], scores[idx] = trial, row["val_auc"]
                rows.append(row)
    for row in rows:
        row["method"] = "de_opt"
        row["notes"] = row["notes"] or "Differential-evolution-style search."
    return rows


def run_optimizer_suite(bundle_cpu: GraphBundle, device: torch.device) -> list[dict]:
    """Compare gradient and population-based training on the same baseline model."""
    epochs = DEFAULT_CONFIG["optimizer_epochs_small"] if bundle_cpu.name == "movielens_100k" else DEFAULT_CONFIG["optimizer_epochs_large"]
    steps = DEFAULT_CONFIG["optimizer_steps_small"] if bundle_cpu.name == "movielens_100k" else DEFAULT_CONFIG["optimizer_steps_large"]
    hparams = {"embedding_dim": 32, "hidden_dim": 64, "dropout": 0.2, "lr": 3e-3}
    return [
        _train_gradient(bundle_cpu, "dnn", hparams, "adam", epochs, steps, "optimizer_training", device),
        _train_gradient(bundle_cpu, "dnn", hparams, "sgd", epochs, steps, "optimizer_training", device),
        train_population_recommender(bundle_cpu, "pso", device),
        train_population_recommender(bundle_cpu, "evolutionary", device),
    ]


def run_hpo_suite(bundle_cpu: GraphBundle, device: torch.device) -> list[dict]:
    """Compare classical and population-based hyperparameter search strategies."""
    model_name = "dnn" if bundle_cpu.task_type == "bipartite" else "gnn"
    return run_grid_search(bundle_cpu, model_name, device) + run_random_search(bundle_cpu, model_name, device) + run_de_search(bundle_cpu, model_name, device)


def run_model_family_suite(bundle_cpu: GraphBundle, device: torch.device) -> list[dict]:
    """Train the baseline and hybrid model families under one shared protocol."""
    epochs = DEFAULT_CONFIG["family_epochs_small"] if bundle_cpu.name == "movielens_100k" else DEFAULT_CONFIG["family_epochs_large"]
    steps = DEFAULT_CONFIG["family_steps_small"] if bundle_cpu.name == "movielens_100k" else DEFAULT_CONFIG["family_steps_large"]
    hparams = {"embedding_dim": 32, "hidden_dim": 64, "dropout": 0.2, "lr": 3e-3}
    rows = [
        _train_gradient(bundle_cpu, "dnn", hparams, "adam", epochs, steps, "model_family", device),
        _train_gradient(bundle_cpu, "gnn", hparams, "adam", epochs, steps, "model_family", device),
        _train_gradient(bundle_cpu, "rhmm", hparams, "adam", epochs, steps, "model_family", device),
        _train_gradient(bundle_cpu, "psl_dnn", hparams, "adam", epochs, steps, "model_family", device),
        _train_gradient(bundle_cpu, "gnn_bilstm", hparams, "adam", epochs, steps, "model_family", device),
        train_rl_family_model(bundle_cpu, hparams, epochs, steps, device),
    ]
    return rows


def load_dataset_bundles(quick_mode: bool = True) -> dict[str, GraphBundle]:
    """Load every dataset requested for the notebook, skipping OGB when unavailable."""
    bundles = {"movielens_100k": load_movielens_100k(quick_mode=quick_mode)}
    if HAS_OGB:
        bundles["ogbl_collab"] = load_ogbl_collab(quick_mode=quick_mode)
    return bundles
