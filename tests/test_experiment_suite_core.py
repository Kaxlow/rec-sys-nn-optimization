"""Fast, download-free tests for core experiment utilities."""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import experiment_suite as suite


class ExperimentSuiteCoreTests(unittest.TestCase):
    def test_set_seed_repeats_python_numpy_and_torch_draws(self) -> None:
        suite.set_seed(314)
        first = (random.random(), np.random.random(), torch.rand(3))

        suite.set_seed(314)
        second = (random.random(), np.random.random(), torch.rand(3))

        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        self.assertTrue(torch.equal(first[2], second[2]))

    def test_graph_helpers_build_expected_training_structure(self) -> None:
        edges = np.array([[0, 2], [0, 1], [1, 3]], dtype=np.int64)
        timestamps = np.array([20, 10, 15])

        history = suite.build_history(edges, timestamps)
        neighbors = suite.build_neighbor_sets(edges, num_nodes=4)
        degrees = suite.compute_degrees(edges, num_nodes=4)

        self.assertEqual(history, {0: [1, 2], 1: [3]})
        self.assertEqual(neighbors[0], {1, 2})
        self.assertEqual(neighbors[1], {0, 3})
        self.assertTrue(torch.equal(degrees, torch.tensor([2.0, 2.0, 1.0, 1.0])))

    def test_negative_sampler_respects_bipartite_ranges_and_observed_edges(self) -> None:
        observed = {(0, 2), (1, 3)}
        bundle = suite.GraphBundle(
            name="synthetic",
            task_type="bipartite",
            num_nodes=4,
            train_edges=torch.tensor([[0, 2], [1, 3]]),
            train_pos=torch.tensor([[0, 2], [1, 3]]),
            val_pos=torch.empty((0, 2), dtype=torch.long),
            val_neg=torch.empty((0, 2), dtype=torch.long),
            test_pos=torch.empty((0, 2), dtype=torch.long),
            test_neg=torch.empty((0, 2), dtype=torch.long),
            edge_lookup=observed,
            history={},
            neighbor_sets={},
            degrees=torch.zeros(4),
            source_nodes=2,
            target_nodes=2,
            target_offset=2,
        )

        negatives = suite.sample_negative_edges(bundle, 20, np.random.default_rng(7))

        self.assertEqual(tuple(negatives.shape), (20, 2))
        for source, target in negatives.tolist():
            self.assertIn(source, range(0, 2))
            self.assertIn(target, range(2, 4))
            self.assertNotIn((source, target), observed)

    def test_metrics_are_perfect_for_separated_logits(self) -> None:
        metrics = suite.score_metrics(torch.tensor([4.0, 3.0]), torch.tensor([-3.0, -4.0]))

        self.assertAlmostEqual(metrics["auc"], 1.0)
        self.assertAlmostEqual(metrics["average_precision"], 1.0)
        self.assertAlmostEqual(metrics["binary_accuracy"], 1.0)

    def test_de_decoder_stays_within_search_bounds(self) -> None:
        decoded = suite.decode_de_vector(np.array([-100.0, 1000.0, 2.0, -10.0]))

        self.assertEqual(decoded["embedding_dim"], 16)
        self.assertEqual(decoded["hidden_dim"], 96)
        self.assertEqual(decoded["dropout"], 0.35)
        self.assertAlmostEqual(decoded["lr"], 10**-3.2)


if __name__ == "__main__":
    unittest.main()
