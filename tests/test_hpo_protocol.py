"""Regression tests for validation-only hyperparameter selection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import experiment_suite as suite


class HyperparameterProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = SimpleNamespace(name="movielens_100k", task_type="bipartite")

    @patch.object(suite, "_train_gradient")
    def test_candidate_evaluation_does_not_evaluate_test_set(self, train_mock) -> None:
        train_mock.return_value = {"val_auc": 0.7}

        row = suite.evaluate_hparams(self.bundle, "dnn", {"lr": 0.001}, "cpu", seed=123)

        self.assertFalse(train_mock.call_args.kwargs["evaluate_test_set"])
        self.assertEqual(train_mock.call_args.kwargs["seed"], 123)
        self.assertEqual(row["selection_stage"], "validation_search")

    @patch.object(suite, "_train_gradient")
    def test_selected_configuration_is_the_only_final_test_run(self, train_mock) -> None:
        train_mock.return_value = {"val_auc": 0.7}

        row = suite.evaluate_selected_hparams(
            self.bundle,
            "dnn",
            {"lr": 0.001},
            "grid_search",
            "cpu",
            seed=456,
        )

        self.assertTrue(train_mock.call_args.kwargs["evaluate_test_set"])
        self.assertEqual(train_mock.call_args.kwargs["seed"], 456)
        self.assertEqual(row["selection_stage"], "final_test")
        self.assertEqual(row["method"], "grid_search")

    def test_grid_search_selects_by_validation_auc(self) -> None:
        def validation_row(bundle, model, hparams, device, seed):
            validation_scores = {
                16: {1: 0.9, 2: 0.1},
                32: {1: 0.4, 2: 0.4},
            }
            return {
                "val_auc": validation_scores[hparams["embedding_dim"]][seed],
                "test_auc": float("nan"),
                "evaluated_on_test": False,
                "seed": seed,
                "method": "",
                "notes": "",
            }

        def final_row(bundle, model, hparams, method, device, seed):
            return {
                "val_auc": float(hparams["embedding_dim"]),
                "test_auc": 0.5,
                "evaluated_on_test": True,
                "seed": seed,
                "selection_stage": "final_test",
                "method": method,
                "notes": "",
                "hparams": hparams,
            }

        with (
            patch.object(suite, "evaluate_hparams", side_effect=validation_row),
            patch.object(suite, "evaluate_selected_hparams", side_effect=final_row),
        ):
            rows = suite.run_grid_search(self.bundle, "dnn", "cpu", seeds=(1, 2))

        final_rows = [row for row in rows if row["evaluated_on_test"]]
        self.assertEqual(len(final_rows), 2)
        self.assertEqual({row["seed"] for row in final_rows}, {1, 2})
        # The 16-dimensional candidate wins on mean validation AUC (0.5 vs.
        # 0.4), even though the 32-dimensional candidate wins for seed 2.
        self.assertEqual(final_rows[0]["hparams"]["embedding_dim"], 16)
        candidate_rows = [row for row in rows if not row["evaluated_on_test"]]
        self.assertTrue(all(row["test_auc"] != row["test_auc"] for row in candidate_rows))

    def test_project_paths_do_not_depend_on_working_directory(self) -> None:
        expected_root = Path(suite.__file__).resolve().parents[1]
        self.assertEqual(suite.PROJECT_ROOT, expected_root)
        self.assertEqual(suite.DATA_DIR, expected_root / "data")
        self.assertEqual(suite.RESULTS_DIR, expected_root / "results")

    def test_optimizer_comparison_uses_every_shared_seed(self) -> None:
        def gradient_row(*args, seed, **kwargs):
            return {"method": args[3], "seed": seed}

        def population_row(bundle, model, method, device, seed):
            return {"method": method, "seed": seed}

        with (
            patch.object(suite, "_train_gradient", side_effect=gradient_row),
            patch.object(suite, "train_population_recommender", side_effect=population_row),
        ):
            rows = suite.run_optimizer_suite(self.bundle, "cpu", seeds=(11, 22))

        self.assertEqual(len(rows), 8)
        for method in ("adam", "sgd", "pso", "evolutionary"):
            method_seeds = {row["seed"] for row in rows if row["method"] == method}
            self.assertEqual(method_seeds, {11, 22})


if __name__ == "__main__":
    unittest.main()
