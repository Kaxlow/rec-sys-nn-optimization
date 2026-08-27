"""Tests for environment-driven notebook run configuration."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import experiment_suite as suite


class RunConfigurationTests(unittest.TestCase):
    def test_boolean_environment_values_are_strict(self) -> None:
        with patch.dict(os.environ, {"REC_SYS_TEST_FLAG": "false"}):
            self.assertFalse(suite.read_bool_env("REC_SYS_TEST_FLAG", default=True))
        with patch.dict(os.environ, {"REC_SYS_TEST_FLAG": "YES"}):
            self.assertTrue(suite.read_bool_env("REC_SYS_TEST_FLAG", default=False))
        with patch.dict(os.environ, {"REC_SYS_TEST_FLAG": "sometimes"}):
            with self.assertRaises(ValueError):
                suite.read_bool_env("REC_SYS_TEST_FLAG", default=False)

    def test_boolean_environment_uses_default_when_unset(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(suite.read_bool_env("REC_SYS_TEST_FLAG", default=True))

    def test_result_path_stays_below_results_directory(self) -> None:
        path = suite.resolve_results_path("full_rtx_pro_6000/experiment_results.csv")
        self.assertEqual(path, suite.RESULTS_DIR / "full_rtx_pro_6000" / "experiment_results.csv")
        with self.assertRaises(ValueError):
            suite.resolve_results_path("../experiment_results.csv")
        with self.assertRaises(ValueError):
            suite.resolve_results_path(Path(tempfile.gettempdir()) / "experiment_results.csv")

    def test_notebook_defaults_to_quick_mode_but_supports_batch_overrides(self) -> None:
        notebook_path = Path(__file__).resolve().parents[1] / "src" / "recommender_optimization_experiments.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertIn('read_bool_env("REC_SYS_QUICK_MODE", default=True)', source)
        self.assertIn('read_bool_env("REC_SYS_REQUIRE_CUDA", default=False)', source)
        self.assertIn("REC_SYS_REQUIRED_CUDA_ARCH", source)
        self.assertIn("RESULTS_PARTIAL_PATH", source)

    def test_gpu_batch_job_requires_full_mode_rtx_pro_6000(self) -> None:
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_full_rtx_pro_6000.sh"
        script = script_path.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --partition=artxpro6000", script)
        self.assertIn("#SBATCH --gres=gpu:rtx_pro_6000:1", script)
        self.assertIn("export REC_SYS_QUICK_MODE=false", script)
        self.assertIn('export REC_SYS_REQUIRED_CUDA_ARCH="sm_120"', script)
        self.assertIn('export REC_SYS_RESULTS_FILE="${RESULT_SUBDIR}/experiment_results.csv"', script)


if __name__ == "__main__":
    unittest.main()
