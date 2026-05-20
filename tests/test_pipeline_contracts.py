"""Pruebas de contrato del pipeline F1 limpio."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cancer_ml.config import (  # noqa: E402
    KNOWN_HIGH_RISK_POST_DIAGNOSIS_COLUMNS,
    KNOWN_SUSPECT_LEAKAGE_COLUMNS,
    RECOMMENDED_FEATURE_VIEW,
    TARGET_COLUMN,
)
from cancer_ml.data import join_collections, load_available_csvs  # noqa: E402
from cancer_ml.evaluation import threshold_search  # noqa: E402
from cancer_ml.features import build_feature_policy, split_data  # noqa: E402


class PipelineContractsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        frames, _ = load_available_csvs()
        cls.dataset = join_collections(frames)

    def test_threshold_search_maximizes_positive_f1(self) -> None:
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.05, 0.20, 0.55, 0.40, 0.60, 0.95])
        threshold, table = threshold_search(y_true, y_proba)
        expected = table.assign(
            recomputed_f1=lambda df: [
                f1_score(y_true, (y_proba >= value).astype(int), pos_label=1, zero_division=0)
                for value in df["threshold"]
            ]
        ).sort_values(["recomputed_f1", "precision_positive", "recall_positive"], ascending=False).iloc[0]
        self.assertEqual(threshold, expected["threshold"])

    def test_metadata_core_excludes_leakage_and_target(self) -> None:
        policy = build_feature_policy(self.dataset, feature_view="metadata_core")
        forbidden = set(KNOWN_SUSPECT_LEAKAGE_COLUMNS + KNOWN_HIGH_RISK_POST_DIAGNOSIS_COLUMNS + [TARGET_COLUMN])
        self.assertFalse(forbidden.intersection(policy.included))

    def test_recommended_view_is_safe_all_and_excludes_leakage(self) -> None:
        policy = build_feature_policy(self.dataset)
        forbidden = set(KNOWN_SUSPECT_LEAKAGE_COLUMNS + KNOWN_HIGH_RISK_POST_DIAGNOSIS_COLUMNS + [TARGET_COLUMN])
        self.assertEqual(RECOMMENDED_FEATURE_VIEW, "safe_all")
        self.assertEqual(policy.feature_view, "safe_all")
        self.assertFalse(forbidden.intersection(policy.included))
        self.assertIn("tipo_seguro", policy.included)
        self.assertIn("edad", policy.included)

    def test_split_is_stratified(self) -> None:
        policy = build_feature_policy(self.dataset, feature_view="metadata_core")
        splits = split_data(self.dataset, policy, seed=42)
        global_rate = float(splits["y"].mean())
        for key in ["y_train", "y_test", "y_train_inner", "y_valid"]:
            self.assertLess(abs(float(splits[key].mean()) - global_rate), 0.005)

    def test_leakage_view_requires_explicit_flag(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_pipeline.py"),
                "--mode",
                "quick",
                "--feature-view",
                "economic_sensitivity",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("economic_sensitivity incluye variables", result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
