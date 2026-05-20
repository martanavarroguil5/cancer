#!/usr/bin/env python
"""Experimentos opcionales de vistas de datos y feature engineering.

No forma parte del pipeline principal. Sirve para comparar de forma
reproducible si las variables derivadas o la inclusion de columnas de cautela
mejoran al nucleo guiado por metadatos. La sensibilidad economica se puede pedir
explicitamente, pero no entra en el default limpio.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cancer_ml.config import DEFAULT_SEED, METRICS_DIR, MODE_CONFIGS, set_global_seed  # noqa: E402
from cancer_ml.data import join_collections, load_available_csvs  # noqa: E402
from cancer_ml.evaluation import evaluate_estimator, threshold_search  # noqa: E402
from cancer_ml.features import (  # noqa: E402
    FEATURE_VIEWS,
    add_engineered_features,
    build_feature_policy,
    build_preprocessor,
    split_data,
)

CLEAN_EXPERIMENT_VIEWS = ("metadata_core", "safe_all", "engineered_selected")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compara vistas de features sin tocar el pipeline principal.")
    parser.add_argument("--mode", choices=sorted(MODE_CONFIGS), default="full")
    parser.add_argument("--seeds", type=int, nargs="+", default=[DEFAULT_SEED])
    parser.add_argument("--views", choices=FEATURE_VIEWS, nargs="+", default=list(CLEAN_EXPERIMENT_VIEWS))
    parser.add_argument(
        "--output",
        type=Path,
        default=METRICS_DIR / "feature_engineering_experiments.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frames, _ = load_available_csvs()
    raw_dataset = join_collections(frames)
    rows = []
    for seed in args.seeds:
        set_global_seed(seed)
        for view in args.views:
            modeling_dataset = add_engineered_features(raw_dataset) if view == "engineered_selected" else raw_dataset
            policy = build_feature_policy(modeling_dataset, feature_view=view)
            splits = split_data(modeling_dataset, policy, seed=seed)
            rows.extend(_run_models_for_view(splits, policy, args.mode, seed, view))

    results = pd.DataFrame(rows).sort_values(
        ["seed", "valid_f1_positive", "f1_positive", "recall_positive", "auc_pr"],
        ascending=[True, False, False, False, False],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f"Guardado {args.output} con {len(results)} filas.")
    print(
        results[
            [
                "seed",
                "feature_view",
                "model",
                "valid_f1_positive",
                "f1_positive",
                "recall_positive",
                "precision_positive",
                "auc_pr",
            ]
        ]
    )
    return 0


def _run_models_for_view(splits, policy, mode: str, seed: int, view: str) -> list[dict[str, object]]:
    mode_config = MODE_CONFIGS[mode]
    X_train = splits["X_train_inner"]
    y_train = splits["y_train_inner"]
    if mode_config.train_sample_size and mode_config.train_sample_size < len(y_train):
        X_train, _, y_train, _ = train_test_split(
            X_train,
            y_train,
            train_size=mode_config.train_sample_size,
            stratify=y_train,
            random_state=seed,
        )

    candidates = [
        (
            "HGB",
            HistGradientBoostingClassifier(
                learning_rate=0.03,
                max_iter=mode_config.hgb_iter,
                max_leaf_nodes=31,
                l2_regularization=0.0,
                class_weight="balanced",
                early_stopping=True,
                random_state=seed,
            ),
        ),
        (
            "HGBRegularized",
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=max(mode_config.hgb_iter, 700),
                max_leaf_nodes=15,
                l2_regularization=0.01,
                class_weight="balanced",
                early_stopping=True,
                random_state=seed,
            ),
        ),
        (
            "LogisticRegression",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                C=0.3,
                solver="lbfgs",
                random_state=seed,
            ),
        ),
    ]

    rows = []
    for model_name, estimator in candidates:
        pipeline = Pipeline([("preprocess", build_preprocessor(policy)), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        valid_proba = pipeline.predict_proba(splits["X_valid"])[:, 1]
        threshold, _ = threshold_search(splits["y_valid"], valid_proba, None)
        valid_pred = (valid_proba >= threshold).astype(int)
        result = evaluate_estimator(
            model_name,
            "FeatureExperiment",
            pipeline,
            splits["X_test"],
            splits["y_test"],
            threshold=threshold,
        )
        row = result.as_row()
        row.update(
            {
                "seed": seed,
                "feature_view": view,
                "valid_f1_positive": float(
                    f1_score(splits["y_valid"], valid_pred, pos_label=1, zero_division=0)
                ),
                "features_included": len(policy.included),
                "engineered_included": len(policy.engineered),
                "numeric_features": len(policy.numeric),
                "binary_features": len(policy.binary),
                "categorical_features": len(policy.categorical),
                "excluded_features": len(policy.excluded),
            }
        )
        row.pop("y_proba", None)
        row.pop("y_pred", None)
        rows.append(row)
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
