"""Entrenamiento de modelos clasicos de machine learning."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from cancer_ml.config import MODELS_DIR, ModeConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class FittedModel:
    name: str
    estimator: Pipeline
    model_type: str = "ML"
    notes: str = ""


def train_classical_models(
    preprocessor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    mode_config: ModeConfig,
    seed: int,
    models_dir: Path = MODELS_DIR,
) -> list[FittedModel]:
    """Entrena modelos ML complejos y un baseline de respaldo."""

    models_dir.mkdir(parents=True, exist_ok=True)
    class_ratio = _negative_positive_ratio(y_train)

    candidates = [
        (
            "RandomForest",
            "ML",
            RandomForestClassifier(
                n_estimators=mode_config.rf_estimators,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=seed,
            ),
        ),
        (
            "ExtraTrees",
            "ML",
            ExtraTreesClassifier(
                n_estimators=mode_config.et_estimators,
                max_depth=None,
                min_samples_leaf=4,
                class_weight="balanced",
                n_jobs=-1,
                random_state=seed,
            ),
        ),
        (
            "HistGradientBoosting",
            "ML",
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
            "HistGradientBoostingRegularized",
            "ML",
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
            "GradientBoosting",
            "ML",
            GradientBoostingClassifier(
                n_estimators=max(mode_config.hgb_iter, 120),
                learning_rate=0.03,
                max_depth=2,
                subsample=0.9,
                random_state=seed,
            ),
        ),
        (
            "LogisticRegression_baseline",
            "Baseline",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                C=0.3,
                solver="lbfgs",
                random_state=seed,
            ),
        ),
    ]

    fitted: list[FittedModel] = []
    for name, model_type, estimator in candidates:
        pipeline = Pipeline([("preprocess", clone(preprocessor)), ("model", estimator)])
        LOGGER.info("Entrenando %s.", name)
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, models_dir / f"{name}.joblib")
        fitted.append(FittedModel(name=name, estimator=pipeline, model_type=model_type))

    fitted.extend(_train_xgboost_models(preprocessor, X_train, y_train, mode_config, seed, class_ratio, models_dir))

    return fitted


def _train_xgboost_models(
    preprocessor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    mode_config: ModeConfig,
    seed: int,
    class_ratio: float,
    models_dir: Path,
) -> FittedModel | None:
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        LOGGER.warning("XGBoost no disponible: %s", exc)
        return []

    common_params = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=class_ratio,
        random_state=seed,
        n_jobs=-1,
    )

    variants = [
        (
            "XGBoostF1Balanced",
            dict(
                n_estimators=max(mode_config.xgb_estimators, 1200),
                max_depth=1,
                learning_rate=0.07,
                subsample=0.95,
                colsample_bytree=0.95,
                min_child_weight=1,
                reg_lambda=1.5,
                reg_alpha=0.0,
                gamma=0.0,
            ),
        ),
        (
            "XGBoost",
            dict(
                n_estimators=mode_config.xgb_estimators,
                max_depth=2,
                learning_rate=0.04,
                subsample=0.95,
                colsample_bytree=0.95,
                min_child_weight=1,
                reg_lambda=1.0,
                reg_alpha=0.0,
            ),
        ),
        (
            "XGBoostAUC",
            dict(
                n_estimators=max(mode_config.xgb_estimators, 600),
                max_depth=1,
                learning_rate=0.075,
                subsample=0.95,
                colsample_bytree=1.0,
                min_child_weight=5,
                reg_lambda=1.0,
                reg_alpha=0.0,
                gamma=0.1,
            ),
        ),
    ]

    fitted: list[FittedModel] = []
    for variant_name, variant_params in variants:
        params = {**common_params, **variant_params}
        for device in ("cuda", "cpu"):
            estimator = XGBClassifier(**params, device=device)
            pipeline = Pipeline([("preprocess", clone(preprocessor)), ("model", estimator)])
            try:
                LOGGER.info("Entrenando %s con device=%s.", variant_name, device)
                pipeline.fit(X_train, y_train)
                joblib.dump(pipeline, models_dir / f"{variant_name}_{device}.joblib")
                note = f"Entrenado con device={device}; scale_pos_weight={class_ratio:.3f}."
                fitted.append(FittedModel(name=f"{variant_name}_{device}", estimator=pipeline, notes=note))
                break
            except Exception as exc:
                LOGGER.warning("%s device=%s fallo: %s", variant_name, device, exc)
    return fitted


def _negative_positive_ratio(y: pd.Series) -> float:
    counts = y.value_counts()
    positives = counts.get(1, 0)
    negatives = counts.get(0, 0)
    return float(negatives / positives) if positives else 1.0
