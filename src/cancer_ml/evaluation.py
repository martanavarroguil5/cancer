"""Metricas comunes para todos los modelos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class EvaluationResult:
    model: str
    model_type: str
    threshold: float
    precision_positive: float
    recall_positive: float
    f1_positive: float
    auc_roc: float
    auc_pr: float
    accuracy: float
    balanced_accuracy: float
    specificity: float
    negative_predictive_value: float
    false_positive_rate: float
    false_negative_rate: float
    matthews_corrcoef: float
    brier_score: float
    predicted_positive_rate: float
    tn: int
    fp: int
    fn: int
    tp: int
    y_proba: np.ndarray
    y_pred: np.ndarray

    def as_row(self) -> dict[str, float | int | str]:
        return {
            "model": self.model,
            "model_type": self.model_type,
            "threshold": self.threshold,
            "precision_positive": self.precision_positive,
            "recall_positive": self.recall_positive,
            "f1_positive": self.f1_positive,
            "auc_roc": self.auc_roc,
            "auc_pr": self.auc_pr,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "specificity": self.specificity,
            "negative_predictive_value": self.negative_predictive_value,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
            "matthews_corrcoef": self.matthews_corrcoef,
            "brier_score": self.brier_score,
            "predicted_positive_rate": self.predicted_positive_rate,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "tp": self.tp,
        }


def get_positive_probabilities(model: Any, X: Any) -> np.ndarray:
    """Obtiene probabilidades de clase positiva con fallback para decision_function."""

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return np.asarray(proba)[:, 1]
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X))
        return 1.0 / (1.0 + np.exp(-scores))
    predictions = np.asarray(model.predict(X))
    return predictions.astype(float)


def evaluate_probabilities(
    model_name: str,
    model_type: str,
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> EvaluationResult:
    """Evalua metricas pedidas usando siempre pos_label=1."""

    y_true_array = np.asarray(y_true).astype(int)
    y_proba_array = np.asarray(y_proba).astype(float)
    y_pred = (y_proba_array >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_array, y_pred, labels=[0, 1]).ravel()
    try:
        auc_roc = float(roc_auc_score(y_true_array, y_proba_array))
    except ValueError:
        auc_roc = float("nan")
    try:
        auc_pr = float(average_precision_score(y_true_array, y_proba_array))
    except ValueError:
        auc_pr = float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    negative_predictive_value = tn / (tn + fn) if (tn + fn) else 0.0
    false_positive_rate = fp / (tn + fp) if (tn + fp) else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) else 0.0
    return EvaluationResult(
        model=model_name,
        model_type=model_type,
        threshold=float(threshold),
        precision_positive=float(precision_score(y_true_array, y_pred, pos_label=1, zero_division=0)),
        recall_positive=float(recall_score(y_true_array, y_pred, pos_label=1, zero_division=0)),
        f1_positive=float(f1_score(y_true_array, y_pred, pos_label=1, zero_division=0)),
        auc_roc=auc_roc,
        auc_pr=auc_pr,
        accuracy=float(accuracy_score(y_true_array, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true_array, y_pred)),
        specificity=float(specificity),
        negative_predictive_value=float(negative_predictive_value),
        false_positive_rate=float(false_positive_rate),
        false_negative_rate=float(false_negative_rate),
        matthews_corrcoef=float(matthews_corrcoef(y_true_array, y_pred)),
        brier_score=float(brier_score_loss(y_true_array, y_proba_array)),
        predicted_positive_rate=float(np.mean(y_pred)),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
        y_proba=y_proba_array,
        y_pred=y_pred,
    )


def evaluate_estimator(
    model_name: str,
    model_type: str,
    estimator: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> EvaluationResult:
    y_proba = get_positive_probabilities(estimator, X_test)
    return evaluate_probabilities(model_name, model_type, y_test, y_proba, threshold)


def threshold_search(
    y_valid: pd.Series | np.ndarray,
    y_valid_proba: np.ndarray,
    output_path: Path | None = None,
) -> tuple[float, pd.DataFrame]:
    """Selecciona en validacion el umbral que maximiza F1 positivo."""

    y_true = np.asarray(y_valid).astype(int)
    rows = []
    for threshold in np.round(np.arange(0.10, 0.901, 0.01), 2):
        y_pred = (np.asarray(y_valid_proba) >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "precision_positive": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
                "recall_positive": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
                "f1_positive": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "predicted_positive_rate": float(np.mean(y_pred)),
            }
        )
    table = pd.DataFrame(rows)
    best = table.sort_values(
        ["f1_positive", "precision_positive", "recall_positive"],
        ascending=[False, False, False],
    ).iloc[0]
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_path, index=False)
    return float(best["threshold"]), table


def save_metrics(results: list[EvaluationResult], output_path: Path) -> pd.DataFrame:
    metrics = pd.DataFrame([result.as_row() for result in results])
    metrics = metrics.sort_values(["f1_positive", "auc_pr", "auc_roc", "recall_positive"], ascending=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_path, index=False)
    return metrics


def save_bootstrap_intervals(
    results: list[EvaluationResult],
    y_true: pd.Series | np.ndarray,
    output_path: Path,
    n_bootstrap: int = 300,
    seed: int = 42,
) -> pd.DataFrame:
    """Guarda intervalos bootstrap de metricas clave sobre el test sellado."""

    y_true_array = np.asarray(y_true).astype(int)
    rng = np.random.default_rng(seed)
    rows = []
    for result in results:
        metric_values = {
            "f1_positive": [],
            "recall_positive": [],
            "precision_positive": [],
            "auc_roc": [],
            "auc_pr": [],
        }
        for _ in range(n_bootstrap):
            indices = rng.integers(0, len(y_true_array), len(y_true_array))
            y_sample = y_true_array[indices]
            if len(np.unique(y_sample)) < 2:
                continue
            proba_sample = result.y_proba[indices]
            pred_sample = (proba_sample >= result.threshold).astype(int)
            metric_values["f1_positive"].append(f1_score(y_sample, pred_sample, pos_label=1, zero_division=0))
            metric_values["recall_positive"].append(recall_score(y_sample, pred_sample, pos_label=1, zero_division=0))
            metric_values["precision_positive"].append(
                precision_score(y_sample, pred_sample, pos_label=1, zero_division=0)
            )
            metric_values["auc_roc"].append(roc_auc_score(y_sample, proba_sample))
            metric_values["auc_pr"].append(average_precision_score(y_sample, proba_sample))
        for metric, values in metric_values.items():
            values_array = np.asarray(values, dtype=float)
            rows.append(
                {
                    "model": result.model,
                    "metric": metric,
                    "mean": float(np.mean(values_array)),
                    "ci95_low": float(np.quantile(values_array, 0.025)),
                    "ci95_high": float(np.quantile(values_array, 0.975)),
                    "bootstrap_samples": int(len(values_array)),
                }
            )
    intervals = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    intervals.to_csv(output_path, index=False)
    return intervals
