#!/usr/bin/env python
"""CLI principal para ejecutar el pipeline de cancer de extremo a extremo."""

from __future__ import annotations

import argparse
from itertools import combinations
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from pypdf import PdfReader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cancer_ml.config import (  # noqa: E402
    DEFAULT_SEED,
    FINAL_PRESENTATION_PATH,
    FIGURES_DIR,
    METRICS_DIR,
    MODE_CONFIGS,
    MODELS_DIR,
    OUTPUT_DIR,
    RECOMMENDED_FEATURE_VIEW,
    RECOMMENDED_MODEL_ARTIFACT,
    ensure_directories,
    set_global_seed,
)
from cancer_ml.data import audit_collections, cancer_balance, create_eda_summary, join_collections, load_available_csvs  # noqa: E402
from cancer_ml.evaluation import (  # noqa: E402
    evaluate_estimator,
    evaluate_probabilities,
    save_bootstrap_intervals,
    save_metrics,
    threshold_search,
)
from cancer_ml.features import (  # noqa: E402
    FEATURE_VIEWS,
    add_engineered_features,
    build_feature_policy,
    build_preprocessor,
    create_feature_signal_report,
    save_feature_policy,
    save_split_summary,
    split_data,
)
from cancer_ml.models_ml import train_classical_models  # noqa: E402
from cancer_ml.models_mlp import train_mlp  # noqa: E402
from cancer_ml.plots import (  # noqa: E402
    plot_confusion_matrix,
    plot_metric_comparison,
    plot_mlp_learning_curves,
    plot_precision_recall_space,
    plot_roc_curves,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecuta el caso de prediccion de cancer.")
    parser.add_argument("--mode", choices=sorted(MODE_CONFIGS), default="quick")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--feature-view",
        choices=FEATURE_VIEWS,
        default=RECOMMENDED_FEATURE_VIEW,
        help=(
            "Vista de datos para modelado: safe_all es la vista final recomendada "
            "por F1 limpio con los datos disponibles; metadata_core conserva el nucleo "
            "clinico estricto segun metadato; base mantiene el conjunto historico; "
            "engineered_selected anade derivadas alineadas con el modelo generativo oficial; "
            "economic_sensitivity mide el efecto de variables economicas con riesgo de fuga."
        ),
    )
    parser.add_argument(
        "--allow-leakage-view",
        action="store_true",
        help="Permite ejecutar economic_sensitivity solo para auditorias de fuga, nunca como entrega operativa.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> int:
    args = parse_args()
    configure_logging()
    logger = logging.getLogger("run_pipeline")
    mode_config = MODE_CONFIGS[args.mode]
    if args.feature_view == "economic_sensitivity" and not args.allow_leakage_view:
        raise SystemExit(
            "economic_sensitivity incluye variables de coste/uso con riesgo de fuga. "
            "Usa --allow-leakage-view solo para auditorias, no para la entrega final."
        )
    ensure_directories()
    set_global_seed(args.seed)

    logger.info(
        "Inicio pipeline mode=%s seed=%s feature_view=%s primary_metric=F1.",
        args.mode,
        args.seed,
        args.feature_view,
    )
    frames, missing = load_available_csvs()
    audit = audit_collections(frames, missing, METRICS_DIR / "data_audit.csv")
    dataset = join_collections(frames)
    dataset.to_parquet(OUTPUT_DIR / "dataset_unido.parquet", index=False)
    eda = create_eda_summary(dataset, METRICS_DIR)
    feature_signal = create_feature_signal_report(dataset, METRICS_DIR / "feature_signal_report.csv")
    balance = cancer_balance(dataset)
    (METRICS_DIR / "target_balance.json").write_text(json.dumps(balance, indent=2) + "\n", encoding="utf-8")
    logger.info("Balance cancer: %s.", balance)

    modeling_dataset = add_engineered_features(dataset) if args.feature_view == "engineered_selected" else dataset
    if args.feature_view == "engineered_selected":
        modeling_dataset.to_parquet(OUTPUT_DIR / "dataset_modelado_engineered.parquet", index=False)
    policy = build_feature_policy(modeling_dataset, feature_view=args.feature_view)
    save_feature_policy(policy)
    splits = split_data(modeling_dataset, policy, seed=args.seed)
    save_split_summary(splits, METRICS_DIR / "split_summary.csv")
    preprocessor = build_preprocessor(policy)

    X_train = splits["X_train"]
    y_train = splits["y_train"]
    X_ml_train = splits["X_train_inner"]
    y_ml_train = splits["y_train_inner"]
    if mode_config.train_sample_size and mode_config.train_sample_size < len(y_train):
        X_ml_train, _, y_ml_train, _ = train_test_split(
            X_ml_train,
            y_ml_train,
            train_size=mode_config.train_sample_size,
            stratify=y_ml_train,
            random_state=args.seed,
        )
        logger.info("Modo quick: entrenamiento ML reducido a %d filas estratificadas.", len(y_ml_train))

    fitted_models = train_classical_models(preprocessor, X_ml_train, y_ml_train, mode_config, args.seed)
    results = []
    probabilities = {}
    validation_probabilities = {}
    threshold_tables = []
    for fitted in fitted_models:
        valid_proba = fitted.estimator.predict_proba(splits["X_valid"])[:, 1]
        best_model_threshold, model_threshold_table = threshold_search(
            splits["y_valid"],
            valid_proba,
            None,
        )
        model_threshold_table.insert(0, "model", fitted.name)
        threshold_tables.append(model_threshold_table)
        result = evaluate_estimator(
            fitted.name,
            fitted.model_type,
            fitted.estimator,
            splits["X_test"],
            splits["y_test"],
            threshold=best_model_threshold,
        )
        results.append(result)
        probabilities[fitted.name] = result.y_proba
        validation_probabilities[fitted.name] = valid_proba
        logger.info(
            "%s | threshold=%.2f precision=%.3f recall=%.3f f1=%.3f auc=%.3f acc=%.3f.",
            fitted.name,
            best_model_threshold,
            result.precision_positive,
            result.recall_positive,
            result.f1_positive,
            result.auc_roc,
            result.accuracy,
        )

    ml_results = [result for result in results if result.model_type == "ML"]
    if len(ml_results) < 3:
        raise RuntimeError(f"Se requieren al menos 3 modelos ML complejos; entrenados={len(ml_results)}.")

    mlp_artifacts = train_mlp(
        preprocessor,
        splits["X_train_inner"],
        splits["y_train_inner"],
        splits["X_valid"],
        splits["y_valid"],
        splits["X_test"],
        mode_config,
        args.seed,
    )
    mlp_artifacts.history.to_csv(METRICS_DIR / "mlp_history.csv", index=False)
    best_threshold, threshold_table = threshold_search(
        mlp_artifacts.validation_target,
        mlp_artifacts.validation_proba,
        METRICS_DIR / "mlp_threshold_search.csv",
    )
    threshold_table.insert(0, "model", "MLP")
    threshold_tables.append(threshold_table)
    logger.info("Umbral MLP seleccionado en validacion: %.2f.", best_threshold)
    mlp_result = evaluate_probabilities(
        "MLP",
        "MLP",
        splits["y_test"],
        mlp_artifacts.test_proba,
        threshold=best_threshold,
    )
    results.append(mlp_result)
    probabilities["MLP"] = mlp_result.y_proba
    validation_probabilities["MLP"] = mlp_artifacts.validation_proba

    ensemble_result, ensemble_threshold_table, ensemble_members = build_validation_selected_ensemble(
        validation_probabilities,
        probabilities,
        splits["y_valid"],
        splits["y_test"],
    )
    if ensemble_result is not None:
        results.append(ensemble_result)
        probabilities[ensemble_result.model] = ensemble_result.y_proba
        ensemble_threshold_table.insert(0, "model", ensemble_result.model)
        threshold_tables.append(ensemble_threshold_table)
        logger.info(
            "%s | members=%s threshold=%.2f precision=%.3f recall=%.3f f1=%.3f auc=%.3f acc=%.3f.",
            ensemble_result.model,
            "+".join(ensemble_members),
            ensemble_result.threshold,
            ensemble_result.precision_positive,
            ensemble_result.recall_positive,
            ensemble_result.f1_positive,
            ensemble_result.auc_roc,
            ensemble_result.accuracy,
        )

    threshold_searches = pd.concat(threshold_tables, ignore_index=True)
    threshold_searches.to_csv(METRICS_DIR / "model_threshold_search.csv", index=False)
    metrics = save_metrics(results, METRICS_DIR / "model_metrics.csv")
    intervals = save_bootstrap_intervals(
        results,
        splits["y_test"],
        METRICS_DIR / "model_metric_intervals.csv",
        seed=args.seed,
    )
    best_ml = (
        metrics[metrics["model_type"] == "ML"]
        .sort_values(["f1_positive", "auc_pr", "auc_roc", "recall_positive"], ascending=False)
        .iloc[0]
    )
    model_card = save_recommended_model_artifacts(best_ml, args.feature_view, args.seed, len(dataset))
    cm = [[int(best_ml["tn"]), int(best_ml["fp"])], [int(best_ml["fn"]), int(best_ml["tp"])]]
    plot_confusion_matrix(cm, str(best_ml["model"]), FIGURES_DIR / "confusion_matrix_best_ml.png")
    plot_mlp_learning_curves(mlp_artifacts.history, FIGURES_DIR / "mlp_learning_curves.png")
    plot_roc_curves(splits["y_test"], probabilities, FIGURES_DIR / "roc_curves.png")
    plot_metric_comparison(metrics, FIGURES_DIR / "metric_comparison.png")
    plot_precision_recall_space(splits["y_test"], probabilities, FIGURES_DIR / "precision_recall_space.png")

    pdf_path, page_count = generate_final_presentation()

    summary = {
        "mode": args.mode,
        "seed": args.seed,
        "missing_csvs": missing,
        "rows_joined": int(len(dataset)),
        "columns_joined": int(dataset.shape[1]),
        "feature_view": args.feature_view,
        "primary_selection_metric": "F1 cancer=1",
        "modeling_columns": int(modeling_dataset.shape[1]),
        "audit_rows": int(len(audit)),
        "eda_rows": int(len(eda)),
        "feature_signal_rows": int(len(feature_signal)),
        "feature_counts": {
            "numeric": len(policy.numeric),
            "binary": len(policy.binary),
            "categorical": len(policy.categorical),
            "engineered": len(policy.engineered),
            "excluded": len(policy.excluded),
        },
        "ml_complex_models_evaluated": int(len(ml_results)),
        "models_evaluated": metrics["model"].tolist(),
        "best_model_by_f1": str(metrics.iloc[0]["model"]),
        "best_ml_by_f1": str(best_ml["model"]),
        "recommended_model_artifact": model_card["artifact"],
        "model_card_path": str(METRICS_DIR / "model_card.json"),
        "mlp_threshold_from_validation": best_threshold,
        "model_thresholds_from_validation": True,
        "ensemble_members": ensemble_members,
        "threshold_search_rows": int(len(threshold_searches)),
        "bootstrap_interval_rows": int(len(intervals)),
        "pdf_path": str(pdf_path),
        "pdf_pages": page_count,
    }
    (METRICS_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Pipeline completado. PDF=%s paginas=%s.", pdf_path, page_count)
    return 0


def build_validation_selected_ensemble(
    validation_probabilities: dict[str, object],
    test_probabilities: dict[str, object],
    y_valid,
    y_test,
):
    """Selecciona un ensemble simple con validacion y lo evalua una vez en test."""

    if len(validation_probabilities) < 2:
        return None, pd.DataFrame(), []

    validation_scores = []
    for name, valid_proba in validation_probabilities.items():
        threshold, _ = threshold_search(y_valid, valid_proba, None)
        valid_pred = (valid_proba >= threshold).astype(int)
        validation_scores.append(
            (float(f1_score(y_valid, valid_pred, pos_label=1, zero_division=0)), name)
        )
    ranked_names = [name for _, name in sorted(validation_scores, reverse=True)]
    candidate_names = ranked_names[:5]

    best = None
    for size in range(2, min(4, len(candidate_names)) + 1):
        for members in combinations(candidate_names, size):
            valid_proba = sum(validation_probabilities[name] for name in members) / len(members)
            threshold, table = threshold_search(y_valid, valid_proba, None)
            valid_pred = (valid_proba >= threshold).astype(int)
            valid_f1 = float(f1_score(y_valid, valid_pred, pos_label=1, zero_division=0))
            if best is None or valid_f1 > best["valid_f1"]:
                test_proba = sum(test_probabilities[name] for name in members) / len(members)
                best = {
                    "members": list(members),
                    "valid_f1": valid_f1,
                    "threshold": threshold,
                    "threshold_table": table,
                    "test_proba": test_proba,
                }

    if best is None:
        return None, pd.DataFrame(), []

    name = "ValidationSoftVoting"
    result = evaluate_probabilities(name, "Ensemble", y_test, best["test_proba"], best["threshold"])
    return result, best["threshold_table"], best["members"]


def save_recommended_model_artifacts(
    best_ml: pd.Series,
    feature_view: str,
    seed: int,
    rows_joined: int,
) -> dict[str, object]:
    """Guarda el modelo ML recomendado y una ficha reproducible de evaluacion."""

    model_name = str(best_ml["model"])
    source_artifact = MODELS_DIR / f"{model_name}.joblib"
    recommended_artifact = RECOMMENDED_MODEL_ARTIFACT
    if source_artifact.exists():
        shutil.copy2(source_artifact, recommended_artifact)
        artifact_value: str | None = str(recommended_artifact.relative_to(PROJECT_ROOT))
        source_value: str | None = str(source_artifact.relative_to(PROJECT_ROOT))
    else:
        artifact_value = None
        source_value = None

    card = {
        "model_name": model_name,
        "artifact": artifact_value,
        "source_artifact": source_value,
        "target": "cancer",
        "positive_class": 1,
        "primary_metric": "f1_positive",
        "threshold_selected_on": "validation",
        "threshold": float(best_ml["threshold"]),
        "test_metrics": {
            "precision_positive": float(best_ml["precision_positive"]),
            "recall_positive": float(best_ml["recall_positive"]),
            "f1_positive": float(best_ml["f1_positive"]),
            "auc_roc": float(best_ml["auc_roc"]),
            "auc_pr": float(best_ml["auc_pr"]),
            "accuracy": float(best_ml["accuracy"]),
        },
        "confusion_matrix_test": {
            "tn": int(best_ml["tn"]),
            "fp": int(best_ml["fp"]),
            "fn": int(best_ml["fn"]),
            "tp": int(best_ml["tp"]),
        },
        "feature_view": feature_view,
        "seed": int(seed),
        "rows_joined": int(rows_joined),
        "limitations": [
            "synthetic dataset",
            "requires temporal and external clinical validation",
            "excludes vive and post-diagnosis cost/usage variables for leakage control",
        ],
    }
    (METRICS_DIR / "model_card.json").write_text(
        json.dumps(card, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return card


def generate_final_presentation() -> tuple[Path, int]:
    """Genera la presentacion final profesional y verifica que tenga 5 diapositivas."""

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "generate_five_slide_presentation.py")],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "No se pudo generar la presentacion final.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    page_count = len(PdfReader(str(FINAL_PRESENTATION_PATH)).pages)
    if page_count != 5:
        raise RuntimeError(f"La presentacion final debe tener 5 diapositivas; tiene {page_count}.")
    return FINAL_PRESENTATION_PATH, page_count


if __name__ == "__main__":
    raise SystemExit(main())
