#!/usr/bin/env python
"""Orquestador pesado de hypertraining limpio para ML tabular y MLP.

El objetivo es exprimir modelos sin contaminar el protocolo:

- split oficial train_inner / valid / test;
- seleccion de hiperparametros y umbrales solo en validacion;
- test usado una vez por candidato para auditoria final;
- vistas limpias por defecto: metadata_core, safe_all y engineered_selected.

No forma parte del pipeline principal. Es una herramienta de investigacion
computacionalmente costosa, pensada para ejecuciones largas en GPU/RTX 3090.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, replace
from itertools import combinations, product
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cancer_ml.config import DEFAULT_SEED, METRICS_DIR, MODE_CONFIGS, set_global_seed  # noqa: E402
from cancer_ml.data import join_collections, load_available_csvs  # noqa: E402
from cancer_ml.evaluation import evaluate_estimator, evaluate_probabilities, threshold_search  # noqa: E402
from cancer_ml.features import (  # noqa: E402
    FEATURE_VIEWS,
    add_engineered_features,
    build_feature_policy,
    build_preprocessor,
    split_data,
)
from cancer_ml.models_mlp import MLPConfig, _as_float32  # noqa: E402

from run_mlp_experiments import (  # noqa: E402
    balanced_class_weight,
    candidate_configs,
    train_candidate,
)


LOGGER = logging.getLogger("run_super_hypertraining")
CLEAN_VIEWS = ("metadata_core", "safe_all", "engineered_selected")
SELECTION_COLUMNS = ["valid_f1_positive", "valid_auc_pr", "valid_auc_roc"]
FINAL_RANKING_COLUMNS = [
    "source",
    "family",
    "candidate",
    "view",
    "valid_threshold",
    "valid_f1_positive",
    "valid_precision_positive",
    "valid_recall_positive",
    "valid_auc_pr",
    "valid_auc_roc",
    "test_f1_positive",
    "test_precision_positive",
    "test_recall_positive",
    "test_auc_roc",
    "test_auc_pr",
]


@dataclass
class ManualPreprocessedEstimator:
    preprocessor: object
    model: object

    def predict_proba(self, X):
        return self.model.predict_proba(self.preprocessor.transform(X))

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hypertraining pesado ML + MLP con validacion limpia.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--views", choices=FEATURE_VIEWS, nargs="+", default=list(CLEAN_VIEWS))
    parser.add_argument("--mode", choices=sorted(MODE_CONFIGS), default="full")
    parser.add_argument(
        "--profile",
        choices=["smoke", "balanced", "rtx3090"],
        default="balanced",
        help="Tamano de busqueda. rtx3090 es el perfil caro.",
    )
    parser.add_argument("--ml-limit", type=int, default=None, help="Limita candidatos ML tras construir la parrilla.")
    parser.add_argument("--skip-ml", action="store_true")
    parser.add_argument("--skip-mlp", action="store_true")
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Abortar si TensorFlow no detecta GPU y prohibir fallbacks CPU en estimadores GPU.",
    )
    parser.add_argument(
        "--gpu-only",
        action="store_true",
        help="Entrenar solo candidatos con backend GPU: XGBoost, LightGBM, CatBoost y MLP.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Detiene la ejecucion ante el primer candidato fallido. Por defecto se registra y continua.",
    )
    parser.add_argument(
        "--ensemble-top-k",
        type=int,
        default=12,
        help="Numero maximo de candidatos por vista usados para ensembles ML.",
    )
    parser.add_argument(
        "--ensemble-max-members",
        type=int,
        default=4,
        help="Tamano maximo de ensembles soft-voting.",
    )
    parser.add_argument("--skip-ensembles", action="store_true", help="Desactiva ensembles para ahorrar tiempo.")
    parser.add_argument("--mlp-suite", choices=["smoke", "broad", "refine", "rtx3090"], default="refine")
    parser.add_argument("--mlp-limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--reduce-patience", type=int, default=None)
    parser.add_argument("--verbose-fit", type=int, default=0)
    parser.add_argument(
        "--include-leakage-view",
        action="store_true",
        help="Permite economic_sensitivity. Usar solo para sensibilidad, no para seleccion final limpia.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=METRICS_DIR / "super_hypertraining",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Directorio donde guardar los mejores modelos. Por defecto: <output-dir>/models.",
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
    set_global_seed(args.seed)
    if args.gpu_only:
        args.require_gpu = True
    verify_required_gpu(args.require_gpu, require_tensorflow=not args.skip_mlp)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.models_dir is None:
        args.models_dir = args.output_dir / "models"
    args.models_dir.mkdir(parents=True, exist_ok=True)

    views = list(args.views)
    if "economic_sensitivity" in views and not args.include_leakage_view:
        raise SystemExit("economic_sensitivity requiere --include-leakage-view porque incluye variables de fuga.")

    mode_config = MODE_CONFIGS[args.mode]
    mode_config = replace(
        mode_config,
        mlp_epochs=args.epochs if args.epochs is not None else default_epochs(args.profile, mode_config.mlp_epochs),
        mlp_patience=args.patience if args.patience is not None else default_patience(args.profile, mode_config.mlp_patience),
        mlp_reduce_patience=(
            args.reduce_patience
            if args.reduce_patience is not None
            else default_reduce_patience(args.profile, mode_config.mlp_reduce_patience)
        ),
    )

    frames, missing = load_available_csvs()
    raw_dataset = join_collections(frames)
    dataset_by_view = {
        view: add_engineered_features(raw_dataset) if view == "engineered_selected" else raw_dataset for view in views
    }

    ml_results = pd.DataFrame()
    ml_ensembles = pd.DataFrame()
    mlp_results = pd.DataFrame()
    mlp_ensembles = pd.DataFrame()
    if not args.skip_ml:
        ml_results, ml_ensembles = run_ml_search(args, mode_config, dataset_by_view)
    if not args.skip_mlp:
        mlp_results, mlp_ensembles = run_mlp_search(args, mode_config, dataset_by_view)

    final_ranking = build_final_ranking(ml_results, ml_ensembles, mlp_results, mlp_ensembles)
    final_path = args.output_dir / "final_ranking.csv"
    final_ranking.to_csv(final_path, index=False)
    summary = {
        "seed": args.seed,
        "profile": args.profile,
        "mode": args.mode,
        "views": views,
        "missing_csvs": missing,
        "ml_candidates": int(len(ml_results)),
        "ml_ensembles": int(len(ml_ensembles)),
        "mlp_candidates": int(len(mlp_results)),
        "mlp_ensembles": int(len(mlp_ensembles)),
        "ensemble_top_k": int(args.ensemble_top_k),
        "ensemble_max_members": int(args.ensemble_max_members),
        "ensembles_enabled": not args.skip_ensembles,
        "models_dir": str(args.models_dir),
        "best_by_validation": final_ranking.head(10).to_dict(orient="records") if not final_ranking.empty else [],
        "selection_rule": render_selection_rule(),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "summary.md").write_text(render_summary(final_ranking, summary), encoding="utf-8")
    LOGGER.info("Ranking final guardado en %s", final_path)
    if not final_ranking.empty:
        print(final_ranking.head(20).to_string(index=False))
    return 0


def run_ml_search(
    args: argparse.Namespace,
    mode_config,
    dataset_by_view: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    ensemble_frames = []
    candidates = ml_candidates(
        args.profile,
        args.seed,
        gpu_only=args.gpu_only,
        use_gpu=args.require_gpu,
    )
    if args.ml_limit:
        candidates = candidates[: args.ml_limit]
    total = len(candidates) * len(dataset_by_view)
    counter = 0
    for view, dataset in dataset_by_view.items():
        policy = build_feature_policy(dataset, view)
        splits = split_data(dataset, policy, seed=args.seed)
        X_train = splits["X_train_inner"]
        y_train = splits["y_train_inner"]
        if mode_config.train_sample_size and mode_config.train_sample_size < len(y_train):
            X_train, _, y_train, _ = train_test_split(
                X_train,
                y_train,
                train_size=mode_config.train_sample_size,
                stratify=y_train,
                random_state=args.seed,
            )
        valid_probas: dict[str, np.ndarray] = {}
        test_probas: dict[str, np.ndarray] = {}
        view_rows = []
        for spec in candidates:
            counter += 1
            LOGGER.info("[%d/%d ML] %s | view=%s", counter, total, spec["name"], view)
            started = perf_counter()
            try:
                row, valid_proba, test_proba = train_ml_candidate(
                    spec=spec,
                    view=view,
                    policy=policy,
                    X_train=X_train,
                    y_train=y_train,
                    splits=splits,
                    seed=args.seed,
                    output_dir=args.output_dir,
                    models_dir=args.models_dir,
                    allow_cpu_fallback=not args.require_gpu,
                )
                valid_probas[spec["name"]] = valid_proba
                test_probas[spec["name"]] = test_proba
            except Exception as exc:
                if args.fail_fast:
                    raise
                LOGGER.exception("ML fallo: %s | view=%s", spec["name"], view)
                row = {
                    "family": spec["family"],
                    "candidate": spec["name"],
                    "view": view,
                    "status": "failed",
                    "error": repr(exc),
                    "params": json.dumps(spec["params"], sort_keys=True),
                }
            row["fit_seconds"] = perf_counter() - started
            rows.append(row)
            view_rows.append(row)
            sort_for_selection(pd.DataFrame(rows)).to_csv(args.output_dir / "ml_results.csv", index=False)
        ranking = sort_for_selection(pd.DataFrame(view_rows))
        if not args.skip_ensembles:
            ensemble_path = args.output_dir / f"ml_ensembles_{view}.csv"
            ensembles = build_limited_ensemble_ranking(
                ranking,
                valid_probas,
                test_probas,
                splits["y_valid"],
                splits["y_test"],
                ensemble_path,
                top_k=args.ensemble_top_k,
                max_members=args.ensemble_max_members,
            )
            if not ensembles.empty:
                ensembles.insert(0, "view", view)
                ensembles.insert(0, "family", "MLEnsemble")
                ensemble_frames.append(ensembles)
    results = pd.DataFrame(rows)
    if not results.empty:
        results = sort_for_selection(results)
        results.to_csv(args.output_dir / "ml_results.csv", index=False)
    ensemble_results = pd.concat(ensemble_frames, ignore_index=True) if ensemble_frames else pd.DataFrame()
    if not ensemble_results.empty:
        ensemble_results = sort_for_selection(ensemble_results)
        ensemble_results.to_csv(args.output_dir / "ml_ensembles.csv", index=False)
    return results, ensemble_results


def train_ml_candidate(
    spec,
    view,
    policy,
    X_train,
    y_train,
    splits,
    seed: int,
    output_dir: Path,
    models_dir: Path,
    allow_cpu_fallback: bool,
):
    if spec["family"] == "CatBoost":
        return train_manual_preprocessed_candidate(
            spec,
            view,
            policy,
            X_train,
            y_train,
            splits,
            output_dir,
            models_dir,
        )

    preprocessor = build_preprocessor(policy)
    estimator = spec["factory"]()
    effective_params = configure_estimator_for_training(estimator, spec, y_train)
    pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
    fit_note = ""
    try:
        pipeline.fit(X_train, y_train)
    except Exception as exc:
        if spec["family"] != "XGBoost" or not allow_cpu_fallback:
            raise
        LOGGER.warning("%s fallo con device=cuda; reintentando XGBoost en CPU. Error: %s", spec["name"], exc)
        estimator = spec["factory"]()
        effective_params = configure_estimator_for_training(estimator, spec, y_train, force_device="cpu")
        pipeline = Pipeline([("preprocess", build_preprocessor(policy)), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        fit_note = f"cuda_failed_then_cpu: {exc!r}"
    valid_proba = pipeline.predict_proba(splits["X_valid"])[:, 1]
    threshold, _ = threshold_search(splits["y_valid"], valid_proba, None)
    valid_result = evaluate_probabilities(
        f"{spec['name']}_valid",
        "MLHypertraining",
        splits["y_valid"],
        valid_proba,
        threshold,
    )
    test_result = evaluate_estimator(
        spec["name"],
        "MLHypertraining",
        pipeline,
        splits["X_test"],
        splits["y_test"],
        threshold,
    )
    row = {
        "family": spec["family"],
        "candidate": spec["name"],
        "view": view,
        "status": "ok",
        "valid_threshold": float(threshold),
        "valid_f1_positive": valid_result.f1_positive,
        "valid_precision_positive": valid_result.precision_positive,
        "valid_recall_positive": valid_result.recall_positive,
        "valid_auc_roc": valid_result.auc_roc,
        "valid_auc_pr": valid_result.auc_pr,
        "valid_brier_score": valid_result.brier_score,
        "test_f1_positive": test_result.f1_positive,
        "test_precision_positive": test_result.precision_positive,
        "test_recall_positive": test_result.recall_positive,
        "test_auc_roc": test_result.auc_roc,
        "test_auc_pr": test_result.auc_pr,
        "test_brier_score": test_result.brier_score,
        "feature_count": len(policy.included),
        "params": json.dumps(effective_params, sort_keys=True),
    }
    if fit_note:
        row["fit_note"] = fit_note
    maybe_save_best_model(pipeline, row, output_dir, models_dir)
    return row, valid_proba, test_result.y_proba


def train_manual_preprocessed_candidate(
    spec,
    view,
    policy,
    X_train,
    y_train,
    splits,
    output_dir: Path,
    models_dir: Path,
):
    preprocessor = build_preprocessor(policy)
    estimator = spec["factory"]()
    X_train_t = preprocessor.fit_transform(X_train)
    X_valid_t = preprocessor.transform(splits["X_valid"])
    X_test_t = preprocessor.transform(splits["X_test"])
    estimator.fit(X_train_t, y_train)

    valid_proba = estimator.predict_proba(X_valid_t)[:, 1]
    test_proba = estimator.predict_proba(X_test_t)[:, 1]
    threshold, _ = threshold_search(splits["y_valid"], valid_proba, None)
    valid_result = evaluate_probabilities(
        f"{spec['name']}_valid",
        "MLHypertraining",
        splits["y_valid"],
        valid_proba,
        threshold,
    )
    test_result = evaluate_probabilities(
        spec["name"],
        "MLHypertraining",
        splits["y_test"],
        test_proba,
        threshold,
    )
    row = {
        "family": spec["family"],
        "candidate": spec["name"],
        "view": view,
        "status": "ok",
        "valid_threshold": float(threshold),
        "valid_f1_positive": valid_result.f1_positive,
        "valid_precision_positive": valid_result.precision_positive,
        "valid_recall_positive": valid_result.recall_positive,
        "valid_auc_roc": valid_result.auc_roc,
        "valid_auc_pr": valid_result.auc_pr,
        "valid_brier_score": valid_result.brier_score,
        "test_f1_positive": test_result.f1_positive,
        "test_precision_positive": test_result.precision_positive,
        "test_recall_positive": test_result.recall_positive,
        "test_auc_roc": test_result.auc_roc,
        "test_auc_pr": test_result.auc_pr,
        "test_brier_score": test_result.brier_score,
        "feature_count": len(policy.included),
        "params": json.dumps(spec["params"], sort_keys=True),
        "fit_note": "manual_preprocessing_to_avoid_sklearn_pipeline_tags",
    }
    wrapped = ManualPreprocessedEstimator(preprocessor=preprocessor, model=estimator)
    maybe_save_best_model(wrapped, row, output_dir, models_dir)
    return row, valid_proba, test_result.y_proba


def run_mlp_search(args: argparse.Namespace, mode_config, dataset_by_view: dict[str, pd.DataFrame]):
    configs = mlp_candidates(args.mlp_suite)
    if args.mlp_limit:
        configs = configs[: args.mlp_limit]
    rows = []
    ensemble_frames = []
    total = len(configs) * len(dataset_by_view)
    counter = 0
    histories_dir = args.output_dir / "mlp_histories"
    histories_dir.mkdir(parents=True, exist_ok=True)
    for view, dataset in dataset_by_view.items():
        policy = build_feature_policy(dataset, view)
        splits = split_data(dataset, policy, seed=args.seed)
        preprocessor = build_preprocessor(policy)
        fitted_preprocessor, X_train, y_train, X_valid, y_valid, X_test, y_test = transform_splits_with_preprocessor(
            preprocessor,
            splits,
        )
        class_weight = balanced_class_weight(y_train)
        valid_probas: dict[str, np.ndarray] = {}
        test_probas: dict[str, np.ndarray] = {}
        view_rows = []
        for config in configs:
            counter += 1
            LOGGER.info("[%d/%d MLP] %s | view=%s", counter, total, config.name, view)
            started = perf_counter()
            try:
                row, history, model, valid_proba, test_proba = train_candidate(
                    config=config,
                    X_train=X_train,
                    y_train=y_train,
                    X_valid=X_valid,
                    y_valid=y_valid,
                    X_test=X_test,
                    y_test=y_test,
                    class_weight=class_weight,
                    mode_config=mode_config,
                    seed=args.seed,
                    verbose_fit=args.verbose_fit,
                )
                row["family"] = "MLP"
                row["view"] = view
                row["fit_seconds"] = perf_counter() - started
                history.to_csv(histories_dir / f"{view}_{config.name}.csv", index=False)
                valid_probas[config.name] = valid_proba
                test_probas[config.name] = test_proba
                apply_selection_metrics(
                    row,
                    y_valid,
                    valid_proba,
                    y_test,
                    test_proba,
                )
                maybe_save_best_mlp(model, row, config, args.output_dir, args.models_dir, fitted_preprocessor)
            except Exception as exc:
                if args.fail_fast:
                    raise
                LOGGER.exception("MLP fallo: %s | view=%s", config.name, view)
                row = {"candidate": config.name, "family": "MLP", "view": view, "status": "failed", "error": repr(exc)}
            row["fit_seconds"] = perf_counter() - started
            rows.append(row)
            view_rows.append(row)
            sort_for_selection(pd.DataFrame(rows)).to_csv(args.output_dir / "mlp_results.csv", index=False)
        ranking = sort_for_selection(pd.DataFrame(view_rows))
        ensemble_path = args.output_dir / f"mlp_ensembles_{view}.csv"
        ensembles = pd.DataFrame()
        if not args.skip_ensembles:
            ensemble_ranking = ranking.head(args.ensemble_top_k)
            ensembles = build_ensemble_ranking(
                ensemble_ranking,
                valid_probas,
                test_probas,
                y_valid,
                y_test,
                ensemble_path,
                max_members=args.ensemble_max_members,
            )
        if not ensembles.empty:
            ensembles.insert(0, "view", view)
            ensembles.insert(0, "family", "MLPEnsemble")
            ensemble_frames.append(ensembles)
    results = pd.DataFrame(rows)
    if not results.empty:
        results = sort_for_selection(results)
        results.to_csv(args.output_dir / "mlp_results.csv", index=False)
    ensemble_results = pd.concat(ensemble_frames, ignore_index=True) if ensemble_frames else pd.DataFrame()
    if not ensemble_results.empty:
        ensemble_results = sort_for_selection(ensemble_results)
        ensemble_results.to_csv(args.output_dir / "mlp_ensembles.csv", index=False)
    return results, ensemble_results


def ml_candidates(profile: str, seed: int, gpu_only: bool = False, use_gpu: bool = False) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    if profile == "smoke":
        hgb_grid = [
            {"learning_rate": 0.03, "max_iter": 220, "max_leaf_nodes": 31, "l2_regularization": 0.0, "min_samples_leaf": 20},
            {"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 15, "l2_regularization": 0.01, "min_samples_leaf": 30},
        ]
    elif profile == "balanced":
        hgb_grid = list(
            dict(
                learning_rate=lr,
                max_iter=iters,
                max_leaf_nodes=leaf,
                l2_regularization=l2,
                min_samples_leaf=min_leaf,
            )
            for lr, iters, leaf, l2, min_leaf in product(
                [0.02, 0.03, 0.05],
                [450, 800],
                [15, 31],
                [0.0, 0.01],
                [25],
            )
        )
    else:
        hgb_grid = list(
            dict(
                learning_rate=lr,
                max_iter=iters,
                max_leaf_nodes=leaf,
                l2_regularization=l2,
                min_samples_leaf=min_leaf,
            )
            for lr, iters, leaf, l2, min_leaf in product(
                [0.015, 0.025, 0.04],
                [700, 1100],
                [15, 31, 63],
                [0.0, 0.01, 0.03],
                [20, 50],
            )
        )
    for params in hgb_grid:
        candidates.append(
            {
                "family": "HistGradientBoosting",
                "name": "HGB_" + compact_params(params),
                "params": params,
                "factory": lambda params=params: HistGradientBoostingClassifier(
                    **params,
                    class_weight="balanced",
                    early_stopping=True,
                    random_state=seed,
                ),
            }
        )

    xgb_grid = xgb_candidates(profile)
    for params in xgb_grid:
        candidates.append(
            {
                "family": "XGBoost",
                "name": "XGB_" + compact_params(params),
                "params": params,
                "factory": lambda params=params: build_xgb(params, seed),
            }
        )

    if profile in {"balanced", "rtx3090"}:
        candidates.extend(extra_tree_candidates(profile, seed))
        candidates.extend(lightgbm_candidates(profile, seed, use_gpu=use_gpu))
        candidates.extend(catboost_candidates(profile, seed, use_gpu=use_gpu))
        candidates.append(
            {
                "family": "LogisticRegression",
                "name": "LogReg_C0.3_balanced",
                "params": {"C": 0.3},
                "factory": lambda: LogisticRegression(max_iter=3000, class_weight="balanced", C=0.3, random_state=seed),
            }
        )
    if gpu_only:
        candidates = [
            candidate
            for candidate in candidates
            if candidate["family"] in {"XGBoost", "LightGBM", "CatBoost"}
        ]
    return candidates


def xgb_candidates(profile: str) -> list[dict[str, object]]:
    if profile == "smoke":
        return [
            {
                "n_estimators": 250,
                "max_depth": 2,
                "learning_rate": 0.04,
                "subsample": 0.95,
                "colsample_bytree": 0.95,
                "min_child_weight": 1,
                "reg_lambda": 1.0,
                "reg_alpha": 0.0,
                "gamma": 0.0,
            }
        ]
    base = []
    for max_depth, lr, estimators, min_child, reg_lambda, gamma in product(
        [1, 2, 3],
        [0.03, 0.06] if profile == "balanced" else [0.02, 0.04, 0.07],
        [600, 1000] if profile == "balanced" else [800, 1200],
        [1, 5],
        [1.0, 3.0] if profile == "balanced" else [0.8, 1.5, 3.0],
        [0.0] if profile == "balanced" else [0.0, 0.1],
    ):
        base.append(
            {
                "n_estimators": estimators,
                "max_depth": max_depth,
                "learning_rate": lr,
                "subsample": 0.95,
                "colsample_bytree": 0.95,
                "min_child_weight": min_child,
                "reg_lambda": reg_lambda,
                "reg_alpha": 0.0,
                "gamma": gamma,
            }
        )
    return base


def build_xgb(params: dict[str, object], seed: int):
    from xgboost import XGBClassifier

    return XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device="cuda",
        scale_pos_weight=1.0,
        random_state=seed,
        n_jobs=-1,
    )


def extra_tree_candidates(profile: str, seed: int) -> list[dict[str, object]]:
    grids = [
        ("RandomForest", RandomForestClassifier, {"n_estimators": 800, "min_samples_leaf": 2, "max_features": "sqrt"}),
        ("ExtraTrees", ExtraTreesClassifier, {"n_estimators": 900, "min_samples_leaf": 4, "max_features": "sqrt"}),
    ]
    if profile == "rtx3090":
        grids.extend(
            [
                ("RandomForest", RandomForestClassifier, {"n_estimators": 1400, "min_samples_leaf": 3, "max_features": 0.7}),
                ("ExtraTrees", ExtraTreesClassifier, {"n_estimators": 1600, "min_samples_leaf": 2, "max_features": 0.7}),
            ]
        )
    candidates = []
    for family, cls, params in grids:
        candidates.append(
            {
                "family": family,
                "name": family + "_" + compact_params(params),
                "params": params,
                "factory": lambda cls=cls, params=params: cls(
                    **params,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=seed,
                ),
            }
        )
    return candidates


def lightgbm_candidates(profile: str, seed: int, use_gpu: bool = False) -> list[dict[str, object]]:
    try:
        from lightgbm import LGBMClassifier
    except Exception:
        return []
    grids = [
        {"n_estimators": 700, "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 30, "reg_lambda": 1.0},
        {"n_estimators": 1000, "learning_rate": 0.02, "num_leaves": 15, "min_child_samples": 40, "reg_lambda": 2.0},
    ]
    if profile == "rtx3090":
        grids.extend(
            [
                {"n_estimators": 1400, "learning_rate": 0.015, "num_leaves": 63, "min_child_samples": 25, "reg_lambda": 1.0},
                {"n_estimators": 1600, "learning_rate": 0.012, "num_leaves": 31, "min_child_samples": 60, "reg_lambda": 4.0},
            ]
        )
    candidates = []
    for params in grids:
        effective_params = dict(params)
        if use_gpu:
            effective_params["device_type"] = "gpu"
        candidates.append(
            {
                "family": "LightGBM",
                "name": "LGBM_" + compact_params(params) + ("_gpu" if use_gpu else ""),
                "params": effective_params,
                "factory": lambda effective_params=effective_params: LGBMClassifier(
                    **effective_params,
                    class_weight="balanced",
                    objective="binary",
                    random_state=seed,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            }
        )
    return candidates


def catboost_candidates(profile: str, seed: int, use_gpu: bool = False) -> list[dict[str, object]]:
    try:
        from catboost import CatBoostClassifier
    except Exception:
        return []
    grids = [
        {"iterations": 700, "depth": 4, "learning_rate": 0.035, "l2_leaf_reg": 3.0},
        {"iterations": 1000, "depth": 3, "learning_rate": 0.025, "l2_leaf_reg": 6.0},
    ]
    if profile == "rtx3090":
        grids.extend(
            [
                {"iterations": 1400, "depth": 5, "learning_rate": 0.02, "l2_leaf_reg": 4.0},
                {"iterations": 1800, "depth": 4, "learning_rate": 0.015, "l2_leaf_reg": 8.0},
            ]
        )
    candidates = []
    for params in grids:
        effective_params = dict(params)
        if use_gpu:
            effective_params.update({"task_type": "GPU", "devices": "0"})
        candidates.append(
            {
                "family": "CatBoost",
                "name": "CatBoost_" + compact_params(params) + ("_gpu" if use_gpu else ""),
                "params": effective_params,
                "factory": lambda effective_params=effective_params: CatBoostClassifier(
                    **effective_params,
                    loss_function="Logloss",
                    eval_metric="F1",
                    auto_class_weights="Balanced",
                    random_seed=seed,
                    verbose=False,
                    allow_writing_files=False,
                ),
            }
        )
    return candidates


def mlp_candidates(suite: str) -> list[MLPConfig]:
    if suite in {"smoke", "broad", "refine"}:
        return candidate_configs(suite)
    base = candidate_configs("broad")
    extras = [
        MLPConfig(
            name="rtx_wide_1536_768_384_192",
            hidden_units=(1536, 768, 384, 192),
            dropout=(0.22, 0.18, 0.14, 0.10),
            l2=3e-6,
            learning_rate=3e-4,
            weight_decay=5e-5,
            batch_size=512,
        ),
        MLPConfig(
            name="rtx_slim_selu_512_256_128",
            hidden_units=(512, 256, 128),
            activation="selu",
            batch_norm=False,
            dropout=(0.08, 0.06, 0.04),
            l2=1e-6,
            learning_rate=4e-4,
            weight_decay=1e-5,
            batch_size=512,
        ),
        MLPConfig(
            name="rtx_residual_768_768_384_384",
            hidden_units=(768, 768, 384, 384),
            block_type="residual",
            dropout=0.12,
            l2=3e-6,
            learning_rate=2.5e-4,
            weight_decay=8e-5,
            batch_size=512,
        ),
    ]
    return base + extras


def build_final_ranking(
    ml_results: pd.DataFrame,
    ml_ensembles: pd.DataFrame,
    mlp_results: pd.DataFrame,
    mlp_ensembles: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    if not ml_results.empty:
        ok = ml_results[ml_results["status"] == "ok"].copy() if "status" in ml_results else ml_results.copy()
        if not ok.empty:
            frames.append(select_final_columns(ok.assign(source="ML")))
    if not ml_ensembles.empty:
        frame = ml_ensembles.rename(columns={"members": "candidate"}).copy()
        frame["source"] = "MLEnsemble"
        frame["family"] = "MLEnsemble"
        frames.append(select_final_columns(frame))
    if not mlp_results.empty:
        ok = mlp_results[mlp_results["status"] == "ok"].copy() if "status" in mlp_results else mlp_results.copy()
        if not ok.empty:
            frames.append(select_final_columns(ok.assign(source="MLP")))
    if not mlp_ensembles.empty:
        frame = mlp_ensembles.rename(columns={"members": "candidate"}).copy()
        frame["source"] = "MLPEnsemble"
        frame["family"] = "MLPEnsemble"
        frames.append(select_final_columns(frame))
    if not frames:
        return pd.DataFrame()
    ranking = pd.concat(frames, ignore_index=True)
    ranking["valid_test_f1_gap"] = ranking["valid_f1_positive"] - ranking["test_f1_positive"]
    return sort_for_selection(ranking)


def select_final_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[[column for column in FINAL_RANKING_COLUMNS if column in frame.columns]]


def sort_for_selection(frame: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [column for column in SELECTION_COLUMNS if column in frame.columns]
    if not sort_columns:
        return frame
    return frame.sort_values(sort_columns, ascending=False, na_position="last")


def build_limited_ensemble_ranking(
    ranking: pd.DataFrame,
    valid_probas: dict[str, np.ndarray],
    test_probas: dict[str, np.ndarray],
    y_valid,
    y_test,
    output_path: Path,
    top_k: int,
    max_members: int,
) -> pd.DataFrame:
    """Construye ensembles solo con los mejores candidatos para evitar una explosion combinatoria."""

    if top_k < 2 or max_members < 2:
        return pd.DataFrame()
    ok = ranking[ranking["status"] == "ok"].copy() if "status" in ranking else ranking.copy()
    ok = ok[ok["candidate"].isin(valid_probas)]
    limited = sort_for_selection(ok).head(top_k)
    return build_ensemble_ranking(
        limited,
        valid_probas,
        test_probas,
        y_valid,
        y_test,
        output_path,
        max_members=max_members,
    )


def build_ensemble_ranking(
    ranking: pd.DataFrame,
    valid_probas: dict[str, np.ndarray],
    test_probas: dict[str, np.ndarray],
    y_valid,
    y_test,
    output_path: Path,
    max_members: int,
) -> pd.DataFrame:
    ok_ranking = ranking[ranking["status"] == "ok"].copy() if "status" in ranking else ranking.copy()
    if len(ok_ranking) < 2:
        return pd.DataFrame()
    candidate_names = [name for name in ok_ranking["candidate"].tolist() if name in valid_probas]
    rows = []
    for size in range(2, min(max_members, len(candidate_names)) + 1):
        for members in combinations(candidate_names, size):
            valid_proba = np.mean([valid_probas[name] for name in members], axis=0)
            test_proba = np.mean([test_probas[name] for name in members], axis=0)
            threshold, _ = threshold_search(y_valid, valid_proba, None)
            valid_eval = evaluate_probabilities(
                f"ensemble_{size}_valid",
                "Ensemble",
                y_valid,
                valid_proba,
                threshold,
            )
            test_eval = evaluate_probabilities(
                f"ensemble_{size}",
                "Ensemble",
                y_test,
                test_proba,
                threshold,
            )
            rows.append(
                {
                    "members": "+".join(members),
                    "member_count": size,
                    "valid_threshold": float(threshold),
                    "valid_f1_positive": valid_eval.f1_positive,
                    "valid_precision_positive": valid_eval.precision_positive,
                    "valid_recall_positive": valid_eval.recall_positive,
                    "valid_auc_roc": valid_eval.auc_roc,
                    "valid_auc_pr": valid_eval.auc_pr,
                    "test_f1_positive": test_eval.f1_positive,
                    "test_precision_positive": test_eval.precision_positive,
                    "test_recall_positive": test_eval.recall_positive,
                    "test_auc_roc": test_eval.auc_roc,
                    "test_auc_pr": test_eval.auc_pr,
                }
            )
    ensemble_ranking = pd.DataFrame(rows)
    if not ensemble_ranking.empty:
        ensemble_ranking = sort_for_selection(ensemble_ranking)
        ensemble_ranking.to_csv(output_path, index=False)
    return ensemble_ranking


def transform_splits_with_preprocessor(preprocessor, splits: dict[str, pd.DataFrame | pd.Series]):
    transformer = clone(preprocessor)
    X_train = _as_float32(transformer.fit_transform(splits["X_train_inner"]))
    X_valid = _as_float32(transformer.transform(splits["X_valid"]))
    X_test = _as_float32(transformer.transform(splits["X_test"]))
    return (
        transformer,
        X_train,
        np.asarray(splits["y_train_inner"]).astype(int),
        X_valid,
        np.asarray(splits["y_valid"]).astype(int),
        X_test,
        np.asarray(splits["y_test"]).astype(int),
    )


def configure_estimator_for_training(estimator, spec, y_train, force_device: str | None = None) -> dict[str, object]:
    effective_params = dict(spec["params"])
    if spec["family"] == "XGBoost" and hasattr(estimator, "set_params"):
        class_ratio = negative_positive_ratio(y_train)
        estimator.set_params(scale_pos_weight=class_ratio)
        effective_params["scale_pos_weight"] = class_ratio
        if force_device is not None:
            estimator.set_params(device=force_device)
        try:
            effective_params["device"] = estimator.get_params().get("device")
        except Exception:
            if force_device is not None:
                effective_params["device"] = force_device
    return effective_params


def negative_positive_ratio(y) -> float:
    y_array = np.asarray(y).astype(int)
    positives = int(np.sum(y_array == 1))
    negatives = int(np.sum(y_array == 0))
    return float(negatives / positives) if positives else 1.0


def verify_required_gpu(require_gpu: bool, require_tensorflow: bool) -> None:
    if not require_gpu:
        return
    try:
        import tensorflow as tf
    except Exception as exc:
        raise SystemExit(f"--require-gpu necesita TensorFlow disponible para verificar GPU: {exc}") from exc
    gpus = tf.config.list_physical_devices("GPU")
    if require_tensorflow and not gpus:
        raise SystemExit("--require-gpu activo: TensorFlow no detecta ninguna GPU; se aborta sin usar CPU.")
    if gpus:
        LOGGER.info("GPU TensorFlow detectada: %s.", ", ".join(gpu.name for gpu in gpus))


def render_selection_rule() -> str:
    return "Ordenar por valid_f1_positive, valid_auc_pr y valid_auc_roc; test solo se reporta."


def apply_selection_metrics(
    row: dict[str, object],
    y_valid,
    valid_proba: np.ndarray,
    y_test,
    test_proba: np.ndarray,
) -> None:
    threshold, _ = threshold_search(y_valid, valid_proba, None)
    valid_eval = evaluate_probabilities(
        f"{row['candidate']}_valid",
        str(row.get("family", "Model")),
        y_valid,
        valid_proba,
        threshold,
    )
    test_eval = evaluate_probabilities(
        str(row["candidate"]),
        str(row.get("family", "Model")),
        y_test,
        test_proba,
        threshold,
    )
    row.update(
        {
            "valid_threshold": float(threshold),
            "valid_f1_positive": valid_eval.f1_positive,
            "valid_precision_positive": valid_eval.precision_positive,
            "valid_recall_positive": valid_eval.recall_positive,
            "valid_auc_roc": valid_eval.auc_roc,
            "valid_auc_pr": valid_eval.auc_pr,
            "valid_brier_score": valid_eval.brier_score,
            "test_f1_positive": test_eval.f1_positive,
            "test_precision_positive": test_eval.precision_positive,
            "test_recall_positive": test_eval.recall_positive,
            "test_auc_roc": test_eval.auc_roc,
            "test_auc_pr": test_eval.auc_pr,
            "test_brier_score": test_eval.brier_score,
            "test_valid_gap_f1": valid_eval.f1_positive - test_eval.f1_positive,
        }
    )


def compact_params(params: dict[str, object]) -> str:
    pieces = []
    aliases = {
        "learning_rate": "lr",
        "max_iter": "it",
        "max_leaf_nodes": "leaf",
        "l2_regularization": "l2",
        "min_samples_leaf": "minleaf",
        "n_estimators": "n",
        "max_depth": "depth",
        "subsample": "sub",
        "colsample_bytree": "col",
        "min_child_weight": "child",
        "reg_lambda": "lambda",
        "reg_alpha": "alpha",
        "num_leaves": "leaves",
        "min_child_samples": "minsamp",
        "l2_leaf_reg": "catl2",
        "iterations": "iter",
        "max_features": "maxfeat",
    }
    for key in sorted(params):
        value = params[key]
        text = str(value).replace(".", "p")
        pieces.append(f"{aliases.get(key, key)}{text}")
    return "_".join(pieces)


def maybe_save_best_model(pipeline, row: dict[str, object], output_dir: Path, models_dir: Path) -> None:
    path = output_dir / "best_ml_marker.json"
    current = None
    if path.exists():
        current = json.loads(path.read_text())
    if current is None or row_is_better(row, current):
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / "super_best_ml.joblib"
        joblib.dump(pipeline, model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {**row, "model_path": str(model_path)}
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n", encoding="utf-8")


def maybe_save_best_mlp(
    model,
    row: dict[str, object],
    config: MLPConfig,
    output_dir: Path,
    models_dir: Path,
    preprocessor,
) -> None:
    path = output_dir / "best_mlp_marker.json"
    current = None
    if path.exists():
        current = json.loads(path.read_text())
    if current is None or row_is_better(row, current):
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / "super_best_mlp.keras"
        preprocessor_path = models_dir / "super_best_mlp_preprocessor.joblib"
        model.save(model_path)
        joblib.dump(preprocessor, preprocessor_path)
        payload = {
            **row,
            "config": config.to_row(),
            "model_path": str(model_path),
            "preprocessor_path": str(preprocessor_path),
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n", encoding="utf-8")


def row_is_better(row: dict[str, object], current: dict[str, object]) -> bool:
    return row_selection_score(row) > row_selection_score(current)


def row_selection_score(row: dict[str, object]) -> tuple[float, ...]:
    return tuple(float(row.get(column, float("-inf"))) for column in SELECTION_COLUMNS)


def default_epochs(profile: str, fallback: int) -> int:
    return {"smoke": 12, "balanced": max(fallback, 80), "rtx3090": max(fallback, 140)}[profile]


def default_patience(profile: str, fallback: int) -> int:
    return {"smoke": 4, "balanced": max(fallback, 10), "rtx3090": max(fallback, 18)}[profile]


def default_reduce_patience(profile: str, fallback: int) -> int:
    return {"smoke": 2, "balanced": max(fallback, 5), "rtx3090": max(fallback, 7)}[profile]


def render_summary(ranking: pd.DataFrame, summary: dict[str, object]) -> str:
    lines = [
        "# Super hypertraining",
        "",
        f"Perfil: `{summary['profile']}`",
        f"Modo: `{summary['mode']}`",
        f"Seed: `{summary['seed']}`",
        f"Vistas: {', '.join(summary['views'])}",
        "",
        "Seleccion por validacion; test solo se reporta.",
        "",
        "## Top validacion",
        "",
    ]
    if ranking.empty:
        lines.append("_Sin resultados._")
    else:
        display = ranking.head(20).copy()
        for col in display.columns:
            if pd.api.types.is_numeric_dtype(display[col]):
                display[col] = display[col].map(lambda value: f"{value:.4f}")
        lines.append(markdown_table(display))
    lines.append("")
    return "\n".join(lines)


def markdown_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        headers = list(df.columns)
        rows = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for _, row in df.iterrows():
            rows.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
        return "\n".join(rows)


def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


if __name__ == "__main__":
    raise SystemExit(main())
