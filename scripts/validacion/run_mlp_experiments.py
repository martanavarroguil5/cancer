#!/usr/bin/env python
"""Bateria opcional y reproducible de experimentos para arquitecturas MLP.

Este script no forma parte del pipeline principal ni de los resultados finales.
Sirve para investigar variantes y dejar evidencia auditable antes de decidir si
un cambio merece entrar en `scripts/run_pipeline.py`.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from itertools import combinations
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.utils.class_weight import compute_class_weight

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cancer_ml.config import (  # noqa: E402
    DEFAULT_SEED,
    METRICS_DIR,
    MODE_CONFIGS,
    MODELS_DIR,
    ensure_directories,
    set_global_seed,
)
from cancer_ml.data import join_collections, load_available_csvs  # noqa: E402
from cancer_ml.evaluation import evaluate_probabilities, threshold_search  # noqa: E402
from cancer_ml.features import build_feature_policy, build_preprocessor, split_data  # noqa: E402
from cancer_ml.models_mlp import (  # noqa: E402
    DEFAULT_MLP_CONFIG,
    MLPConfig,
    ValidationThresholdMetrics,
    _as_float32,
    _monitor_should_maximize,
    build_mlp,
)


LOGGER = logging.getLogger("run_mlp_experiments")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Busca mejores MLPs usando solo validacion para seleccionar.")
    parser.add_argument("--mode", choices=sorted(MODE_CONFIGS), default="full")
    parser.add_argument("--suite", choices=["smoke", "broad", "refine"], default="broad")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--reduce-patience", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--names",
        default="",
        help="Lista separada por comas para filtrar candidatos por nombre exacto.",
    )
    parser.add_argument("--verbose-fit", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=METRICS_DIR / "mlp_experiment_results.csv",
        help="CSV de ranking por F1 de validacion.",
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
    ensure_directories()
    set_global_seed(args.seed)

    mode_config = MODE_CONFIGS[args.mode]
    mode_config = replace(
        mode_config,
        mlp_epochs=args.epochs or mode_config.mlp_epochs,
        mlp_patience=args.patience or mode_config.mlp_patience,
        mlp_reduce_patience=args.reduce_patience or mode_config.mlp_reduce_patience,
    )
    candidates = candidate_configs(args.suite)
    if args.names.strip():
        wanted = {name.strip() for name in args.names.split(",") if name.strip()}
        candidates = [config for config in candidates if config.name in wanted]
        missing = sorted(wanted - {config.name for config in candidates})
        if missing:
            raise ValueError(f"Candidatos no encontrados en suite {args.suite}: {missing}")
    if args.limit:
        candidates = candidates[: args.limit]

    frames, _ = load_available_csvs()
    dataset = join_collections(frames)
    policy = build_feature_policy(dataset)
    splits = split_data(dataset, policy, seed=args.seed)
    preprocessor = build_preprocessor(policy)
    X_train, y_train, X_valid, y_valid, X_test, y_test = transform_splits(preprocessor, splits)
    class_weight = balanced_class_weight(y_train)

    histories_dir = args.output.parent / "mlp_experiment_histories"
    histories_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    valid_probas: dict[str, np.ndarray] = {}
    test_probas: dict[str, np.ndarray] = {}
    best_valid_f1 = -np.inf
    best_name = ""
    for index, config in enumerate(candidates, start=1):
        LOGGER.info("[%d/%d] Entrenando %s.", index, len(candidates), config.name)
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
        except Exception as exc:
            LOGGER.exception("%s fallo.", config.name)
            row = {
                "candidate": config.name,
                "status": "failed",
                "error": repr(exc),
                **config.to_row(),
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(args.output, index=False)
            continue

        history.to_csv(histories_dir / f"{config.name}.csv", index=False)
        valid_probas[config.name] = valid_proba
        test_probas[config.name] = test_proba
        rows.append(row)
        if row["valid_f1_positive"] > best_valid_f1:
            best_valid_f1 = row["valid_f1_positive"]
            best_name = config.name
            model.save(MODELS_DIR / "MLP_best_experiment.keras")
            (MODELS_DIR / "MLP_best_experiment_config.json").write_text(
                json.dumps(config.to_row(), indent=2) + "\n",
                encoding="utf-8",
            )
        pd.DataFrame(rows).sort_values(
            ["valid_f1_positive", "valid_recall_positive", "valid_auc_pr", "valid_auc_roc"],
            ascending=False,
        ).to_csv(args.output, index=False)
        LOGGER.info(
            "%s | valid_f1=%.4f test_f1=%.4f recall=%.4f threshold=%.2f epochs=%d params=%d.",
            config.name,
            row["valid_f1_positive"],
            row["test_f1_positive"],
            row["test_recall_positive"],
            row["valid_threshold"],
            row["epochs_ran"],
            row["params"],
        )

    ranking = pd.DataFrame(rows)
    sort_columns = [
        column for column in ["valid_f1_positive", "valid_recall_positive", "valid_auc_pr", "valid_auc_roc"] if column in ranking
    ]
    if sort_columns:
        ranking = ranking.sort_values(sort_columns, ascending=False)
    ranking.to_csv(args.output, index=False)
    ensemble_path = args.output.with_name(f"{args.output.stem}_ensembles.csv")
    ensemble_ranking = build_ensemble_ranking(
        ranking,
        valid_probas,
        test_probas,
        y_valid,
        y_test,
        output_path=ensemble_path,
    )
    if not ensemble_ranking.empty:
        best_ensemble = ensemble_ranking.iloc[0]
        LOGGER.info(
            "Mejor ensemble MLP por validacion: %s valid_f1=%.4f test_f1=%.4f. CSV=%s",
            best_ensemble["members"],
            best_ensemble["valid_f1_positive"],
            best_ensemble["test_f1_positive"],
            ensemble_path,
        )
    LOGGER.info("Mejor por validacion: %s (F1 %.4f). CSV=%s", best_name, best_valid_f1, args.output)
    return 0


def transform_splits(preprocessor, splits: dict[str, pd.DataFrame | pd.Series]):
    transformer = clone(preprocessor)
    X_train = _as_float32(transformer.fit_transform(splits["X_train_inner"]))
    X_valid = _as_float32(transformer.transform(splits["X_valid"]))
    X_test = _as_float32(transformer.transform(splits["X_test"]))
    return (
        X_train,
        np.asarray(splits["y_train_inner"]).astype(int),
        X_valid,
        np.asarray(splits["y_valid"]).astype(int),
        X_test,
        np.asarray(splits["y_test"]).astype(int),
    )


def balanced_class_weight(y_train: np.ndarray) -> dict[int, float]:
    weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y_train)
    return {0: float(weights[0]), 1: float(weights[1])}


def train_candidate(
    config: MLPConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_weight: dict[int, float],
    mode_config,
    seed: int,
    verbose_fit: int,
) -> tuple[dict[str, object], pd.DataFrame, object, np.ndarray, np.ndarray]:
    import tensorflow as tf

    set_global_seed(seed)
    tf.keras.backend.clear_session()

    model = build_mlp(X_train.shape[1], seed, config)
    batch_size = config.batch_size or mode_config.batch_size
    monitor_mode = "max" if _monitor_should_maximize(config.monitor) else "min"
    callbacks = [
        ValidationThresholdMetrics(X_valid, y_valid, batch_size=batch_size),
        tf.keras.callbacks.EarlyStopping(
            monitor=config.monitor,
            mode=monitor_mode,
            patience=mode_config.mlp_patience,
            min_delta=1e-4,
            restore_best_weights=True,
            start_from_epoch=config.warmup_epochs,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=config.monitor,
            mode=monitor_mode,
            factor=0.5,
            patience=mode_config.mlp_reduce_patience,
            min_lr=1e-5,
        ),
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_valid, y_valid),
        epochs=mode_config.mlp_epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=verbose_fit,
    )
    history_df = pd.DataFrame(history.history)
    valid_proba = model.predict(X_valid, batch_size=batch_size, verbose=0).ravel()
    test_proba = model.predict(X_test, batch_size=batch_size, verbose=0).ravel()
    valid_threshold, _ = threshold_search(y_valid, valid_proba, None)
    valid_eval = evaluate_probabilities(
        f"{config.name}_valid",
        "MLPExperiment",
        y_valid,
        valid_proba,
        valid_threshold,
    )
    test_eval = evaluate_probabilities(
        config.name,
        "MLPExperiment",
        y_test,
        test_proba,
        valid_threshold,
    )
    best_epoch_metric = "val_f1_best"
    best_epoch = int(history_df[best_epoch_metric].idxmax() + 1) if best_epoch_metric in history_df else len(history_df)
    row = {
        "candidate": config.name,
        "status": "ok",
        "valid_threshold": float(valid_threshold),
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
        "epochs_ran": int(len(history_df)),
        "best_epoch_by_valid_f1": best_epoch,
        "params": int(model.count_params()),
        "final_learning_rate": float(tf.keras.backend.get_value(model.optimizer.learning_rate)),
        **config.to_row(),
    }
    row["valid_f1_from_history"] = float(history_df["val_f1_best"].max()) if "val_f1_best" in history_df else np.nan
    row["test_valid_gap_f1"] = row["valid_f1_positive"] - row["test_f1_positive"]
    return row, history_df, model, valid_proba, test_proba


def build_ensemble_ranking(
    ranking: pd.DataFrame,
    valid_probas: dict[str, np.ndarray],
    test_probas: dict[str, np.ndarray],
    y_valid: np.ndarray,
    y_test: np.ndarray,
    output_path: Path,
    max_members: int = 5,
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
                "MLPEnsemble",
                y_valid,
                valid_proba,
                threshold,
            )
            test_eval = evaluate_probabilities(
                f"ensemble_{size}",
                "MLPEnsemble",
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
        ensemble_ranking = ensemble_ranking.sort_values(
            ["valid_f1_positive", "valid_recall_positive", "valid_auc_pr", "valid_auc_roc"],
            ascending=False,
        )
        ensemble_ranking.to_csv(output_path, index=False)
    return ensemble_ranking


def candidate_configs(suite: str) -> list[MLPConfig]:
    smoke = [
        DEFAULT_MLP_CONFIG,
        MLPConfig(
            name="baseline_128_64_32_adam",
            hidden_units=(128, 64, 32),
            dropout=(0.25, 0.25, 0.20),
            optimizer="adam",
            learning_rate=1e-3,
            weight_decay=0.0,
            batch_size=512,
            clipnorm=None,
        ),
    ]
    if suite == "smoke":
        return smoke

    broad = [
        *smoke,
        MLPConfig(
            name="dense_256_128_64_low_dropout",
            hidden_units=(256, 128, 64),
            dropout=(0.12, 0.10, 0.08),
            l2=1e-5,
            learning_rate=8e-4,
            weight_decay=5e-5,
            batch_size=512,
        ),
        MLPConfig(
            name="dense_512_256_128_64",
            hidden_units=(512, 256, 128, 64),
            dropout=(0.18, 0.16, 0.14, 0.10),
            l2=1e-5,
            learning_rate=7e-4,
            weight_decay=1e-4,
            batch_size=512,
        ),
        MLPConfig(
            name="dense_512_256_128_64_lr3e4",
            hidden_units=(512, 256, 128, 64),
            dropout=(0.16, 0.14, 0.12, 0.10),
            l2=3e-6,
            learning_rate=3e-4,
            weight_decay=5e-5,
            batch_size=512,
        ),
        MLPConfig(
            name="dense_1024_512_256_128",
            hidden_units=(1024, 512, 256, 128),
            dropout=(0.20, 0.18, 0.15, 0.12),
            l2=3e-6,
            learning_rate=5e-4,
            weight_decay=1e-4,
            batch_size=512,
        ),
        MLPConfig(
            name="deep_512_512_256_128_64",
            hidden_units=(512, 512, 256, 128, 64),
            dropout=(0.20, 0.18, 0.16, 0.12, 0.10),
            l2=1e-5,
            learning_rate=5e-4,
            weight_decay=1e-4,
            batch_size=512,
        ),
        MLPConfig(
            name="deep_1024_512_512_256_128",
            hidden_units=(1024, 512, 512, 256, 128),
            dropout=(0.22, 0.20, 0.18, 0.14, 0.10),
            l2=3e-6,
            learning_rate=4e-4,
            weight_decay=1e-4,
            batch_size=512,
        ),
        MLPConfig(
            name="huge_2048_1024_512_256",
            hidden_units=(2048, 1024, 512, 256),
            dropout=(0.25, 0.22, 0.18, 0.14),
            l2=1e-6,
            learning_rate=3e-4,
            weight_decay=5e-5,
            batch_size=512,
        ),
        MLPConfig(
            name="swish_512_256_128_64",
            hidden_units=(512, 256, 128, 64),
            activation="swish",
            dropout=(0.18, 0.16, 0.12, 0.10),
            l2=1e-5,
            learning_rate=5e-4,
            weight_decay=1e-4,
            batch_size=512,
        ),
        MLPConfig(
            name="gelu_512_256_128_64",
            hidden_units=(512, 256, 128, 64),
            activation="gelu",
            dropout=(0.18, 0.16, 0.12, 0.10),
            l2=1e-5,
            learning_rate=5e-4,
            weight_decay=1e-4,
            batch_size=512,
        ),
        MLPConfig(
            name="selu_256_128_64_no_bn",
            hidden_units=(256, 128, 64),
            activation="selu",
            dropout=(0.08, 0.06, 0.04),
            batch_norm=False,
            l2=1e-6,
            learning_rate=6e-4,
            weight_decay=1e-5,
            batch_size=512,
        ),
        MLPConfig(
            name="dense_256_128_64_no_bn",
            hidden_units=(256, 128, 64),
            dropout=(0.12, 0.10, 0.08),
            batch_norm=False,
            l2=1e-5,
            learning_rate=7e-4,
            weight_decay=1e-4,
            batch_size=512,
        ),
        MLPConfig(
            name="residual_256_256_256_256",
            hidden_units=(256, 256, 256, 256),
            block_type="residual",
            dropout=0.12,
            l2=1e-5,
            learning_rate=5e-4,
            weight_decay=1e-4,
            batch_size=512,
        ),
        MLPConfig(
            name="residual_512_512_512_512",
            hidden_units=(512, 512, 512, 512),
            block_type="residual",
            dropout=0.14,
            l2=3e-6,
            learning_rate=3e-4,
            weight_decay=1e-4,
            batch_size=512,
        ),
        MLPConfig(
            name="residual_512_256_128_64",
            hidden_units=(512, 256, 128, 64),
            block_type="residual",
            dropout=(0.16, 0.14, 0.12, 0.10),
            l2=1e-5,
            learning_rate=4e-4,
            weight_decay=1e-4,
            batch_size=512,
        ),
        MLPConfig(
            name="dense_512_256_128_batch1024",
            hidden_units=(512, 256, 128),
            dropout=(0.16, 0.14, 0.10),
            l2=1e-5,
            learning_rate=8e-4,
            weight_decay=1e-4,
            batch_size=1024,
        ),
        MLPConfig(
            name="dense_256_128_64_batch256",
            hidden_units=(256, 128, 64),
            dropout=(0.16, 0.14, 0.10),
            l2=1e-5,
            learning_rate=5e-4,
            weight_decay=1e-4,
            batch_size=256,
        ),
    ]
    if suite == "broad":
        return broad

    return [
        config
        for config in broad
        if config.name
        in {
            "dense_256_128_64_adamw_f1",
            "dense_256_128_64_low_dropout",
            "dense_512_256_128_64",
            "dense_512_256_128_64_lr3e4",
            "dense_1024_512_256_128",
            "deep_512_512_256_128_64",
            "swish_512_256_128_64",
            "residual_256_256_256_256",
            "residual_512_256_128_64",
        }
    ]


if __name__ == "__main__":
    raise SystemExit(main())
