#!/usr/bin/env python
"""Benchmark opcional de AutoGluon contra el split oficial.

Este script no forma parte del pipeline principal. Sirve para comprobar si un
AutoML tabular moderno encuentra señal adicional sin usar `vive` ni el test para
seleccionar modelos o umbrales.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cancer_ml.config import DEFAULT_SEED, OUTPUT_DIR, RECOMMENDED_FEATURE_VIEW, TARGET_COLUMN, set_global_seed  # noqa: E402
from cancer_ml.data import join_collections, load_available_csvs  # noqa: E402
from cancer_ml.evaluation import evaluate_probabilities, threshold_search  # noqa: E402
from cancer_ml.features import (  # noqa: E402
    FEATURE_VIEWS,
    add_engineered_features,
    build_feature_policy,
    split_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecuta benchmark opcional de AutoGluon.")
    parser.add_argument("--feature-view", choices=FEATURE_VIEWS, default=RECOMMENDED_FEATURE_VIEW)
    parser.add_argument("--preset", default="best_v150")
    parser.add_argument("--time-limit", type=int, default=900)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR / "autogluon",
        help="Directorio raiz de salidas ignoradas por git.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Borra la salida previa de esta configuracion antes de entrenar.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_global_seed(args.seed)

    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as exc:
        raise SystemExit(
            "AutoGluon no esta instalado. Usa un entorno separado, por ejemplo:\n"
            "python3 -m venv .venv-autogluon\n"
            "source .venv-autogluon/bin/activate\n"
            "python -m pip install -r requirements-autogluon.txt"
        ) from exc

    frames, missing = load_available_csvs()
    dataset = join_collections(frames)
    modeling_dataset = (
        add_engineered_features(dataset)
        if args.feature_view == "engineered_selected"
        else dataset
    )
    policy = build_feature_policy(modeling_dataset, feature_view=args.feature_view)
    splits = split_data(modeling_dataset, policy, seed=args.seed)

    run_name = f"{args.feature_view}_{args.preset}_{args.time_limit}s"
    output_dir = args.output_dir / run_name
    predictor_dir = output_dir / "predictor"
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    elif predictor_dir.exists():
        raise SystemExit(f"La salida {output_dir} ya existe. Usa --overwrite para regenerarla.")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = _with_label(splits["X_train_inner"], splits["y_train_inner"])
    valid_df = _with_label(splits["X_valid"], splits["y_valid"])
    test_df = splits["X_test"].copy()
    test_with_label = _with_label(splits["X_test"], splits["y_test"])

    predictor = TabularPredictor(
        label=TARGET_COLUMN,
        eval_metric="f1",
        path=str(predictor_dir),
        verbosity=2,
    ).fit(
        train_data=train_df,
        tuning_data=valid_df,
        use_bag_holdout=True,
        presets=args.preset,
        time_limit=args.time_limit,
        calibrate_decision_threshold="auto",
        num_gpus=args.num_gpus,
    )

    valid_proba = _positive_probability(predictor, valid_df.drop(columns=[TARGET_COLUMN]))
    test_proba = _positive_probability(predictor, test_df)
    threshold, _ = threshold_search(
        splits["y_valid"],
        valid_proba,
        output_dir / "threshold_search.csv",
    )
    result = evaluate_probabilities(
        f"AutoGluon_{args.preset}_{args.feature_view}",
        "AutoGluon",
        splits["y_test"],
        test_proba,
        threshold=threshold,
    )
    metrics = pd.DataFrame([result.as_row()]).drop(columns=["y_proba", "y_pred"], errors="ignore")
    metrics.to_csv(output_dir / "metrics.csv", index=False)

    leaderboard_valid = predictor.leaderboard(valid_df, silent=True)
    leaderboard_test = predictor.leaderboard(test_with_label, silent=True)
    leaderboard_valid.to_csv(output_dir / "leaderboard_valid.csv", index=False)
    leaderboard_test.to_csv(output_dir / "leaderboard_test.csv", index=False)

    summary = {
        "feature_view": args.feature_view,
        "preset": args.preset,
        "time_limit": args.time_limit,
        "seed": args.seed,
        "missing_csvs": missing,
        "feature_count": len(policy.included),
        "excluded": policy.excluded,
        "best_model": predictor.model_best,
        "autogluon_decision_threshold": float(predictor.decision_threshold),
        "repo_validation_threshold": float(threshold),
        "metrics": metrics.iloc[0].to_dict(),
        "leaderboard_valid_top5": leaderboard_valid.head(5).to_dict(orient="records"),
        "leaderboard_test_top5": leaderboard_test.head(5).to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(metrics.to_string(index=False))
    print(f"Guardado benchmark en {output_dir}")
    return 0


def _with_label(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    frame = X.copy()
    frame[TARGET_COLUMN] = pd.Series(y.to_numpy(), index=frame.index)
    return frame


def _positive_probability(predictor, X: pd.DataFrame):
    probabilities = predictor.predict_proba(X)
    positive_label = 1 if 1 in probabilities.columns else probabilities.columns[-1]
    return probabilities[positive_label].to_numpy()


if __name__ == "__main__":
    raise SystemExit(main())
