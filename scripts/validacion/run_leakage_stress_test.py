#!/usr/bin/env python
"""Stress test de fuga por variable, grupos y ablations.

Este experimento no forma parte del pipeline oficial. Entrena modelos
controlados para detectar columnas con senal anormalmente alta, posible fuga
temporal o dependencia de variables posteriores al diagnostico.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cancer_ml.config import (  # noqa: E402
    DEFAULT_SEED,
    ID_COLUMN,
    KNOWN_CATEGORICAL_COLUMNS,
    KNOWN_HIGH_RISK_POST_DIAGNOSIS_COLUMNS,
    KNOWN_SUSPECT_LEAKAGE_COLUMNS,
    METRICS_DIR,
    MODE_CONFIGS,
    TARGET_COLUMN,
    set_global_seed,
)
from cancer_ml.data import join_collections, load_available_csvs  # noqa: E402
from cancer_ml.evaluation import evaluate_probabilities, threshold_search  # noqa: E402
from cancer_ml.features import FeaturePolicy, build_feature_policy, build_preprocessor  # noqa: E402


BIOCHEMICAL_COLUMNS = [
    "glucosa",
    "colesterol",
    "trigliceridos",
    "hemoglobina",
    "leucocitos",
    "plaquetas",
    "creatinina",
]
CLINICAL_COLUMNS = ["diabetes", "hipertension", "obesidad", "enfermedad_cardiaca", "asma", "epoc"]
GENETIC_COLUMNS = ["mut_BRCA1", "mut_TP53", "mut_EGFR", "mut_KRAS", "mut_PIK3CA", "mut_ALK", "mut_BRAF"]
GENERAL_CLEAN_COLUMNS = ["fumador", "actividad_fisica"]
SOCIODEMOGRAPHIC_COLUMNS = [
    "edad",
    "nivel_educativo",
    "nivel_ingresos",
    "zona",
    "estado_civil",
    "num_hijos",
    "distancia_hospital_km",
]
ECONOMIC_COLUMNS = ["tipo_seguro", "coste_total", "coste_farmaco", "num_ingresos", "dias_hospital"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audita fuga de variables mediante entrenamiento controlado.")
    parser.add_argument("--mode", choices=sorted(MODE_CONFIGS), default="quick")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=METRICS_DIR / "leakage_stress_test",
        help="Directorio donde guardar resultados del stress test.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help="Iteraciones HGB. Por defecto usa el modo elegido.",
    )
    parser.add_argument(
        "--skip-base-minus",
        action="store_true",
        help="Omite ablations base - variable para acelerar.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_global_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames, missing = load_available_csvs()
    dataset = join_collections(frames)
    split_indices = make_split_indices(dataset, args.seed)
    scenarios = build_scenarios(dataset, include_base_minus=not args.skip_base_minus)

    mode_config = MODE_CONFIGS[args.mode]
    max_iter = args.max_iter or mode_config.hgb_iter
    fit_indices = split_indices["train_inner"]
    if mode_config.train_sample_size and mode_config.train_sample_size < len(fit_indices):
        fit_indices, _ = train_test_split(
            fit_indices,
            train_size=mode_config.train_sample_size,
            stratify=dataset.loc[fit_indices, TARGET_COLUMN],
            random_state=args.seed,
        )

    rows = []
    for i, scenario in enumerate(scenarios, start=1):
        print(f"[{i:03d}/{len(scenarios):03d}] {scenario['name']} ({len(scenario['features'])} vars)")
        row = run_scenario(
            dataset=dataset,
            scenario=scenario,
            fit_indices=np.asarray(fit_indices),
            valid_indices=np.asarray(split_indices["valid"]),
            test_indices=np.asarray(split_indices["test"]),
            max_iter=max_iter,
            seed=args.seed,
        )
        rows.append(row)

    results = pd.DataFrame(rows)
    results = add_risk_columns(results)
    results = results.sort_values(["risk_rank", "f1_positive", "auc_pr"], ascending=[False, False, False])
    results_path = args.output_dir / "leakage_stress_results.csv"
    results.to_csv(results_path, index=False)

    summary = build_summary(results, missing, dataset.shape, args)
    summary_path = args.output_dir / "leakage_stress_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )
    markdown_path = args.output_dir / "leakage_stress_summary.md"
    markdown_path.write_text(render_markdown(results, summary), encoding="utf-8")

    print(f"Guardado {results_path}")
    print(f"Guardado {summary_path}")
    print(f"Guardado {markdown_path}")
    print(
        results.head(20)[
            [
                "scenario_type",
                "name",
                "risk_label",
                "f1_positive",
                "recall_positive",
                "precision_positive",
                "auc_pr",
                "delta_f1_vs_base",
            ]
        ]
    )
    return 0


def make_split_indices(df: pd.DataFrame, seed: int) -> dict[str, np.ndarray]:
    y = df[TARGET_COLUMN].astype(int)
    train_indices, test_indices = train_test_split(
        df.index.to_numpy(),
        test_size=0.20,
        random_state=seed,
        stratify=y,
    )
    train_inner_indices, valid_indices = train_test_split(
        train_indices,
        test_size=0.20,
        random_state=seed,
        stratify=y.loc[train_indices],
    )
    return {
        "train": np.asarray(train_indices),
        "train_inner": np.asarray(train_inner_indices),
        "valid": np.asarray(valid_indices),
        "test": np.asarray(test_indices),
    }


def build_scenarios(df: pd.DataFrame, include_base_minus: bool = True) -> list[dict[str, object]]:
    metadata_policy = build_feature_policy(df, "metadata_core")
    base_policy = build_feature_policy(df, "base")
    safe_policy = build_feature_policy(df, "safe_all")
    economic_policy = build_feature_policy(df, "economic_sensitivity")
    all_predictors = [col for col in df.columns if col not in {ID_COLUMN, TARGET_COLUMN}]
    reference_features = list(metadata_policy.included)

    scenarios: list[dict[str, object]] = [
        scenario("group", "metadata_core_clean", reference_features, "Modelo limpio oficial segun metadato."),
        scenario("group", "base_clean", base_policy.included, "Modelo compacto historico."),
        scenario("group", "safe_all_clean", safe_policy.included, "Variables no fuga mas cautela/opcionales."),
        scenario("group", "economic_sensitivity", economic_policy.included, "Sensibilidad con costes/uso hospitalario."),
        scenario("group", "all_raw_including_risks", all_predictors, "Todas las variables crudas salvo id/target."),
        scenario("group", "economic_only", ECONOMIC_COLUMNS, "CSV economico completo."),
        scenario("group", "economic_cost_use_only", KNOWN_HIGH_RISK_POST_DIAGNOSIS_COLUMNS, "Costes y uso hospitalario."),
        scenario("group", "vive_only", ["vive"], "Fuga temporal conocida."),
        scenario("group", "known_risk_only", ["vive", *KNOWN_HIGH_RISK_POST_DIAGNOSIS_COLUMNS], "Fugas conocidas/sospechadas."),
        scenario("group", "biochemical_only", BIOCHEMICAL_COLUMNS, "Bioquimica."),
        scenario("group", "clinical_only", CLINICAL_COLUMNS, "Clinicas binarias."),
        scenario("group", "genetic_only", GENETIC_COLUMNS, "Geneticas binarias."),
        scenario("group", "general_clean_only", GENERAL_CLEAN_COLUMNS, "Habitos limpios."),
        scenario("group", "sociodemographic_only", SOCIODEMOGRAPHIC_COLUMNS, "Sociodemograficas/proxies."),
    ]

    for column in all_predictors:
        scenarios.append(scenario("single", f"single::{column}", [column], "Variable individual."))

    for column in all_predictors:
        if column not in reference_features:
            scenarios.append(
                scenario(
                    "base_plus",
                    f"base_plus::{column}",
                    [*reference_features, column],
                    "Vista metadata_core mas una variable excluida/sospechosa.",
                )
            )

    if include_base_minus:
        for column in reference_features:
            scenarios.append(
                scenario(
                    "base_minus",
                    f"base_minus::{column}",
                    [feature for feature in reference_features if feature != column],
                    "Ablation de una variable incluida en metadata_core.",
                )
            )

    return [clean_scenario_features(item, df) for item in scenarios]


def scenario(scenario_type: str, name: str, features: list[str], notes: str) -> dict[str, object]:
    return {
        "scenario_type": scenario_type,
        "name": name,
        "features": features,
        "notes": notes,
    }


def clean_scenario_features(scenario_def: dict[str, object], df: pd.DataFrame) -> dict[str, object]:
    seen = set()
    features = []
    for feature in scenario_def["features"]:
        if feature in df.columns and feature not in seen and feature not in {ID_COLUMN, TARGET_COLUMN}:
            seen.add(feature)
            features.append(feature)
    scenario_def["features"] = features
    return scenario_def


def run_scenario(
    dataset: pd.DataFrame,
    scenario: dict[str, object],
    fit_indices: np.ndarray,
    valid_indices: np.ndarray,
    test_indices: np.ndarray,
    max_iter: int,
    seed: int,
) -> dict[str, object]:
    features = list(scenario["features"])
    policy = infer_policy(dataset, features, str(scenario["name"]))
    preprocessor = build_preprocessor(policy)
    estimator = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=max_iter,
        max_leaf_nodes=31,
        l2_regularization=0.01,
        class_weight="balanced",
        early_stopping=True,
        random_state=seed,
    )
    model = Pipeline([("preprocess", preprocessor), ("model", estimator)])
    X_fit = dataset.loc[fit_indices, features]
    y_fit = dataset.loc[fit_indices, TARGET_COLUMN].astype(int)
    X_valid = dataset.loc[valid_indices, features]
    y_valid = dataset.loc[valid_indices, TARGET_COLUMN].astype(int)
    X_test = dataset.loc[test_indices, features]
    y_test = dataset.loc[test_indices, TARGET_COLUMN].astype(int)

    model.fit(X_fit, y_fit)
    valid_proba = model.predict_proba(X_valid)[:, 1]
    threshold, _ = threshold_search(y_valid, valid_proba, None)
    test_proba = model.predict_proba(X_test)[:, 1]
    result = evaluate_probabilities(str(scenario["name"]), str(scenario["scenario_type"]), y_test, test_proba, threshold)
    row = result.as_row()
    row.update(
        {
            "scenario_type": scenario["scenario_type"],
            "name": scenario["name"],
            "notes": scenario["notes"],
            "feature_count": len(features),
            "features": ";".join(features),
            "contains_vive": "vive" in features,
            "contains_post_diagnosis_risk": bool(set(features) & set(KNOWN_HIGH_RISK_POST_DIAGNOSIS_COLUMNS)),
            "contains_any_known_risk": bool(
                set(features) & set([*KNOWN_SUSPECT_LEAKAGE_COLUMNS, *KNOWN_HIGH_RISK_POST_DIAGNOSIS_COLUMNS])
            ),
            "numeric_count": len(policy.numeric),
            "binary_count": len(policy.binary),
            "categorical_count": len(policy.categorical),
        }
    )
    return row


def infer_policy(df: pd.DataFrame, features: list[str], name: str) -> FeaturePolicy:
    categorical = [col for col in KNOWN_CATEGORICAL_COLUMNS if col in features]
    numeric: list[str] = []
    binary: list[str] = []
    for column in features:
        if column in categorical:
            continue
        series = df[column].dropna()
        if pd.api.types.is_numeric_dtype(df[column]):
            values = set(series.unique().tolist())
            if values.issubset({0, 1}):
                binary.append(column)
            else:
                numeric.append(column)
        else:
            categorical.append(column)
    included = numeric + binary + categorical
    return FeaturePolicy(
        target=TARGET_COLUMN,
        feature_view=name,
        excluded=[col for col in df.columns if col not in included and col != TARGET_COLUMN],
        numeric=numeric,
        binary=binary,
        categorical=categorical,
        engineered=[],
        included=included,
        notes=[f"Stress test scenario: {name}"],
    )


def add_risk_columns(results: pd.DataFrame) -> pd.DataFrame:
    base_rows = results[results["name"] == "metadata_core_clean"]
    if not len(base_rows):
        base_rows = results[results["name"] == "base_clean"]
    base_f1 = float(base_rows.iloc[0]["f1_positive"]) if len(base_rows) else np.nan
    base_f1 = float(base_rows.iloc[0]["f1_positive"]) if len(base_rows) else np.nan
    base_auc_pr = float(base_rows.iloc[0]["auc_pr"]) if len(base_rows) else np.nan
    results = results.copy()
    results["delta_f1_vs_base"] = results["f1_positive"] - base_f1
    results["delta_f1_vs_base"] = results["f1_positive"] - base_f1
    results["delta_auc_pr_vs_base"] = results["auc_pr"] - base_auc_pr
    risk_labels = []
    risk_ranks = []
    for _, row in results.iterrows():
        label, rank = classify_risk(row)
        risk_labels.append(label)
        risk_ranks.append(rank)
    results["risk_label"] = risk_labels
    results["risk_rank"] = risk_ranks
    return results


def classify_risk(row: pd.Series) -> tuple[str, int]:
    f1 = float(row["f1_positive"])
    f1 = float(row["f1_positive"])
    auc = float(row["auc_roc"])
    delta_f1 = float(row.get("delta_f1_vs_base", 0.0))
    delta_f1 = float(row.get("delta_f1_vs_base", 0.0))
    contains_known_risk = bool(row.get("contains_any_known_risk", False))
    contains_post = bool(row.get("contains_post_diagnosis_risk", False))
    contains_vive = bool(row.get("contains_vive", False))

    if contains_post and (f1 >= 0.80 or f1 >= 0.80 or auc >= 0.95 or delta_f1 >= 0.15 or delta_f1 >= 0.15):
        return "critical_post_diagnosis_signature", 5
    if f1 >= 0.80 or f1 >= 0.80 or auc >= 0.95:
        return "critical_performance_anomaly", 5
    if contains_vive and (delta_f1 >= 0.02 or delta_f1 >= 0.02 or auc >= 0.75):
        return "high_known_temporal_leakage", 4
    if contains_known_risk:
        return "high_prior_risk_review", 3
    if delta_f1 >= 0.02 or delta_f1 >= 0.02:
        return "moderate_unexpected_gain", 2
    if auc >= 0.85:
        return "moderate_strong_signal", 2
    return "low_no_leakage_signature", 1


def build_summary(results: pd.DataFrame, missing: list[str], dataset_shape: tuple[int, int], args: argparse.Namespace) -> dict:
    top = results.sort_values(["f1_positive", "auc_pr"], ascending=False).head(10)
    base_plus = (
        results[results["scenario_type"] == "base_plus"].sort_values("delta_f1_vs_base", ascending=False).head(10)
    )
    singles = results[results["scenario_type"] == "single"].sort_values(["f1_positive", "auc_pr"], ascending=False).head(10)
    return {
        "mode": args.mode,
        "seed": args.seed,
        "dataset_shape": list(dataset_shape),
        "missing_csvs": missing,
        "scenarios_evaluated": int(len(results)),
        "metadata_core_clean": _first_row(results[results["name"] == "metadata_core_clean"]),
        "base_clean": _first_row(results[results["name"] == "base_clean"]),
        "risk_counts": results["risk_label"].value_counts().to_dict(),
        "top_scenarios": top[["scenario_type", "name", "risk_label", "f1_positive", "recall_positive", "auc_pr"]].to_dict(
            orient="records"
        ),
        "top_base_plus": base_plus[
            ["name", "risk_label", "delta_f1_vs_base", "f1_positive", "recall_positive", "auc_pr"]
        ].to_dict(orient="records"),
        "top_single_variables": singles[
            ["name", "risk_label", "f1_positive", "recall_positive", "auc_pr", "features"]
        ].to_dict(orient="records"),
    }


def render_markdown(results: pd.DataFrame, summary: dict) -> str:
    lines = [
        "# Leakage stress test",
        "",
        f"Escenarios evaluados: {summary['scenarios_evaluated']}",
        f"Dataset: {summary['dataset_shape'][0]} filas x {summary['dataset_shape'][1]} columnas",
        f"Modo: `{summary['mode']}`, seed: `{summary['seed']}`",
        "",
        "## Lectura ejecutiva",
        "",
        "- Las variables marcadas como `critical_post_diagnosis_signature` no deben usarse en el modelo limpio.",
        "- `base_plus` mide cuanto cambia el modelo oficial al anadir una variable excluida.",
        "- `single` mide si una variable sola ya predice demasiado bien el target.",
        "",
        "## Conteo de riesgo",
        "",
        _markdown_table(pd.DataFrame([{"risk_label": key, "count": value} for key, value in summary["risk_counts"].items()])),
        "",
        "## Top escenarios por F1 test",
        "",
        _markdown_table(
            results.sort_values(["f1_positive", "auc_pr"], ascending=False)
            .head(15)[
                [
                    "scenario_type",
                    "name",
                    "risk_label",
                    "f1_positive",
                    "recall_positive",
                    "precision_positive",
                    "auc_pr",
                    "delta_f1_vs_base",
                ]
            ]
        ),
        "",
        "## Top variables individuales",
        "",
        _markdown_table(
            results[results["scenario_type"] == "single"]
            .sort_values(["f1_positive", "auc_pr"], ascending=False)
            .head(15)[["name", "risk_label", "f1_positive", "recall_positive", "precision_positive", "auc_pr", "features"]]
        ),
        "",
        "## Mayor ganancia al anadir a base",
        "",
        _markdown_table(
            results[results["scenario_type"] == "base_plus"]
            .sort_values("delta_f1_vs_base", ascending=False)
            .head(15)[["name", "risk_label", "delta_f1_vs_base", "f1_positive", "recall_positive", "auc_pr"]]
        ),
        "",
    ]
    return "\n".join(lines)


def _first_row(df: pd.DataFrame) -> dict[str, object]:
    if df.empty:
        return {}
    row = df.iloc[0]
    return {
        key: value
        for key, value in row.to_dict().items()
        if key
        in {
            "name",
            "f1_positive",
            "auc_roc",
            "auc_pr",
            "precision_positive",
            "recall_positive",
            "threshold",
        }
    }


def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_Sin filas._"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_numeric_dtype(display[col]):
            display[col] = display[col].map(lambda value: f"{value:.4f}")
    try:
        return display.to_markdown(index=False)
    except Exception:
        headers = list(display.columns)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for _, row in display.iterrows():
            lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
        return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
