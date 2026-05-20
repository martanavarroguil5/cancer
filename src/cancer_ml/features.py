"""Politica de features, splits y preprocesamiento."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cancer_ml.config import (
    DEFAULT_SEED,
    ID_COLUMN,
    KNOWN_CATEGORICAL_COLUMNS,
    KNOWN_CONSTANT_CANDIDATES,
    KNOWN_HIGH_RISK_POST_DIAGNOSIS_COLUMNS,
    KNOWN_SUSPECT_LEAKAGE_COLUMNS,
    LOW_SIGNAL_OR_PROXY_COLUMNS,
    METADATA_CAUTION_COMORBIDITY_COLUMNS,
    METADATA_OPTIONAL_OR_PROXY_COLUMNS,
    METADATA_OPTIONAL_SOCIODEMOGRAPHIC_COLUMNS,
    METRICS_DIR,
    RECOMMENDED_FEATURE_VIEW,
    TARGET_COLUMN,
)

LOGGER = logging.getLogger(__name__)

FEATURE_VIEWS = ("metadata_core", "base", "safe_all", "engineered_selected", "economic_sensitivity")
GENETIC_COLUMNS = [
    "mut_BRCA1",
    "mut_TP53",
    "mut_EGFR",
    "mut_KRAS",
    "mut_PIK3CA",
    "mut_ALK",
    "mut_BRAF",
]
HIGH_RISK_GENETIC_COLUMNS = ["mut_BRCA1", "mut_TP53", "mut_KRAS", "mut_EGFR"]
COMORBIDITY_COLUMNS = ["diabetes", "hipertension", "obesidad", "enfermedad_cardiaca", "asma", "epoc"]
CORE_COMORBIDITY_COLUMNS = ["diabetes", "hipertension", "obesidad", "epoc"]
METABOLIC_COMORBIDITY_COLUMNS = ["diabetes", "hipertension", "obesidad"]

ALL_ENGINEERED_FEATURE_COLUMNS = [
    "actividad_baja",
    "actividad_moderada",
    "actividad_alta",
    "actividad_score",
    "edad_decada",
    "edad_sq",
    "edad_ge_50",
    "edad_ge_60",
    "edad_ge_65",
    "edad_ge_70",
    "edad_gt_55",
    "tyg_index",
    "trigliceridos_colesterol_ratio",
    "colesterol_trigliceridos_ratio",
    "hemoglobina_leucocitos_ratio",
    "glucosa_ge_100",
    "glucosa_ge_126",
    "glucosa_gt_130",
    "colesterol_ge_240",
    "trigliceridos_ge_150",
    "trigliceridos_ge_200",
    "hemoglobina_lt_12",
    "hemoglobina_lt_11",
    "hemoglobina_gt_16_5",
    "leucocitos_lt_4",
    "leucocitos_gt_10",
    "leucocitos_gt_11",
    "plaquetas_lt_150",
    "plaquetas_gt_400",
    "creatinina_gt_1_3",
    "metabolic_lab_count",
    "inflammation_count",
    "comorbidity_count",
    "core_comorbidity_count",
    "metabolic_syndrome_count",
    "mutation_count",
    "high_risk_mutation_count",
    "any_mutation",
    "multi_mutation",
    "tp53_or_kras",
    "brca1_or_tp53",
    "fumador_x_obesidad",
    "fumador_x_epoc",
    "fumador_x_mutation_count",
    "fumador_x_highrisk_mutation_count",
    "fumador_x_tp53",
    "fumador_x_kras",
    "obesidad_x_diabetes",
    "obesidad_x_hipertension",
    "obesidad_x_trigliceridos_altos",
    "actividad_baja_x_obesidad",
    "actividad_baja_x_fumador",
    "edad_x_mutation_count",
    "edad_x_high_risk_mutation_count",
    "edad_x_comorbidity_count",
    "educacion_score",
    "ingresos_score",
    "capital_social_score",
    "log1p_distancia_hospital_km",
    "hospital_lejano_50km",
    "hospital_lejano_100km",
]

SELECTED_ENGINEERED_FEATURE_COLUMNS = [
    "actividad_moderada",
    "actividad_alta",
    "actividad_score",
    "edad_decada",
    "edad_gt_55",
    "tyg_index",
    "trigliceridos_colesterol_ratio",
    "glucosa_ge_100",
    "glucosa_gt_130",
    "trigliceridos_ge_150",
    "trigliceridos_ge_200",
    "hemoglobina_lt_11",
    "leucocitos_gt_10",
    "metabolic_lab_count",
    "inflammation_count",
    "mutation_count",
    "high_risk_mutation_count",
    "multi_mutation",
    "tp53_or_kras",
    "brca1_or_tp53",
    "fumador_x_mutation_count",
    "fumador_x_highrisk_mutation_count",
    "fumador_x_tp53",
    "fumador_x_kras",
    "edad_x_high_risk_mutation_count",
]


@dataclass(frozen=True)
class FeaturePolicy:
    target: str
    feature_view: str
    excluded: list[str]
    numeric: list[str]
    binary: list[str]
    categorical: list[str]
    engineered: list[str]
    included: list[str]
    notes: list[str]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Anade features clinicas deterministas sin usar el target ni `vive`."""

    engineered = df.copy()
    if "actividad_fisica" in engineered.columns:
        engineered["actividad_baja"] = (engineered["actividad_fisica"] == "Baja").astype(int)
        engineered["actividad_moderada"] = (engineered["actividad_fisica"] == "Moderada").astype(int)
        engineered["actividad_alta"] = (engineered["actividad_fisica"] == "Alta").astype(int)
        engineered["actividad_score"] = (
            engineered["actividad_fisica"].map({"Baja": 0, "Moderada": 1, "Alta": 2}).astype(float)
        )

    if "edad" in engineered.columns:
        engineered["edad_decada"] = engineered["edad"] / 10.0
        engineered["edad_sq"] = engineered["edad"] ** 2
        for threshold in (50, 60, 65, 70):
            engineered[f"edad_ge_{threshold}"] = (engineered["edad"] >= threshold).astype(int)
        engineered["edad_gt_55"] = (engineered["edad"] > 55).astype(int)

    if _has_columns(engineered, ["trigliceridos", "glucosa"]):
        engineered["tyg_index"] = np.log(
            (engineered["trigliceridos"].clip(lower=1) * engineered["glucosa"].clip(lower=1)) / 2.0
        )
    if _has_columns(engineered, ["trigliceridos", "colesterol"]):
        engineered["trigliceridos_colesterol_ratio"] = engineered["trigliceridos"] / (engineered["colesterol"] + 1)
        engineered["colesterol_trigliceridos_ratio"] = engineered["colesterol"] / (engineered["trigliceridos"] + 1)
    if _has_columns(engineered, ["hemoglobina", "leucocitos"]):
        engineered["hemoglobina_leucocitos_ratio"] = engineered["hemoglobina"] / (engineered["leucocitos"] + 0.1)

    _add_threshold(engineered, "glucosa", "glucosa_ge_100", lower=100)
    _add_threshold(engineered, "glucosa", "glucosa_ge_126", lower=126)
    _add_threshold(engineered, "glucosa", "glucosa_gt_130", lower=130, inclusive=False)
    _add_threshold(engineered, "colesterol", "colesterol_ge_240", lower=240)
    _add_threshold(engineered, "trigliceridos", "trigliceridos_ge_150", lower=150)
    _add_threshold(engineered, "trigliceridos", "trigliceridos_ge_200", lower=200)
    _add_threshold(engineered, "hemoglobina", "hemoglobina_lt_12", upper=12)
    _add_threshold(engineered, "hemoglobina", "hemoglobina_lt_11", upper=11)
    _add_threshold(engineered, "hemoglobina", "hemoglobina_gt_16_5", lower=16.5)
    _add_threshold(engineered, "leucocitos", "leucocitos_lt_4", upper=4)
    _add_threshold(engineered, "leucocitos", "leucocitos_gt_10", lower=10, inclusive=False)
    _add_threshold(engineered, "leucocitos", "leucocitos_gt_11", lower=11)
    _add_threshold(engineered, "plaquetas", "plaquetas_lt_150", upper=150)
    _add_threshold(engineered, "plaquetas", "plaquetas_gt_400", lower=400)
    _add_threshold(engineered, "creatinina", "creatinina_gt_1_3", lower=1.3)

    engineered["metabolic_lab_count"] = _sum_existing(
        engineered,
        ["glucosa_gt_130", "colesterol_ge_240", "trigliceridos_ge_200"],
    )
    engineered["inflammation_count"] = _sum_existing(
        engineered,
        ["hemoglobina_lt_11", "leucocitos_gt_10", "plaquetas_gt_400"],
    )
    engineered["comorbidity_count"] = _sum_existing(engineered, COMORBIDITY_COLUMNS)
    engineered["core_comorbidity_count"] = _sum_existing(engineered, CORE_COMORBIDITY_COLUMNS)
    engineered["metabolic_syndrome_count"] = _sum_existing(engineered, METABOLIC_COMORBIDITY_COLUMNS)
    engineered["mutation_count"] = _sum_existing(engineered, GENETIC_COLUMNS)
    engineered["high_risk_mutation_count"] = _sum_existing(engineered, HIGH_RISK_GENETIC_COLUMNS)
    engineered["any_mutation"] = (engineered["mutation_count"] > 0).astype(int)
    engineered["multi_mutation"] = (engineered["mutation_count"] >= 2).astype(int)

    if _has_columns(engineered, ["mut_TP53", "mut_KRAS"]):
        engineered["tp53_or_kras"] = ((engineered["mut_TP53"] == 1) | (engineered["mut_KRAS"] == 1)).astype(int)
    if _has_columns(engineered, ["mut_BRCA1", "mut_TP53"]):
        engineered["brca1_or_tp53"] = ((engineered["mut_BRCA1"] == 1) | (engineered["mut_TP53"] == 1)).astype(int)

    _add_interaction(engineered, "fumador", "obesidad", "fumador_x_obesidad")
    _add_interaction(engineered, "fumador", "epoc", "fumador_x_epoc")
    _add_interaction(engineered, "fumador", "mutation_count", "fumador_x_mutation_count")
    _add_interaction(engineered, "fumador", "high_risk_mutation_count", "fumador_x_highrisk_mutation_count")
    _add_interaction(engineered, "fumador", "mut_TP53", "fumador_x_tp53")
    _add_interaction(engineered, "fumador", "mut_KRAS", "fumador_x_kras")
    _add_interaction(engineered, "obesidad", "diabetes", "obesidad_x_diabetes")
    _add_interaction(engineered, "obesidad", "hipertension", "obesidad_x_hipertension")
    _add_interaction(engineered, "obesidad", "trigliceridos_ge_200", "obesidad_x_trigliceridos_altos")
    _add_interaction(engineered, "actividad_baja", "obesidad", "actividad_baja_x_obesidad")
    _add_interaction(engineered, "actividad_baja", "fumador", "actividad_baja_x_fumador")
    _add_interaction(engineered, "edad_decada", "mutation_count", "edad_x_mutation_count")
    _add_interaction(engineered, "edad_decada", "high_risk_mutation_count", "edad_x_high_risk_mutation_count")
    _add_interaction(engineered, "edad_decada", "comorbidity_count", "edad_x_comorbidity_count")

    if "nivel_educativo" in engineered.columns:
        engineered["educacion_score"] = (
            engineered["nivel_educativo"]
            .map({"Sin estudios": 0, "Primaria": 1, "Secundaria": 2, "Universitario": 3})
            .astype(float)
        )
    if "nivel_ingresos" in engineered.columns:
        engineered["ingresos_score"] = (
            engineered["nivel_ingresos"].map({"Muy bajo": 0, "Bajo": 1, "Medio": 2, "Alto": 3}).astype(float)
        )
    if _has_columns(engineered, ["educacion_score", "ingresos_score"]):
        engineered["capital_social_score"] = engineered["educacion_score"] + engineered["ingresos_score"]
    if "distancia_hospital_km" in engineered.columns:
        engineered["log1p_distancia_hospital_km"] = np.log1p(engineered["distancia_hospital_km"].clip(lower=0))
        engineered["hospital_lejano_50km"] = (engineered["distancia_hospital_km"] > 50).astype(int)
        engineered["hospital_lejano_100km"] = (engineered["distancia_hospital_km"] > 100).astype(int)

    return engineered


def create_feature_signal_report(
    df: pd.DataFrame, output_path: Path = METRICS_DIR / "feature_signal_report.csv"
) -> pd.DataFrame:
    """Resume senal univariante y posibles fugas; no se usa para entrenar."""

    y = df[TARGET_COLUMN].astype(int)
    rows = []
    for column in df.columns:
        if column == TARGET_COLUMN:
            continue
        series = df[column]
        row = {
            "column": column,
            "dtype": str(series.dtype),
            "missing_pct": float(series.isna().mean()),
            "unique": int(series.nunique(dropna=True)),
            "known_leakage": bool(column in KNOWN_SUSPECT_LEAKAGE_COLUMNS),
            "policy_post_diagnosis_risk": bool(column in KNOWN_HIGH_RISK_POST_DIAGNOSIS_COLUMNS),
            "policy_low_signal_or_proxy": bool(column in LOW_SIGNAL_OR_PROXY_COLUMNS),
            "metadata_caution_comorbidity": bool(column in METADATA_CAUTION_COMORBIDITY_COLUMNS),
            "metadata_optional_or_proxy": bool(column in METADATA_OPTIONAL_OR_PROXY_COLUMNS),
            "numeric_corr_with_target": None,
            "numeric_abs_auc": None,
            "categorical_rate_spread": None,
            "level_target_rates": "",
        }
        if column == ID_COLUMN:
            rows.append(row)
            continue
        if pd.api.types.is_numeric_dtype(series) and row["unique"] > 1:
            values = series.astype(float)
            corr = np.corrcoef(values, y)[0, 1]
            row["numeric_corr_with_target"] = float(corr)
            try:
                auc = float(roc_auc_score(y, values))
                row["numeric_abs_auc"] = max(auc, 1.0 - auc)
            except ValueError:
                row["numeric_abs_auc"] = None
        elif row["unique"] > 1:
            rates = df.groupby(column, dropna=False)[TARGET_COLUMN].agg(["mean", "count"])
            row["categorical_rate_spread"] = float(rates["mean"].max() - rates["mean"].min())
            sorted_rates = rates.sort_values("mean", ascending=False).head(12)
            row["level_target_rates"] = "; ".join(
                f"{level}={record['mean']:.4f} n={int(record['count'])}"
                for level, record in sorted_rates.iterrows()
            )
        rows.append(row)

    report = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    return report


def build_feature_policy(df: pd.DataFrame, feature_view: str = RECOMMENDED_FEATURE_VIEW) -> FeaturePolicy:
    """Define columnas incluidas y excluidas del modelo principal."""

    if feature_view not in FEATURE_VIEWS:
        raise ValueError(f"feature_view debe ser uno de {FEATURE_VIEWS}; recibido={feature_view!r}.")

    excluded = [ID_COLUMN, TARGET_COLUMN]
    notes = [
        f"{ID_COLUMN} se excluye por ser identificador.",
        f"{TARGET_COLUMN} se usa exclusivamente como target.",
        f"Vista de features: {feature_view}.",
    ]

    for column in KNOWN_SUSPECT_LEAKAGE_COLUMNS:
        if column in df.columns:
            excluded.append(column)
            notes.append(f"{column} se excluye por fuga temporal/post-diagnostico confirmada en auditoria.")

    if feature_view != "economic_sensitivity":
        for column in KNOWN_HIGH_RISK_POST_DIAGNOSIS_COLUMNS:
            if column in df.columns:
                excluded.append(column)
                notes.append(
                    f"{column} se excluye por alto riesgo de fuga temporal: puede reflejar coste, "
                    "tratamiento o uso hospitalario posterior al diagnostico."
                )
    else:
        notes.append(
            "economic_sensitivity incluye costes/uso hospitalario solo como sensibilidad de fuga; "
            "no debe presentarse como modelo operativo de cribado."
        )

    for column in KNOWN_CONSTANT_CANDIDATES:
        if column in df.columns and df[column].nunique(dropna=False) <= 1:
            excluded.append(column)
            notes.append(f"{column} se excluye por ser constante en los CSV disponibles.")

    generated_engineered = [column for column in ALL_ENGINEERED_FEATURE_COLUMNS if column in df.columns]
    selected_engineered = [column for column in SELECTED_ENGINEERED_FEATURE_COLUMNS if column in df.columns]
    if feature_view == "metadata_core":
        excluded.extend([column for column in METADATA_OPTIONAL_OR_PROXY_COLUMNS if column in df.columns])
        excluded.extend(generated_engineered)
        notes.append(
            "La vista metadata_core aplica la guia oficial: bioquimica, genetica, fumador, "
            "actividad_fisica y edad; deja fuera comorbilidades de cautela, proxies "
            "sociodemograficos opcionales, tipo_seguro, constantes y fugas."
        )
    elif feature_view == "base":
        excluded.extend([column for column in LOW_SIGNAL_OR_PROXY_COLUMNS if column in df.columns])
        excluded.extend(generated_engineered)
        notes.append(
            "La vista base conserva el conjunto compacto historico usado antes de disponer "
            "del metadato oficial; se mantiene para reproducibilidad."
        )
    elif feature_view == "safe_all":
        excluded.extend(generated_engineered)
        notes.append(
            "La vista safe_all incluye toda variable prediagnostico disponible salvo fugas, "
            "constantes e identificadores; es util para auditoria, no es el baseline."
        )
    elif feature_view == "engineered_selected":
        excluded.extend([column for column in METADATA_OPTIONAL_OR_PROXY_COLUMNS if column in df.columns])
        excluded.extend([column for column in generated_engineered if column not in selected_engineered])
        notes.append(
            "La vista engineered_selected anade derivadas deterministas alineadas con el "
            "modelo generativo oficial, sin target, vive, costes, comorbilidades de cautela "
            "ni proxies sociodemograficos opcionales."
        )
    elif feature_view == "economic_sensitivity":
        excluded.extend(generated_engineered)
        notes.append(
            "La vista economic_sensitivity conserva variables socioeconomicas y economicas de alto riesgo "
            "para medir sensibilidad a posibles fugas, manteniendo fuera identificadores, target, "
            "constantes y fugas confirmadas."
        )

    excluded = _deduplicate_preserve_order(excluded)
    for column in excluded:
        if column in METADATA_CAUTION_COMORBIDITY_COLUMNS:
            notes.append(
                f"{column} se excluye del modelo principal porque el metadato la marca como "
                "comorbilidad a valorar: correlaciona con cancer por diseno y puede introducir "
                "leakage indirecto si no se justifica temporalmente."
            )
        elif column in METADATA_OPTIONAL_SOCIODEMOGRAPHIC_COLUMNS:
            notes.append(
                f"{column} se excluye del modelo principal porque el metadato la marca como "
                "sociodemografica opcional y de bajo peso predictivo."
            )
        elif column == "tipo_seguro":
            notes.append(
                "tipo_seguro se excluye del modelo principal porque pertenece al bloque economico "
                "y puede actuar como proxy socioeconomico; queda solo para vistas ampliadas."
            )
        elif column in LOW_SIGNAL_OR_PROXY_COLUMNS:
            notes.append(
                f"{column} se excluye del modelo principal por baja senal local, "
                "riesgo de ruido o papel de proxy no clinico."
            )

    categorical = [col for col in KNOWN_CATEGORICAL_COLUMNS if col in df.columns and col not in excluded]
    numeric: list[str] = []
    binary: list[str] = []
    for column in df.columns:
        if column in excluded or column in categorical:
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
    engineered = [column for column in included if column in ALL_ENGINEERED_FEATURE_COLUMNS]
    LOGGER.info(
        "Politica de features (%s): %d numericas, %d binarias, %d categoricas, %d engineered, %d excluidas.",
        feature_view,
        len(numeric),
        len(binary),
        len(categorical),
        len(engineered),
        len(excluded),
    )
    return FeaturePolicy(
        target=TARGET_COLUMN,
        feature_view=feature_view,
        excluded=excluded,
        numeric=numeric,
        binary=binary,
        categorical=categorical,
        engineered=engineered,
        included=included,
        notes=notes,
    )


def save_feature_policy(policy: FeaturePolicy, output_path: Path = METRICS_DIR / "feature_policy.json") -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(policy), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_preprocessor(policy: FeaturePolicy) -> ColumnTransformer:
    """Construye ColumnTransformer ajustable solo con datos de entrenamiento."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    binary_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if policy.numeric:
        transformers.append(("numeric", numeric_pipeline, policy.numeric))
    if policy.binary:
        transformers.append(("binary", binary_pipeline, policy.binary))
    if policy.categorical:
        transformers.append(("categorical", categorical_pipeline, policy.categorical))

    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def split_data(
    df: pd.DataFrame,
    policy: FeaturePolicy,
    seed: int = DEFAULT_SEED,
    test_size: float = 0.20,
    validation_size: float = 0.20,
) -> dict[str, pd.Series | pd.DataFrame]:
    """Crea split test 80/20 estratificado y validacion interna dentro de train."""

    X = df[policy.included].copy()
    y = df[policy.target].astype(int).copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    X_train_inner, X_valid, y_train_inner, y_valid = train_test_split(
        X_train,
        y_train,
        test_size=validation_size,
        random_state=seed,
        stratify=y_train,
    )
    return {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_inner": X_train_inner,
        "X_valid": X_valid,
        "y_train_inner": y_train_inner,
        "y_valid": y_valid,
    }


def save_split_summary(splits: dict[str, pd.Series | pd.DataFrame], output_path: Path) -> None:
    rows = []
    for name in ["y_train", "y_test", "y_train_inner", "y_valid"]:
        y = splits[name]
        counts = y.value_counts().sort_index()
        negatives = int(counts.get(0, 0))
        positives = int(counts.get(1, 0))
        rows.append(
            {
                "split": name.replace("y_", ""),
                "rows": int(len(y)),
                "negatives": negatives,
                "positives": positives,
                "positive_prevalence": positives / len(y),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def _has_columns(df: pd.DataFrame, columns: list[str]) -> bool:
    return all(column in df.columns for column in columns)


def _sum_existing(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    present = [column for column in columns if column in df.columns]
    if not present:
        return pd.Series(0, index=df.index)
    return df[present].sum(axis=1)


def _add_threshold(
    df: pd.DataFrame,
    source: str,
    output: str,
    lower: float | None = None,
    upper: float | None = None,
    inclusive: bool = True,
) -> None:
    if source not in df.columns:
        return
    if lower is not None:
        if inclusive:
            df[output] = (df[source] >= lower).astype(int)
        else:
            df[output] = (df[source] > lower).astype(int)
    elif upper is not None:
        df[output] = (df[source] < upper).astype(int)


def _add_interaction(df: pd.DataFrame, left: str, right: str, output: str) -> None:
    if left in df.columns and right in df.columns:
        df[output] = df[left] * df[right]


def _deduplicate_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    deduplicated = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduplicated.append(value)
    return deduplicated
