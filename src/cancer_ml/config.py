"""Configuracion central del proyecto."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"
FINAL_PRESENTATION_PATH = REPORTS_DIR / "presentacion_final_cancer_5_diapositivas.pdf"
RECOMMENDED_MODEL_ARTIFACT = MODELS_DIR / "modelo_recomendado_xgboost_auc.joblib"

TARGET_COLUMN = "cancer"
ID_COLUMN = "paciente_id"
DEFAULT_SEED = 42
PRIMARY_SELECTION_METRIC = "f1_positive"
PRIMARY_SELECTION_LABEL = "F1 cancer=1"

RECOMMENDED_FEATURE_VIEW = "safe_all"
RECOMMENDED_THRESHOLD = 0.66
RECOMMENDED_MODEL_NAME = "XGBoostAUC_cuda"

EXPECTED_CSVS = [
    "CASOCANCER_01_BIOQUIMICOS.csv",
    "CASOCANCER_02_CLINICOS.csv",
    "CASOCANCER_03_GENETICOS.csv",
    "CASOCANCER_04_ECONOMICOS.csv",
    "CASOCANCER_05_GENERALES.csv",
    "CASOCANCER_06_SOCIODEMOGRAFICOS.csv",
]

ECONOMIC_CSV = "CASOCANCER_04_ECONOMICOS.csv"

KNOWN_CATEGORICAL_COLUMNS = [
    "tipo_seguro",
    "actividad_fisica",
    "nivel_educativo",
    "nivel_ingresos",
    "zona",
    "estado_civil",
]

KNOWN_SUSPECT_LEAKAGE_COLUMNS = ["vive"]
KNOWN_HIGH_RISK_POST_DIAGNOSIS_COLUMNS = [
    "coste_total",
    "coste_farmaco",
    "num_ingresos",
    "dias_hospital",
]
KNOWN_CONSTANT_CANDIDATES = ["alcohol"]
LOW_SIGNAL_OR_PROXY_COLUMNS = [
    "plaquetas",
    "creatinina",
    "asma",
    "enfermedad_cardiaca",
    "mut_ALK",
    "nivel_educativo",
    "nivel_ingresos",
    "zona",
    "estado_civil",
    "num_hijos",
    "distancia_hospital_km",
    "tipo_seguro",
]

METADATA_CAUTION_COMORBIDITY_COLUMNS = [
    "diabetes",
    "hipertension",
    "obesidad",
    "enfermedad_cardiaca",
    "asma",
    "epoc",
]

METADATA_OPTIONAL_SOCIODEMOGRAPHIC_COLUMNS = [
    "nivel_educativo",
    "nivel_ingresos",
    "zona",
    "estado_civil",
    "num_hijos",
    "distancia_hospital_km",
]

METADATA_OPTIONAL_OR_PROXY_COLUMNS = [
    *METADATA_CAUTION_COMORBIDITY_COLUMNS,
    *METADATA_OPTIONAL_SOCIODEMOGRAPHIC_COLUMNS,
    "tipo_seguro",
]


@dataclass(frozen=True)
class ModeConfig:
    """Hiperparametros principales por modo de ejecucion."""

    mode: str
    rf_estimators: int
    et_estimators: int
    hgb_iter: int
    xgb_estimators: int
    mlp_epochs: int
    mlp_patience: int
    mlp_reduce_patience: int
    batch_size: int
    train_sample_size: int | None = None


MODE_CONFIGS = {
    "quick": ModeConfig(
        mode="quick",
        rf_estimators=80,
        et_estimators=100,
        hgb_iter=120,
        xgb_estimators=140,
        mlp_epochs=10,
        mlp_patience=4,
        mlp_reduce_patience=2,
        batch_size=1024,
        train_sample_size=14000,
    ),
    "full": ModeConfig(
        mode="full",
        rf_estimators=600,
        et_estimators=600,
        hgb_iter=450,
        xgb_estimators=600,
        mlp_epochs=90,
        mlp_patience=12,
        mlp_reduce_patience=6,
        batch_size=512,
        train_sample_size=None,
    ),
}


def ensure_directories() -> None:
    """Crea directorios locales de salida, aunque outputs/ y models/ esten ignorados."""

    for path in [OUTPUT_DIR, METRICS_DIR, FIGURES_DIR, REPORTS_DIR, MODELS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    """Fija semilla en Python, NumPy y, si esta disponible, TensorFlow."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
    except Exception:
        pass
