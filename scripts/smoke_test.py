#!/usr/bin/env python
"""Comprobacion rapida de imports, carga, union, features y preprocesamiento."""

from __future__ import annotations

import sys
from pathlib import Path
from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cancer_ml.config import FINAL_PRESENTATION_PATH, FIGURES_DIR, ensure_directories, set_global_seed  # noqa: E402
from cancer_ml.data import audit_collections, cancer_balance, join_collections, load_available_csvs  # noqa: E402
from cancer_ml.features import add_engineered_features, build_feature_policy, build_preprocessor, split_data  # noqa: E402


def main() -> int:
    ensure_directories()
    set_global_seed(42)
    frames, missing = load_available_csvs()
    audit = audit_collections(frames, missing)
    dataset = join_collections(frames)
    policy = build_feature_policy(dataset)
    splits = split_data(dataset, policy, seed=42)
    preprocessor = build_preprocessor(policy)
    transformed = preprocessor.fit_transform(splits["X_train"].head(512), splits["y_train"].head(512))
    engineered_dataset = add_engineered_features(dataset)
    engineered_policy = build_feature_policy(engineered_dataset, feature_view="engineered_selected")
    engineered_splits = split_data(engineered_dataset, engineered_policy, seed=42)
    engineered_preprocessor = build_preprocessor(engineered_policy)
    engineered_transformed = engineered_preprocessor.fit_transform(
        engineered_splits["X_train"].head(512),
        engineered_splits["y_train"].head(512),
    )
    economic_policy = build_feature_policy(dataset, feature_view="economic_sensitivity")
    economic_splits = split_data(dataset, economic_policy, seed=42)
    economic_preprocessor = build_preprocessor(economic_policy)
    economic_transformed = economic_preprocessor.fit_transform(
        economic_splits["X_train"].head(512),
        economic_splits["y_train"].head(512),
    )
    print("smoke: csv_cargados", len(frames))
    print("smoke: csv_faltantes", missing)
    print("smoke: audit_ok", audit["present"].sum(), "presentes")
    print("smoke: dataset", dataset.shape)
    print("smoke: balance", cancer_balance(dataset))
    print("smoke: features", len(policy.included), "incluidas", policy.excluded, "excluidas")
    print("smoke: preprocessor_shape", transformed.shape)
    print(
        "smoke: engineered_features",
        len(engineered_policy.included),
        "incluidas",
        len(engineered_policy.engineered),
        "derivadas",
    )
    print("smoke: engineered_preprocessor_shape", engineered_transformed.shape)
    print(
        "smoke: economic_sensitivity_features",
        len(economic_policy.included),
        "incluidas",
        economic_transformed.shape,
    )
    pdf_path = FINAL_PRESENTATION_PATH
    if pdf_path.exists():
        print("smoke: pdf_pages", len(PdfReader(str(pdf_path)).pages))
    else:
        print("smoke: pdf_pendiente", pdf_path)
    for figure in [
        FIGURES_DIR / "confusion_matrix_best_ml.png",
        FIGURES_DIR / "mlp_learning_curves.png",
        FIGURES_DIR / "roc_curves.png",
        FIGURES_DIR / "precision_recall_space.png",
    ]:
        print("smoke: figure", figure.name, figure.exists())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
