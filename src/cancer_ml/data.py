"""Carga, union y auditoria de los CSV locales."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

import pandas as pd

from cancer_ml.config import (
    ECONOMIC_CSV,
    EXPECTED_CSVS,
    ID_COLUMN,
    METRICS_DIR,
    RAW_DATA_DIR,
    TARGET_COLUMN,
)

LOGGER = logging.getLogger(__name__)


def read_csv_tolerant(path: Path) -> pd.DataFrame:
    """Lee CSV con tolerancia a BOM y encodings comunes."""

    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise UnicodeDecodeError("csv", b"", 0, 1, "; ".join(errors))


def load_available_csvs(raw_dir: Path = RAW_DATA_DIR) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Carga los CSV esperados que existen localmente y lista los faltantes."""

    frames: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for filename in EXPECTED_CSVS:
        path = raw_dir / filename
        if not path.exists():
            missing.append(filename)
            if filename == ECONOMIC_CSV:
                LOGGER.warning(
                    "%s no esta localmente; se ignora sin bloquear el pipeline.",
                    filename,
                )
            else:
                LOGGER.warning("%s no esta localmente.", filename)
            continue
        df = read_csv_tolerant(path)
        if ID_COLUMN not in df.columns:
            raise ValueError(f"{filename} no contiene la clave {ID_COLUMN}.")
        duplicate_ids = int(df[ID_COLUMN].duplicated().sum())
        if duplicate_ids:
            raise ValueError(f"{filename} contiene {duplicate_ids} IDs duplicados.")
        frames[filename] = df
        LOGGER.info("Cargado %s con forma %s.", filename, df.shape)

    if not frames:
        raise FileNotFoundError(f"No hay CSV disponibles en {raw_dir}.")
    return frames, missing


def audit_collections(
    frames: Mapping[str, pd.DataFrame], missing: list[str], output_path: Path | None = None
) -> pd.DataFrame:
    """Audita filas, columnas, IDs, duplicados y faltantes por coleccion."""

    first_ids = next(iter(frames.values()))[ID_COLUMN]
    first_id_set = set(first_ids)
    rows = []
    for filename, df in frames.items():
        id_set = set(df[ID_COLUMN])
        rows.append(
            {
                "collection": filename,
                "present": True,
                "rows": int(len(df)),
                "columns": int(df.shape[1]),
                "duplicate_paciente_id": int(df[ID_COLUMN].duplicated().sum()),
                "missing_values_total": int(df.isna().sum().sum()),
                "same_ids_as_first_csv": id_set == first_id_set,
                "ids_only_in_first": int(len(first_id_set - id_set)),
                "ids_only_in_collection": int(len(id_set - first_id_set)),
            }
        )
    for filename in missing:
        rows.append(
            {
                "collection": filename,
                "present": False,
                "rows": 0,
                "columns": 0,
                "duplicate_paciente_id": 0,
                "missing_values_total": 0,
                "same_ids_as_first_csv": False,
                "ids_only_in_first": None,
                "ids_only_in_collection": None,
            }
        )
    audit = pd.DataFrame(rows)
    if output_path is None:
        output_path = METRICS_DIR / "data_audit.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_path, index=False)
    return audit


def join_collections(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Une todas las colecciones presentes por paciente_id sin perder pacientes."""

    ordered_names = [name for name in EXPECTED_CSVS if name in frames]
    if not ordered_names:
        raise ValueError("No hay colecciones disponibles para unir.")

    joined = frames[ordered_names[0]].copy()
    expected_rows = len(joined)
    expected_ids = set(joined[ID_COLUMN])
    for name in ordered_names[1:]:
        df = frames[name]
        current_ids = set(df[ID_COLUMN])
        if current_ids != expected_ids:
            missing_in_current = len(expected_ids - current_ids)
            extra_in_current = len(current_ids - expected_ids)
            raise ValueError(
                f"Los IDs de {name} no coinciden con la primera coleccion "
                f"(faltan={missing_in_current}, sobran={extra_in_current})."
            )
        joined = joined.merge(df, on=ID_COLUMN, how="inner", validate="one_to_one")
        if len(joined) != expected_rows:
            raise ValueError(
                f"La union con {name} cambio el numero de filas "
                f"({len(joined)} vs {expected_rows})."
            )

    if TARGET_COLUMN not in joined.columns:
        raise ValueError(f"No se encontro la variable objetivo {TARGET_COLUMN}.")
    LOGGER.info("Dataset unido con forma %s.", joined.shape)
    return joined


def cancer_balance(df: pd.DataFrame) -> dict[str, float]:
    """Resume prevalencia y razon de desbalance del target."""

    counts = df[TARGET_COLUMN].value_counts().sort_index()
    negatives = int(counts.get(0, 0))
    positives = int(counts.get(1, 0))
    prevalence = positives / (positives + negatives)
    ratio = negatives / positives if positives else float("inf")
    return {
        "negatives": negatives,
        "positives": positives,
        "prevalence": prevalence,
        "negative_positive_ratio": ratio,
    }


def create_eda_summary(df: pd.DataFrame, output_dir: Path = METRICS_DIR) -> pd.DataFrame:
    """Genera resumen reproducible de tipos, rangos, cardinalidades y prevalencias."""

    rows = []
    for column in df.columns:
        series = df[column]
        non_null = series.dropna()
        row = {
            "column": column,
            "dtype": str(series.dtype),
            "missing": int(series.isna().sum()),
            "missing_pct": float(series.isna().mean()),
            "unique": int(series.nunique(dropna=True)),
            "min": None,
            "max": None,
            "mean_or_prevalence": None,
            "top_values": "",
        }
        if pd.api.types.is_numeric_dtype(series):
            row["min"] = float(non_null.min()) if len(non_null) else None
            row["max"] = float(non_null.max()) if len(non_null) else None
            row["mean_or_prevalence"] = float(non_null.mean()) if len(non_null) else None
            values = set(non_null.unique().tolist())
            if values.issubset({0, 1}):
                row["positive_count"] = int((series == 1).sum())
                row["positive_pct"] = float((series == 1).mean())
        else:
            value_counts = series.value_counts(dropna=False).head(8)
            row["top_values"] = "; ".join(f"{idx}={int(value)}" for idx, value in value_counts.items())
        rows.append(row)

    summary = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "eda_summary.csv", index=False)
    try:
        markdown = summary.fillna("").to_markdown(index=False)
    except Exception:
        markdown = _dataframe_to_markdown(summary.fillna(""))
    (output_dir / "eda_summary.md").write_text(markdown + "\n", encoding="utf-8")
    return summary


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)

