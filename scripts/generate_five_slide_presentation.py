from __future__ import annotations

import base64
import html
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"

PDF_OUTPUT = REPORTS_DIR / "presentacion_final_cancer_5_diapositivas.pdf"
HTML_OUTPUT = REPORTS_DIR / "presentacion_final_cancer_5_diapositivas.html"
PREVIEW_DIR = OUTPUTS_DIR / "reports" / "presentation_html_preview"
SLIDE_COUNT = 5

PALETTE = {
    "ink": "#211d1a",
    "muted": "#756e66",
    "line": "#d4c9b8",
    "paper": "#f2ecdf",
    "panel": "#fbf8f0",
    "teal": "#7a1f2b",
    "teal2": "#b08a5a",
    "berry": "#4c5d48",
    "gold": "#a66b21",
    "blue": "#2d4658",
    "red": "#8d2d1f",
    "green": "#596f4e",
}

MODEL_SHORT = {
    "XGBoostAUC_cuda": "XGB AUC",
    "ValidationSoftVoting": "SoftVoting",
    "XGBoostF1Balanced_cuda": "XGB F1Bal",
    "GradientBoosting": "GradBoost",
    "HistGradientBoostingRegularized": "HGB Reg",
    "XGBoost_cuda": "XGB",
    "HistGradientBoosting": "HGB",
    "LogisticRegression_baseline": "LogReg",
    "RandomForest": "RF",
    "ExtraTrees": "ExtraTrees",
    "MLP": "MLP",
}

MODEL_COLORS = {
    "XGBoostAUC_cuda": PALETTE["teal"],
    "ValidationSoftVoting": PALETTE["berry"],
    "XGBoostF1Balanced_cuda": PALETTE["blue"],
    "GradientBoosting": PALETTE["gold"],
    "HistGradientBoostingRegularized": PALETTE["green"],
    "XGBoost_cuda": "#635a50",
    "MLP": "#211d1a",
}


@dataclass(frozen=True)
class Context:
    metrics: pd.DataFrame
    intervals: pd.DataFrame
    threshold_search: pd.DataFrame
    mlp_threshold_search: pd.DataFrame
    split: pd.DataFrame
    data_audit: pd.DataFrame
    feature_signal: pd.DataFrame
    mlp_history: pd.DataFrame
    balance: dict
    policy: dict
    model_card: dict
    run_summary: dict


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    context = load_context()
    html_text = build_html(context)
    HTML_OUTPUT.write_text(html_text, encoding="utf-8")
    export_pdf(HTML_OUTPUT, PDF_OUTPUT)
    validate_pdf(PDF_OUTPUT)
    render_previews(PDF_OUTPUT)
    print(f"HTML generado: {HTML_OUTPUT}")
    print(f"PDF generado: {PDF_OUTPUT}")
    return 0


def load_context() -> Context:
    metrics = pd.read_csv(METRICS_DIR / "model_metrics.csv").sort_values(
        ["f1_positive", "auc_roc"], ascending=False
    ).reset_index(drop=True)
    return Context(
        metrics=metrics,
        intervals=pd.read_csv(METRICS_DIR / "model_metric_intervals.csv"),
        threshold_search=pd.read_csv(METRICS_DIR / "model_threshold_search.csv"),
        mlp_threshold_search=pd.read_csv(METRICS_DIR / "mlp_threshold_search.csv"),
        split=pd.read_csv(METRICS_DIR / "split_summary.csv"),
        data_audit=pd.read_csv(METRICS_DIR / "data_audit.csv"),
        feature_signal=pd.read_csv(METRICS_DIR / "feature_signal_report.csv"),
        mlp_history=pd.read_csv(METRICS_DIR / "mlp_history.csv"),
        balance=read_json(METRICS_DIR / "target_balance.json"),
        policy=read_json(METRICS_DIR / "feature_policy.json"),
        model_card=read_json(METRICS_DIR / "model_card.json"),
        run_summary=read_json(METRICS_DIR / "run_summary.json"),
    )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_html(ctx: Context) -> str:
    slides = [slide_1(ctx), slide_2(ctx), slide_3(ctx), slide_4(ctx), slide_5(ctx)]
    return f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8" />
<title>Estudio de viabilidad - Cancer ML</title>
<style>{css()}</style>
</head>
<body>
{''.join(slides)}
</body>
</html>
"""


def css() -> str:
    return """
@page { size: 13.333in 7.5in; margin: 0; }
* { box-sizing: border-box; }
html, body { margin: 0; background: #d8cfbf; color: #211d1a; }
body { font-family: "Avenir Next", "Helvetica Neue", sans-serif; }
.slide {
  width: 13.333in;
  height: 7.5in;
  page-break-after: always;
  position: relative;
  overflow: hidden;
  padding: 0.46in 0.58in 0.46in 0.62in;
  background:
    linear-gradient(90deg, rgba(122,31,43,.10) 0 0.08in, transparent 0.08in),
    linear-gradient(180deg, rgba(255,255,255,.52), rgba(242,236,223,.92)),
    #f2ecdf;
}
.slide::before {
  content: "";
  position: absolute;
  inset: 0;
  pointer-events: none;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='180' height='180' viewBox='0 0 180 180'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='180' height='180' filter='url(%23n)' opacity='.16'/%3E%3C/svg%3E");
  mix-blend-mode: multiply;
  opacity: .10;
}
.slide::after {
  content: "";
  position: absolute;
  left: 0.34in;
  top: 0.45in;
  bottom: 0.45in;
  width: 1px;
  background: rgba(122,31,43,.34);
}
.slide > * { position: relative; z-index: 1; }
.kicker { color: #7a1f2b; font-size: 0.105in; font-weight: 850; text-transform: uppercase; letter-spacing: .11em; }
.title-row { display: grid; grid-template-columns: 1fr auto; gap: 0.24in; align-items: end; margin-bottom: 0.18in; }
h1 { font-family: "Iowan Old Style", Georgia, serif; font-size: 0.43in; line-height: 1.02; margin: 0.03in 0 0; font-weight: 700; letter-spacing: 0; max-width: 10.3in; }
.subtitle { color: #756e66; font-size: 0.145in; line-height: 1.34; max-width: 8.45in; margin-top: 0.08in; }
.slide-no { min-width: 0.62in; border-top: 2px solid #7a1f2b; border-bottom: 1px solid rgba(33,29,26,.24); color: #7a1f2b; text-align: right; padding: 0.07in 0 0.06in; font-family: "Iowan Old Style", Georgia, serif; font-weight: 700; font-size: 0.15in; }
.grid { display: grid; gap: 0.18in; }
.cols-2 { grid-template-columns: 1fr 1fr; }
.cols-3 { grid-template-columns: repeat(3, 1fr); }
.cols-4 { grid-template-columns: repeat(4, 1fr); }
.cols-5 { grid-template-columns: repeat(5, 1fr); }
.stat-row { display: grid; grid-template-columns: repeat(6, 1fr); gap: 0.13in; margin: 0.13in 0 0.2in; }
.stat {
  background: rgba(251,248,240,.72);
  border-top: 2px solid var(--accent, #7a1f2b);
  border-bottom: 1px solid rgba(33,29,26,.16);
  padding: 0.12in 0.08in 0.10in;
  min-height: 0.77in;
}
.stat .value { font-size: 0.27in; font-weight: 900; line-height: 1; white-space: nowrap; }
.stat .label { margin-top: 0.08in; color: #756e66; font-size: 0.085in; font-weight: 760; line-height: 1.24; text-transform: uppercase; letter-spacing: .04em; }
.panel {
  background: rgba(251,248,240,.84);
  border: 1px solid rgba(33,29,26,.16);
  border-top: 3px solid rgba(33,29,26,.82);
  border-radius: 0.035in;
  padding: 0.18in;
}
.panel.tight { padding: 0.13in; }
.panel h2 { margin: 0 0 0.12in; font-size: 0.19in; line-height: 1.1; font-weight: 850; }
.panel h3 { margin: 0 0 0.08in; font-size: 0.125in; line-height: 1.12; color: #7a1f2b; font-weight: 850; text-transform: uppercase; letter-spacing: .06em; }
.panel p, .panel li { color: #423c36; font-size: 0.125in; line-height: 1.34; margin: 0; }
ul { margin: 0; padding-left: 0.19in; }
li + li { margin-top: 0.055in; }
.callout { border-left: 0.04in solid #7a1f2b; background: rgba(251,248,240,.70); padding: 0.12in 0.15in; border-radius: 0.025in; color: #342722; font-size: 0.135in; line-height: 1.38; font-weight: 700; }
.callout.berry { border-left-color: #7a1f2b; background: rgba(122,31,43,.09); color: #4b1921; }
.callout.gold { border-left-color: #a66b21; background: rgba(166,107,33,.10); color: #4d3514; }
table { width: 100%; border-collapse: collapse; font-size: 0.103in; }
th { background: transparent; color: #211d1a; text-align: left; padding: 0.07in 0.08in; font-weight: 850; border-top: 2px solid #211d1a; border-bottom: 1px solid rgba(33,29,26,.55); }
td { background: transparent; border-bottom: 1px solid rgba(33,29,26,.12); padding: 0.064in 0.08in; color: #423c36; vertical-align: middle; }
tr:nth-child(even) td { background: rgba(255,255,255,.20); }
.badge { display: inline-block; padding: 0.015in 0.055in; border: 1px solid rgba(122,31,43,.32); border-radius: 0.02in; font-weight: 850; font-size: 0.075in; color: #7a1f2b; white-space: nowrap; text-transform: uppercase; letter-spacing: .05em; }
.badge.warn { border-color: rgba(166,107,33,.36); color: #83551e; }
.badge.risk { border-color: rgba(122,31,43,.36); color: #7a1f2b; }
.figure { background: rgba(251,248,240,.78); border: 1px solid rgba(33,29,26,.16); border-top: 3px solid #7a1f2b; border-radius: 0.035in; padding: 0.12in; }
.figure-title { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.08in; color: #211d1a; font-weight: 850; font-size: 0.14in; }
img.chart { display: block; width: 100%; height: 100%; object-fit: contain; }
.bars { display: grid; gap: 0.064in; }
.bar-line { display: grid; grid-template-columns: 1.28in 1fr 0.42in; gap: 0.08in; align-items: center; font-size: 0.102in; color: #31413c; }
.track { height: 0.08in; background: #ddd4c5; overflow: hidden; }
.fill { height: 100%; width: var(--w); background: var(--c, #7a1f2b); }
.metric-strip { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.1in; }
.mini { border-top: 2px solid rgba(33,29,26,.72); border-bottom: 1px solid rgba(33,29,26,.18); background: rgba(251,248,240,.62); padding: 0.1in 0.04in; }
.mini .n { font-size: 0.22in; font-weight: 900; }
.mini .t { color: #756e66; font-size: 0.08in; text-transform: uppercase; font-weight: 800; margin-top: 0.04in; letter-spacing: .04em; }
.arch { display: grid; grid-template-columns: repeat(5, 1fr); align-items: center; gap: 0.08in; margin-top: 0.08in; }
.layer { border: 1px solid #7a1f2b; border-top: 3px solid #7a1f2b; border-radius: 0.025in; background: rgba(251,248,240,.76); min-height: 0.72in; padding: 0.1in; text-align: center; }
.layer.dark { border-color: #211d1a; }
.layer .l1 { font-weight: 900; color: #211d1a; font-size: 0.13in; }
.layer .l2 { color: #756e66; margin-top: 0.045in; font-size: 0.095in; line-height: 1.18; }
.arrow { color: #756e66; font-weight: 900; text-align: center; margin-left: -0.04in; margin-right: -0.04in; }
.footer { position: absolute; left: 0.62in; right: 0.58in; bottom: 0.19in; display: flex; justify-content: space-between; color: #756e66; font-size: 0.072in; }
.watermark { display: none; }
@media print { .slide { break-after: page; } }
"""


def slide_shell(number: int, kicker: str, title: str, subtitle: str, content: str) -> str:
    return f"""
<section class="slide">
  <div class="title-row">
    <div>
      <div class="kicker">{esc(kicker)}</div>
      <h1>{esc(title)}</h1>
      <div class="subtitle">{esc(subtitle)}</div>
    </div>
    <div class="slide-no">{number}/5</div>
  </div>
  {content}
  <div class="watermark">0{number}</div>
  <div class="footer"><span>Prediccion de diagnostico de cancer · comparativa ML vs MLP</span><span>seed 42 · umbrales seleccionados en validacion</span></div>
</section>
"""


def slide_1(ctx: Context) -> str:
    balance = ctx.balance
    train = ctx.split[ctx.split["split"] == "train"].iloc[0]
    test = ctx.split[ctx.split["split"] == "test"].iloc[0]
    stats = stat_row([
        ("50.001", "Pacientes unidos", "teal"),
        (pct(balance["prevalence"], 1), "Prevalencia cancer=1", "berry"),
        (f"{balance['negative_positive_ratio']:.2f}:1", "Desbalance", "gold"),
        ("6/6", "Colecciones CSV", "blue"),
        (str(len(ctx.policy["included"])), "Features incluidas", "green"),
        (str(len(ctx.policy["excluded"])), "Excluidas", "red"),
    ])
    audit_rows = []
    for _, row in ctx.data_audit.iterrows():
        audit_rows.append([
            row["collection"].replace("CASOCANCER_", "").replace(".csv", ""),
            fmt_int(row["rows"]),
            fmt_int(row["columns"]),
            badge("OK" if row["present"] else "Falta"),
            "sin duplicados" if int(row["duplicate_paciente_id"]) == 0 else "revisar",
        ])
    table = html_table(["Coleccion", "Filas", "Cols", "Estado", "IDs"], audit_rows)
    content = f"""
{stats}
<div class="grid cols-2" style="grid-template-columns: 1.12fr .88fr;">
  <div class="panel">
    <h2>Datos disponibles y union</h2>
    {table}
  </div>
  <div class="grid" style="gap:.14in;">
    <div class="callout">Objetivo: evaluar si los datos multimodales permiten anticipar <b>cancer=1</b>, comparando modelos ML complejos con una MLP y recomendando una estrategia de cribado.</div>
    <div class="panel tight">
      <h3>Politica de features</h3>
      <p><b>Incluidas:</b> {esc(', '.join(ctx.policy['included'][:18]))}...</p>
      <p style="margin-top:.08in;"><b>Excluidas:</b> {esc(', '.join(ctx.policy['excluded']))}. Se retiran identificadores, target, constantes y variables con riesgo post-diagnostico.</p>
    </div>
    <div class="metric-strip">
      <div class="mini"><div class="n">{fmt_int(train['rows'])}</div><div class="t">train</div></div>
      <div class="mini"><div class="n">{fmt_int(test['rows'])}</div><div class="t">test</div></div>
      <div class="mini"><div class="n">{fmt_int(train['positives'])}</div><div class="t">positivos train</div></div>
      <div class="mini"><div class="n">{fmt_int(test['positives'])}</div><div class="t">positivos test</div></div>
    </div>
  </div>
</div>
"""
    return slide_shell(1, "Objetivo y datos", "Un dataset completo, una pregunta clinica exigente", "El proyecto se plantea como viabilidad de cribado: maximizar informacion util sin permitir fuga temporal.", content)


def slide_2(ctx: Context) -> str:
    best = ctx.metrics.iloc[0]
    ml = ctx.metrics[ctx.metrics["model_type"].isin(["ML", "Ensemble"])].head(7)
    rows = []
    for _, row in ml.iterrows():
        rows.append([
            esc(short(row["model"])),
            f"{row['threshold']:.2f}",
            f"{row['precision_positive']:.3f}",
            f"{row['recall_positive']:.3f}",
            f"{row['f1_positive']:.3f}",
            f"{row['auc_roc']:.3f}",
        ])
    cm = confusion_svg(best)
    stats = stat_row([
        (short(best["model"]), "Mejor ML por F1", "teal"),
        (f"{best['f1_positive']:.3f}", "F1 cancer=1", "green"),
        (f"{best['precision_positive']:.3f}", "Precision", "blue"),
        (f"{best['recall_positive']:.3f}", "Recall", "berry"),
        (f"{best['auc_roc']:.3f}", "AUC-ROC", "gold"),
        (f"{best['threshold']:.2f}", "Umbral validacion", "red"),
    ])
    content = f"""
{stats}
<div class="grid" style="grid-template-columns: 1.26fr .74fr;">
  <div class="panel">
    <h2>Ranking de modelos complejos en test</h2>
    {html_table(['Modelo', 'Umbral', 'Precision', 'Recall', 'F1', 'AUC-ROC'], rows)}
    <div class="callout gold" style="margin-top:.16in;">El umbral ganador se fija en validacion y se aplica una sola vez al test. La diferencia con SoftVoting es minima en F1, pero XGB AUC conserva mejor precision.</div>
  </div>
  <div class="grid" style="gap:.14in;">
    <div class="figure">
      <div class="figure-title"><span>Matriz de confusion · {esc(short(best['model']))}</span><span class="badge">test</span></div>
      {cm}
    </div>
    <div class="panel tight">
      <h3>Lectura del desbalance</h3>
      <p>La accuracy global es alta, pero solo {pct(ctx.balance['prevalence'], 1)} son positivos. Por eso se prioriza F1 de la clase cancer=1.</p>
    </div>
  </div>
</div>
"""
    return slide_shell(2, "Modelos ML complejos", "Boosting lidera con una ventaja estrecha", "La comparativa se centra en precision, recall, F1 y AUC-ROC sobre test, evitando decidir por accuracy.", content)


def slide_3(ctx: Context) -> str:
    mlp = ctx.metrics[ctx.metrics["model"] == "MLP"].iloc[0]
    best = ctx.metrics.iloc[0]
    history_svg = mlp_history_svg(ctx.mlp_history, height_in=1.48)
    threshold_svg = threshold_svg_from_df(ctx.mlp_threshold_search, "MLP", height_in=1.48)
    rows = [
        [short(best["model"]), f"{best['precision_positive']:.3f}", f"{best['recall_positive']:.3f}", f"{best['f1_positive']:.3f}", f"{best['auc_roc']:.3f}"],
        ["MLP", f"{mlp['precision_positive']:.3f}", f"{mlp['recall_positive']:.3f}", f"{mlp['f1_positive']:.3f}", f"{mlp['auc_roc']:.3f}"],
    ]
    content = f"""
<div class="grid" style="grid-template-columns: 1.05fr .95fr;">
  <div class="panel">
    <h2>Arquitectura validada</h2>
    <div class="arch">
      <div class="layer"><div class="l1">Entrada</div><div class="l2">features preprocesadas</div></div>
      <div class="layer dark"><div class="l1">Dense 256</div><div class="l2">BN + Dropout .25</div></div>
      <div class="layer dark"><div class="l1">Dense 128</div><div class="l2">BN + Dropout .25</div></div>
      <div class="layer dark"><div class="l1">Dense 64</div><div class="l2">BN + Dropout .20</div></div>
      <div class="layer"><div class="l1">Salida</div><div class="l2">1 sigmoid</div></div>
    </div>
    <div class="metric-strip" style="margin-top:.14in;">
      <div class="mini"><div class="n">46.913</div><div class="t">parametros</div></div>
      <div class="mini"><div class="n">{mlp['threshold']:.2f}</div><div class="t">umbral</div></div>
      <div class="mini"><div class="n">{mlp['f1_positive']:.3f}</div><div class="t">F1 test</div></div>
      <div class="mini"><div class="n">{mlp['recall_positive']:.3f}</div><div class="t">recall</div></div>
    </div>
    <div class="callout" style="margin-top:.14in;">Regularizacion: BatchNormalization, Dropout, Early Stopping, ReduceLROnPlateau y class_weight para compensar el desbalance.</div>
  </div>
  <div class="panel">
    <h2>Comparacion directa</h2>
    {html_table(['Modelo', 'Precision', 'Recall', 'F1', 'AUC-ROC'], rows)}
    <div style="height:.14in"></div>
    <p>La MLP mantiene recall competitivo, pero el mejor XGBoost ofrece mejor precision y F1. En este dataset tabular, boosting aprovecha mejor las interacciones con menor complejidad operativa.</p>
  </div>
</div>
<div class="grid" style="grid-template-columns: 1fr 1fr; margin-top:.16in;">
  <div class="figure"><div class="figure-title"><span>Curvas de perdida y F1 de validacion</span><span class="badge">Early Stopping</span></div>{history_svg}</div>
  <div class="figure"><div class="figure-title"><span>Barrido de umbral MLP</span><span class="badge">validacion</span></div>{threshold_svg}</div>
</div>
"""
    return slide_shell(3, "Red Neuronal MLP", "La red cumple el nucleo tecnico, pero no gana el ranking", "La arquitectura esta justificada por regularizacion, control del desbalance y seleccion de umbral fuera del test.", content)


def slide_4(ctx: Context) -> str:
    top = ctx.metrics.head(8)
    grouped = grouped_metric_svg(top)
    roc = img_data(FIGURES_DIR / "roc_curves.png")
    pr = img_data(FIGURES_DIR / "precision_recall_space.png")
    rows = []
    for idx, (_, row) in enumerate(top.head(5).iterrows(), start=1):
        rows.append([idx, short(row["model"]), row["model_type"], f"{row['precision_positive']:.3f}", f"{row['recall_positive']:.3f}", f"{row['f1_positive']:.3f}", f"{row['auc_pr']:.3f}"])
    content = f"""
<div class="grid" style="grid-template-columns: 1.16fr .84fr;">
  <div class="figure">
    <div class="figure-title"><span>Metricas comparadas</span><span class="badge">test</span></div>
    {grouped}
  </div>
  <div class="grid" style="grid-template-columns: 1fr 1fr; gap:.13in;">
    <div class="figure" style="height:2.37in;"><div class="figure-title"><span>ROC</span></div><img class="chart" src="{roc}" /></div>
    <div class="figure" style="height:2.37in;"><div class="figure-title"><span>Precision-Recall</span></div><img class="chart" src="{pr}" /></div>
  </div>
</div>
<div class="grid" style="grid-template-columns: 1.1fr .9fr; margin-top:.16in;">
  <div class="panel">
    <h2>Ranking final</h2>
    {html_table(['#', 'Modelo', 'Tipo', 'Precision', 'Recall', 'F1', 'AUC-PR'], rows)}
  </div>
  <div class="panel">
    <h2>Lectura global</h2>
    <ul>
      <li>Los boosting dominan F1 y AUC con diferencias muy estrechas.</li>
      <li>SoftVoting casi iguala al ganador y recupera mas positivos.</li>
      <li>ExtraTrees y XGBoost_cuda elevan recall, pagando precision.</li>
      <li>La MLP es solida como referencia avanzada, no como eleccion final.</li>
    </ul>
  </div>
</div>
"""
    return slide_shell(4, "Comparativa global", "La decision no es una metrica aislada", "El ranking combina rendimiento, sensibilidad y coste de errores para construir una recomendacion defendible.", content)


def slide_5(ctx: Context) -> str:
    best = ctx.metrics.iloc[0]
    soft = ctx.metrics[ctx.metrics["model"] == "ValidationSoftVoting"].iloc[0]
    threshold_svg = threshold_svg_from_df(ctx.threshold_search[ctx.threshold_search["model"] == "XGBoostAUC_cuda"], "XGBoostAUC_cuda", height_in=2.0)
    rows = [
        ["Ganador tecnico", short(best["model"]), f"F1={best['f1_positive']:.3f}; AUC={best['auc_roc']:.3f}"],
        ["Opcion sensible", "SoftVoting", f"Recall={soft['recall_positive']:.3f}; mas positivos detectados"],
        ["Uso recomendado", "2 zonas", "alto riesgo + zona gris para revision clinica"],
    ]
    content = f"""
<div class="callout berry" style="font-size:.16in; margin-bottom:.16in;">Recomendacion final: implantar <b>{esc(best['model'])}</b> como baseline operativo limpio de cribado experimental, manteniendo SoftVoting como alternativa cuando la sensibilidad pese mas que la precision.</div>
<div class="grid" style="grid-template-columns: .96fr 1.04fr;">
  <div class="figure">
    <div class="figure-title"><span>Trade-off precision / recall</span><span class="badge">XGB AUC</span></div>
    {threshold_svg}
  </div>
  <div class="grid" style="gap:.14in;">
    <div class="panel">
      <h2>Decision operativa</h2>
      {html_table(['Decision', 'Modelo', 'Motivo'], rows)}
    </div>
    <div class="panel tight">
      <h3>Regla propuesta</h3>
      <ul>
        <li><b>&gt;= 0.66:</b> alto riesgo, buena precision.</li>
        <li><b>0.43 - 0.66:</b> zona gris, revision clinica o prueba complementaria.</li>
        <li><b>&lt; 0.43:</b> bajo riesgo, nunca descarte medico autonomo.</li>
      </ul>
    </div>
  </div>
</div>
<div class="grid" style="grid-template-columns: 1fr 1fr; margin-top:.16in;">
  <div class="panel"><h2>Limitaciones</h2><ul><li>Dataset sintetico: exige validacion externa y temporal.</li><li>Faltan imagen medica, sintomas, tratamientos y estadio tumoral.</li><li>Variables economicas y vive se excluyen por posible fuga post-diagnostico.</li></ul></div>
  <div class="panel"><h2>Veredicto</h2><p>Los datos son suficientes para demostrar viabilidad tecnica y comparar estrategias. No son suficientes para despliegue clinico autonomo sin validacion real por hospital, periodo y poblacion.</p></div>
</div>
"""
    return slide_shell(5, "Viabilidad y decision", "Suficiente para viabilidad, no para diagnostico autonomo", "La recomendacion convierte el ranking de modelos en una estrategia prudente de cribado clinico.", content)


def stat_row(items: Iterable[tuple[str, str, str]]) -> str:
    accent = {"teal": PALETTE["teal"], "berry": PALETTE["berry"], "gold": PALETTE["gold"], "blue": PALETTE["blue"], "green": PALETTE["green"], "red": PALETTE["red"]}
    cards = []
    for value, label, color in items:
        cards.append(f'<div class="stat" style="--accent:{accent[color]}"><div class="value">{esc(value)}</div><div class="label">{esc(label)}</div></div>')
    return '<div class="stat-row">' + ''.join(cards) + '</div>'


def html_table(headers: list[object], rows: list[list[object]]) -> str:
    head = ''.join(f"<th>{cell}</th>" for cell in headers)
    body = ''.join("<tr>" + ''.join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def badge(text: str, kind: str = "") -> str:
    cls = "badge" + (f" {kind}" if kind else "")
    return f'<span class="{cls}">{esc(text)}</span>'


def metric_bars(df: pd.DataFrame, metric: str, min_value: float, max_value: float) -> str:
    rows = []
    span = max_value - min_value
    for _, row in df.iterrows():
        value = float(row[metric])
        width = max(0.02, min(1, (value - min_value) / span)) * 100
        color = MODEL_COLORS.get(str(row["model"]), PALETTE["teal"])
        rows.append(f"""
        <div class="bar-line"><span>{esc(short(row['model']))}</span><div class="track"><div class="fill" style="--w:{width:.1f}%; --c:{color}"></div></div><b>{value:.3f}</b></div>
        """)
    return '<div class="bars">' + ''.join(rows) + '</div>'


def confusion_svg(row: pd.Series) -> str:
    values = [[int(row["tn"]), int(row["fp"])], [int(row["fn"]), int(row["tp"])]]
    labels = [["TN", "FP"], ["FN", "TP"]]
    max_v = max(max(v) for v in values)
    cells = []
    for i in range(2):
        for j in range(2):
            value = values[i][j]
            alpha = 0.18 + 0.72 * value / max_v
            x = 18 + j * 170
            y = 34 + i * 110
            text_color = "#fbf8f0" if alpha > 0.62 else PALETTE["ink"]
            cells.append(f'<rect x="{x}" y="{y}" width="150" height="88" rx="3" fill="rgba(122,31,43,{alpha:.3f})"/><text x="{x+75}" y="{y+35}" text-anchor="middle" fill="{text_color}" font-size="18" font-weight="800">{labels[i][j]}</text><text x="{x+75}" y="{y+64}" text-anchor="middle" fill="{text_color}" font-size="24" font-weight="900">{fmt_int(value)}</text>')
    return f"""<svg viewBox="0 0 360 270" width="100%" height="2.36in" role="img" aria-label="Matriz de confusion">
      <text x="93" y="22" text-anchor="middle" fill="#756e66" font-size="13" font-weight="800">Pred. no cancer</text>
      <text x="263" y="22" text-anchor="middle" fill="#756e66" font-size="13" font-weight="800">Pred. cancer</text>
      <text x="7" y="82" transform="rotate(-90 7 82)" text-anchor="middle" fill="#756e66" font-size="13" font-weight="800">Real no</text>
      <text x="7" y="192" transform="rotate(-90 7 192)" text-anchor="middle" fill="#756e66" font-size="13" font-weight="800">Real cancer</text>
      {''.join(cells)}
    </svg>"""


def grouped_metric_svg(df: pd.DataFrame) -> str:
    metrics = [("precision_positive", "Precision", PALETTE["blue"]), ("recall_positive", "Recall", PALETTE["berry"]), ("f1_positive", "F1", PALETTE["teal"]), ("auc_roc", "AUC", PALETTE["gold"])]
    width, height = 650, 285
    left, top, bottom = 48, 26, 42
    plot_h = height - top - bottom
    group_w = (width - left - 20) / len(df)
    bar_w = group_w / 5
    items = []
    for i, (_, row) in enumerate(df.iterrows()):
        base_x = left + i * group_w
        for j, (metric, _label, color) in enumerate(metrics):
            value = float(row[metric])
            y = top + (1 - value) * plot_h
            h = value * plot_h
            x = base_x + j * bar_w + 4
            items.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-2:.1f}" height="{h:.1f}" rx="4" fill="{color}"/>')
        items.append(f'<text x="{base_x + group_w/2 - 4:.1f}" y="268" text-anchor="middle" fill="#756e66" font-size="10">{esc(short(row["model"]))}</text>')
    grid = ''.join(f'<line x1="{left}" x2="630" y1="{top + plot_h * (1-t):.1f}" y2="{top + plot_h * (1-t):.1f}" stroke="#d4c9b8" stroke-width="1"/><text x="9" y="{top + plot_h * (1-t)+4:.1f}" fill="#756e66" font-size="10">{t:.1f}</text>' for t in [0.5, 0.6, 0.7, 0.8, 0.9])
    legend = ''.join(f'<rect x="{240+i*84}" y="4" width="10" height="10" rx="1" fill="{color}"/><text x="{254+i*84}" y="13" fill="#756e66" font-size="10">{label}</text>' for i, (_, label, color) in enumerate(metrics))
    return f'<svg viewBox="0 0 {width} {height}" width="100%" height="2.42in">{grid}{items}{legend}</svg>'


def mlp_history_svg(history: pd.DataFrame, height_in: float = 2.0) -> str:
    return line_svg(
        history.reset_index().assign(epoch=lambda d: d["index"] + 1),
        "epoch",
        [("loss", "Train loss", PALETTE["blue"]), ("val_loss", "Val loss", PALETTE["berry"]), ("val_f1_best", "Val F1", PALETTE["teal"])],
        y_min=0.45,
        y_max=0.62,
        height_in=height_in,
    )


def threshold_svg_from_df(df: pd.DataFrame, title: str, height_in: float = 2.0) -> str:
    return line_svg(
        df,
        "threshold",
        [("precision_positive", "Precision", PALETTE["blue"]), ("recall_positive", "Recall", PALETTE["berry"]), ("f1_positive", "F1", PALETTE["teal"])],
        y_min=0.20,
        y_max=1.0,
        height_in=height_in,
    )


def line_svg(df: pd.DataFrame, x_col: str, series: list[tuple[str, str, str]], y_min: float, y_max: float, height_in: float) -> str:
    width, height = 620, 230
    left, right, top, bottom = 36, 12, 20, 36
    xs = df[x_col].astype(float).tolist()
    x_min, x_max = min(xs), max(xs)
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(value: float) -> float:
        return left + ((value - x_min) / (x_max - x_min)) * plot_w if x_max != x_min else left

    def sy(value: float) -> float:
        return top + (1 - ((value - y_min) / (y_max - y_min))) * plot_h

    grid = []
    for tick in [y_min, (y_min + y_max) / 2, y_max]:
        y = sy(tick)
        grid.append(f'<line x1="{left}" x2="{width-right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#d4c9b8"/><text x="4" y="{y+4:.1f}" fill="#756e66" font-size="10">{tick:.2f}</text>')
    paths = []
    for idx, (col, label, color) in enumerate(series):
        points = []
        for _, row in df.iterrows():
            value = float(row[col])
            if pd.notna(value):
                points.append(f'{sx(float(row[x_col])):.1f},{sy(value):.1f}')
        paths.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>')
        paths.append(f'<rect x="{260+idx*92}" y="4" width="10" height="10" rx="1" fill="{color}"/><text x="{274+idx*92}" y="13" fill="#756e66" font-size="10">{label}</text>')
    return f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height_in}in">{"".join(grid)}{"".join(paths)}</svg>'


def img_data(path: Path) -> str:
    mime = "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def export_pdf(html_path: Path, pdf_path: Path) -> None:
    chrome = find_chrome()
    if chrome:
        command = [
            chrome,
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            "--run-all-compositor-stages-before-draw",
            "--print-to-pdf-no-header",
            f"--print-to-pdf={pdf_path}",
            html_path.resolve().as_uri(),
        ]
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)
        return
    weasyprint = shutil.which("weasyprint")
    if weasyprint:
        subprocess.run([weasyprint, str(html_path), str(pdf_path)], check=True, cwd=PROJECT_ROOT)
        return
    raise RuntimeError("No se encontro Chrome/Chromium ni WeasyPrint para exportar HTML a PDF.")


def find_chrome() -> str | None:
    candidates = [
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def validate_pdf(path: Path) -> None:
    reader = PdfReader(str(path))
    if len(reader.pages) != SLIDE_COUNT:
        raise RuntimeError(f"El PDF debe tener {SLIDE_COUNT} diapositivas; tiene {len(reader.pages)}.")
    if path.stat().st_size < 100_000:
        raise RuntimeError("El PDF generado parece demasiado pequeno; revisa el render HTML.")


def render_previews(pdf_path: Path) -> None:
    pdftoppm = shutil.which("pdftoppm")
    if not pdftoppm:
        return
    for old in PREVIEW_DIR.glob("slide-*.png"):
        old.unlink()
    subprocess.run([pdftoppm, "-png", "-r", "130", str(pdf_path), str(PREVIEW_DIR / "slide")], check=True)


def short(value: object) -> str:
    return MODEL_SHORT.get(str(value), str(value).replace("_", " "))


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def pct(value: float, digits: int = 1) -> str:
    return f"{float(value) * 100:.{digits}f}%"


def fmt_int(value: object) -> str:
    return f"{int(round(float(value))):,}".replace(",", ".")


if __name__ == "__main__":
    raise SystemExit(main())
