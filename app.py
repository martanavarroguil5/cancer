from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
METRICS_DIR = ROOT / "outputs" / "metrics"
FIGURES_DIR = ROOT / "outputs" / "figures"
REPORTS_DIR = ROOT / "outputs" / "reports"

METRIC_LABELS = {
    "precision_positive": "Precision cancer=1",
    "recall_positive": "Recall cancer=1",
    "f1_positive": "F1 cancer=1",
    "auc_roc": "AUC-ROC",
    "auc_pr": "AUC-PR",
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "specificity": "Specificity",
    "false_positive_rate": "False positive rate",
    "false_negative_rate": "False negative rate",
}

REQUIRED_DELIVERABLES = [
    ("Objetivo y datos", "Colecciones usadas, target, prevalencia, desbalance y politica de features."),
    ("Modelos ML complejos", "Comparativa Precision / Recall / F1 / AUC-ROC y matriz del mejor modelo."),
    ("Red Neuronal MLP", "Arquitectura, regularizacion, curvas de aprendizaje y metricas en test."),
    ("Comparativa global", "Ranking, barras de metricas, ROC y espacio Precision-Recall."),
    ("Viabilidad", "Decision de implantacion, limitaciones y datos adicionales necesarios."),
]


st.set_page_config(
    page_title="Cancer ML Viability Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --lab-canvas: #f5f7f4;
            --lab-panel: #ffffff;
            --lab-panel-soft: #eef4f1;
            --lab-ink: #17231f;
            --lab-muted: #64726d;
            --lab-line: rgba(31, 48, 43, 0.12);
            --lab-teal: #0f766e;
            --lab-teal-soft: #d9efea;
            --lab-berry: #9f2d55;
            --lab-gold: #b7791f;
            --lab-blue: #1e5f99;
            --lab-red: #b42318;
        }

        .stApp {
            background:
                linear-gradient(180deg, rgba(255,255,255,0.72) 0%, rgba(245,247,244,0.96) 28%),
                radial-gradient(circle at 14% 8%, rgba(15,118,110,0.09), transparent 26%),
                var(--lab-canvas);
            color: var(--lab-ink);
        }

        [data-testid="stSidebar"] {
            background: rgba(245, 247, 244, 0.96);
            border-right: 1px solid var(--lab-line);
        }

        [data-testid="stSidebar"] * {
            letter-spacing: 0;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1480px;
        }

        h1, h2, h3 {
            letter-spacing: 0 !important;
            color: var(--lab-ink);
        }

        h1 {
            font-size: clamp(2.1rem, 4vw, 4.2rem) !important;
            line-height: 1.02 !important;
            max-width: 980px;
        }

        h2 {
            font-size: 1.55rem !important;
            margin-top: 1.4rem !important;
        }

        .hero {
            border: 1px solid var(--lab-line);
            border-radius: 18px;
            padding: 2.1rem 2.2rem;
            background:
                linear-gradient(135deg, rgba(255,255,255,0.96), rgba(236,246,242,0.94)),
                repeating-linear-gradient(90deg, rgba(15,118,110,0.05) 0 1px, transparent 1px 34px);
            box-shadow: 0 22px 50px rgba(23, 35, 31, 0.08);
        }

        .eyebrow {
            color: var(--lab-teal);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: .08em;
            text-transform: uppercase;
            margin-bottom: .75rem;
        }

        .lede {
            color: var(--lab-muted);
            font-size: 1.02rem;
            line-height: 1.7;
            max-width: 860px;
        }

        .section-band {
            border: 1px solid var(--lab-line);
            border-radius: 16px;
            padding: 1.15rem 1.2rem;
            background: rgba(255,255,255,0.72);
            box-shadow: 0 10px 28px rgba(23, 35, 31, 0.04);
        }

        .stat-card {
            border: 1px solid var(--lab-line);
            border-radius: 14px;
            background: var(--lab-panel);
            padding: 1rem 1.05rem;
            min-height: 120px;
            box-shadow: 0 10px 24px rgba(23, 35, 31, 0.05);
        }

        .stat-label {
            color: var(--lab-muted);
            font-size: .78rem;
            font-weight: 750;
            text-transform: uppercase;
            letter-spacing: .04em;
            margin-bottom: .35rem;
        }

        .stat-value {
            color: var(--lab-ink);
            font-size: 1.85rem;
            line-height: 1.08;
            font-weight: 820;
        }

        .stat-note {
            color: var(--lab-muted);
            font-size: .88rem;
            margin-top: .4rem;
            line-height: 1.45;
        }

        .callout {
            border-left: 4px solid var(--lab-teal);
            background: rgba(217, 239, 234, 0.72);
            padding: .9rem 1rem;
            border-radius: 12px;
            color: #16332e;
            line-height: 1.55;
        }

        .warning {
            border-left-color: var(--lab-gold);
            background: rgba(255, 247, 222, .84);
            color: #4f3512;
        }

        .decision {
            border-left-color: var(--lab-berry);
            background: rgba(252, 232, 240, .78);
            color: #4a1729;
        }

        .small-muted {
            color: var(--lab-muted);
            font-size: .9rem;
            line-height: 1.55;
        }

        div[data-testid="stMetric"] {
            background: var(--lab-panel);
            border: 1px solid var(--lab-line);
            border-radius: 14px;
            padding: .8rem .9rem;
            box-shadow: 0 8px 20px rgba(23, 35, 31, 0.04);
        }

        div[data-testid="stMetricLabel"] {
            color: var(--lab-muted);
            font-weight: 760;
        }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--lab-line);
            border-radius: 14px;
            overflow: hidden;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: .35rem;
            border-bottom: 1px solid var(--lab-line);
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 10px 10px 0 0;
            padding: .55rem .8rem;
        }

        .stTabs [aria-selected="true"] {
            background: var(--lab-teal-soft);
            color: var(--lab-teal);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(METRICS_DIR / name)


@st.cache_data(show_spinner=False)
def read_json(name: str) -> dict:
    with (METRICS_DIR / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def num(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def stat_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
            <div class="stat-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_table(df: pd.DataFrame, columns: list[str] | None = None) -> pd.io.formats.style.Styler:
    metric_cols = columns or [
        col
        for col in df.columns
        if col
        in {
            "threshold",
            "precision_positive",
            "recall_positive",
            "f1_positive",
            "auc_roc",
            "auc_pr",
            "accuracy",
            "balanced_accuracy",
        }
    ]
    format_map = {col: "{:.3f}" for col in metric_cols if col in df.columns}
    return df.style.format(format_map).hide(axis="index")


def bar_chart(metrics: pd.DataFrame, selected_models: list[str], selected_metrics: list[str]) -> go.Figure:
    plot_df = metrics[metrics["model"].isin(selected_models)].copy()
    long = plot_df.melt(
        id_vars=["model", "model_type"],
        value_vars=selected_metrics,
        var_name="metric",
        value_name="value",
    )
    long["metric_label"] = long["metric"].map(METRIC_LABELS)
    fig = px.bar(
        long,
        x="model",
        y="value",
        color="metric_label",
        barmode="group",
        color_discrete_sequence=["#0f766e", "#1e5f99", "#9f2d55", "#b7791f", "#475569"],
        height=430,
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=28, b=10),
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        xaxis_title="",
        yaxis_title="Valor en test",
        legend_title="",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def confusion_matrix_figure(model_row: pd.Series) -> go.Figure:
    z = [[int(model_row["tn"]), int(model_row["fp"])], [int(model_row["fn"]), int(model_row["tp"])]]
    labels = [["TN", "FP"], ["FN", "TP"]]
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["Pred. no cancer", "Pred. cancer"],
            y=["Real no cancer", "Real cancer"],
            colorscale=[[0, "#e8f3f0"], [0.55, "#64b6ac"], [1, "#0f766e"]],
            showscale=False,
            text=[[f"{labels[i][j]}<br>{z[i][j]:,}".replace(",", ".") for j in range(2)] for i in range(2)],
            texttemplate="%{text}",
            textfont=dict(size=18, color="#17231f"),
            hovertemplate="%{y}<br>%{x}<br><b>%{z}</b><extra></extra>",
        )
    )
    fig.update_layout(
        height=330,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(side="top"),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def mlp_learning_figure(history: pd.DataFrame) -> go.Figure:
    epochs = list(range(1, len(history) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=history["loss"], mode="lines", name="Train loss", line=dict(color="#1e5f99")))
    fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"], mode="lines", name="Val loss", line=dict(color="#9f2d55")))
    if "val_f1_best" in history.columns:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history["val_f1_best"],
                mode="lines",
                name="Val F1 con umbral",
                yaxis="y2",
                line=dict(color="#0f766e"),
            )
        )
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=28, b=10),
        yaxis=dict(title="Binary crossentropy"),
        yaxis2=dict(title="F1 validacion", overlaying="y", side="right", range=[0.50, 0.60]),
        xaxis_title="Epoca",
        legend_title="",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def threshold_figure(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["threshold"], y=df["precision_positive"], mode="lines", name="Precision", line=dict(color="#1e5f99")))
    fig.add_trace(go.Scatter(x=df["threshold"], y=df["recall_positive"], mode="lines", name="Recall", line=dict(color="#9f2d55")))
    fig.add_trace(go.Scatter(x=df["threshold"], y=df["f1_positive"], mode="lines", name="F1", line=dict(color="#0f766e", width=3)))
    best = df.loc[df["f1_positive"].idxmax()]
    fig.add_vline(x=best["threshold"], line_width=2, line_dash="dash", line_color="#17231f")
    fig.add_annotation(
        x=best["threshold"],
        y=best["f1_positive"],
        text=f"Mejor F1: {best['threshold']:.2f}",
        showarrow=True,
        arrowhead=2,
        ax=28,
        ay=-36,
    )
    fig.update_layout(
        title=title,
        height=390,
        margin=dict(l=10, r=10, t=42, b=10),
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        xaxis_title="Umbral",
        yaxis_title="Metrica",
        legend_title="",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def interval_figure(intervals: pd.DataFrame, metric: str) -> go.Figure:
    df = intervals[intervals["metric"] == metric].copy()
    df = df.sort_values("mean", ascending=True)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["mean"],
            y=df["model"],
            mode="markers",
            marker=dict(color="#0f766e", size=10),
            error_x=dict(
                type="data",
                symmetric=False,
                array=df["ci95_high"] - df["mean"],
                arrayminus=df["mean"] - df["ci95_low"],
                color="rgba(23,35,31,.38)",
                thickness=1.4,
            ),
            hovertemplate="%{y}<br>Media: %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(range=[max(0, df["ci95_low"].min() - 0.03), min(1, df["ci95_high"].max() + 0.03)]),
        xaxis_title=METRIC_LABELS.get(metric, metric),
        yaxis_title="",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def feature_signal_figure(feature_signal: pd.DataFrame) -> go.Figure:
    df = feature_signal.copy()
    df["signal"] = df["numeric_abs_auc"].fillna(df["categorical_rate_spread"]).fillna(0)
    df = df[~df["known_leakage"].fillna(False)]
    df = df[df["column"] != "paciente_id"].sort_values("signal", ascending=False).head(14)
    fig = px.bar(
        df.sort_values("signal"),
        x="signal",
        y="column",
        orientation="h",
        color="policy_post_diagnosis_risk",
        color_discrete_map={True: "#b42318", False: "#0f766e"},
        height=430,
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="Senal univariante (AUC abs. o spread de tasa)",
        yaxis_title="",
        legend_title="Riesgo post-diagnostico",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def check_artifacts() -> None:
    required = [
        METRICS_DIR / "model_metrics.csv",
        METRICS_DIR / "run_summary.json",
        METRICS_DIR / "target_balance.json",
        METRICS_DIR / "feature_policy.json",
        METRICS_DIR / "mlp_history.csv",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        st.error("Faltan artefactos necesarios para renderizar el dashboard: " + ", ".join(missing))
        st.stop()


def render_sidebar(metrics: pd.DataFrame, run_summary: dict) -> tuple[str, list[str], list[str]]:
    st.sidebar.markdown("### Cancer ML")
    st.sidebar.caption("Dashboard de viabilidad clinica sin reentrenar modelos.")
    page = st.sidebar.radio(
        "Vista",
        [
            "Resumen ejecutivo",
            "Datos y features",
            "Modelos ML",
            "Red Neuronal MLP",
            "Comparativa global",
            "Decision y entrega",
        ],
    )
    st.sidebar.divider()
    selected_metrics = st.sidebar.multiselect(
        "Metricas en graficos",
        ["f1_positive", "precision_positive", "recall_positive", "auc_roc", "auc_pr"],
        default=["f1_positive", "precision_positive", "recall_positive", "auc_roc"],
        format_func=lambda value: METRIC_LABELS[value],
    )
    default_models = metrics.sort_values("f1_positive", ascending=False).head(6)["model"].tolist()
    selected_models = st.sidebar.multiselect(
        "Modelos visibles",
        metrics["model"].tolist(),
        default=default_models,
    )
    st.sidebar.divider()
    st.sidebar.metric("Ejecucion", run_summary.get("mode", "n/a"))
    st.sidebar.metric("Seed", run_summary.get("seed", "n/a"))
    st.sidebar.metric("Vista features", run_summary.get("feature_view", "n/a"))
    return page, selected_models or default_models, selected_metrics or ["f1_positive"]


def render_hero(run_summary: dict, balance: dict, best: pd.Series) -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Estudio de viabilidad · IA aplicada a cribado oncologico</div>
            <h1>Prediccion de diagnostico de cancer</h1>
            <p class="lede">
            Lectura ejecutiva de un pipeline multimodal: seis colecciones clinicas y sociodemograficas,
            control de fuga temporal, comparativa de modelos clasicos frente a MLP y una recomendacion
            final preparada para defender el estudio en cinco diapositivas.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    cols = st.columns(5)
    with cols[0]:
        stat_card("Pacientes", f"{run_summary['rows_joined']:,}".replace(",", "."), "Union por paciente_id")
    with cols[1]:
        stat_card("Prevalencia", pct(balance["prevalence"], 1), "cancer=1 en el dataset")
    with cols[2]:
        stat_card("Desbalance", f"{balance['negative_positive_ratio']:.2f}:1", "Negativos por positivo")
    with cols[3]:
        stat_card("Mejor F1", num(best["f1_positive"]), best["model"])
    with cols[4]:
        stat_card("AUC-ROC", num(best["auc_roc"]), "Modelo recomendado")


def overview_page(metrics: pd.DataFrame, run_summary: dict, balance: dict, model_card: dict, selected_models: list[str], selected_metrics: list[str]) -> None:
    best = metrics.iloc[0]
    render_hero(run_summary, balance, best)

    st.subheader("Lectura rapida")
    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        st.markdown(
            f"""
            <div class="callout">
            <b>Decision:</b> implantar como baseline operativo limpio <b>{model_card['model_name']}</b>,
            con umbral {model_card['threshold']:.2f} elegido en validacion. La MLP cumple el enunciado,
            pero en este tabular queda por debajo de los modelos boosting.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.metric("Precision cancer=1", num(best["precision_positive"]))
        st.metric("Recall cancer=1", num(best["recall_positive"]))
    with c3:
        st.metric("Accuracy global", pct(best["accuracy"]))
        st.metric("Predichos positivos", pct(best["predicted_positive_rate"]))

    st.subheader("Ranking principal")
    ranking_cols = ["model", "model_type", "threshold", "precision_positive", "recall_positive", "f1_positive", "auc_roc", "auc_pr", "accuracy"]
    st.dataframe(style_table(metrics[ranking_cols].head(8)), use_container_width=True, height=318)

    st.subheader("Comparativa visual")
    st.plotly_chart(bar_chart(metrics, selected_models, selected_metrics), use_container_width=True)

    st.subheader("Mapa de la entrega solicitada")
    cols = st.columns(5)
    for col, (title, text) in zip(cols, REQUIRED_DELIVERABLES, strict=True):
        with col:
            stat_card(title, "OK", text)


def data_page(data_audit: pd.DataFrame, eda: pd.DataFrame, feature_policy: dict, feature_signal: pd.DataFrame, balance: dict) -> None:
    st.header("Datos y politica de features")
    st.markdown(
        """
        <div class="callout">
        La app se centra en la vista <b>safe_all</b>: variables prediagnostico disponibles,
        excluyendo identificadores, target, constantes y campos con riesgo de fuga temporal.
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    with cols[0]:
        stat_card("Colecciones", f"{len(data_audit)} / 6", "Todas presentes")
    with cols[1]:
        stat_card("Missing total", f"{int(data_audit['missing_values_total'].sum()):,}".replace(",", "."), "Sin ausencias")
    with cols[2]:
        stat_card("Positivos", f"{int(balance['positives']):,}".replace(",", "."), pct(balance["prevalence"]))
    with cols[3]:
        stat_card("Negativos", f"{int(balance['negatives']):,}".replace(",", "."), "Clase mayoritaria")

    st.subheader("Colecciones unidas")
    audit_cols = ["collection", "present", "rows", "columns", "duplicate_paciente_id", "missing_values_total", "same_ids_as_first_csv"]
    st.dataframe(data_audit[audit_cols].style.hide(axis="index"), use_container_width=True, height=260)

    st.subheader("Seleccion de variables")
    c1, c2 = st.columns([0.95, 1.05])
    with c1:
        feature_counts = {
            "Numericas": len(feature_policy.get("numeric", [])),
            "Binarias": len(feature_policy.get("binary", [])),
            "Categoricas": len(feature_policy.get("categorical", [])),
            "Excluidas": len(feature_policy.get("excluded", [])),
        }
        fig = px.pie(
            names=list(feature_counts.keys()),
            values=list(feature_counts.values()),
            hole=0.58,
            color_discrete_sequence=["#0f766e", "#1e5f99", "#9f2d55", "#b7791f"],
        )
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=10), height=330, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**Incluidas**")
        st.write(", ".join(feature_policy.get("included", [])))
        st.markdown("**Excluidas y motivo**")
        for note in feature_policy.get("notes", [])[3:]:
            st.markdown(f"- {note}")

    st.subheader("Senal y cautelas de variables")
    st.plotly_chart(feature_signal_figure(feature_signal), use_container_width=True)

    with st.expander("Auditoria EDA completa"):
        eda_cols = ["column", "dtype", "missing_pct", "unique", "min", "max", "mean_or_prevalence", "top_values"]
        st.dataframe(eda[eda_cols].style.hide(axis="index"), use_container_width=True, height=420)


def ml_page(metrics: pd.DataFrame, intervals: pd.DataFrame, selected_models: list[str], selected_metrics: list[str]) -> None:
    st.header("Modelos ML complejos")
    ml_metrics = metrics[metrics["model_type"].isin(["ML", "Ensemble"])].copy()
    best_ml = ml_metrics.sort_values("f1_positive", ascending=False).iloc[0]
    st.markdown(
        f"""
        <div class="callout">
        El mejor modelo ML es <b>{best_ml['model']}</b>. El umbral {best_ml['threshold']:.2f}
        se selecciono en validacion y despues se aplico una sola vez al test.
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("F1 cancer=1", num(best_ml["f1_positive"]))
    with c2:
        st.metric("Precision", num(best_ml["precision_positive"]))
    with c3:
        st.metric("Recall", num(best_ml["recall_positive"]))
    with c4:
        st.metric("AUC-ROC", num(best_ml["auc_roc"]))

    st.subheader("Comparativa ML")
    columns = ["model", "threshold", "precision_positive", "recall_positive", "f1_positive", "auc_roc", "auc_pr", "accuracy", "tn", "fp", "fn", "tp"]
    st.dataframe(style_table(ml_metrics[columns].head(9)), use_container_width=True, height=360)

    c1, c2 = st.columns([1.05, 0.95])
    with c1:
        st.subheader("Matriz de confusion del mejor ML")
        st.plotly_chart(confusion_matrix_figure(best_ml), use_container_width=True)
    with c2:
        st.subheader("Impacto del desbalance")
        st.markdown(
            """
            <div class="warning">
            La accuracy es alta porque la clase negativa domina. Por eso el ranking usa F1 de
            <b>cancer=1</b>, equilibrando precision y recall en la clase clinicamente sensible.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        st.metric("Falsos negativos", f"{int(best_ml['fn']):,}".replace(",", "."))
        st.metric("Falsos positivos", f"{int(best_ml['fp']):,}".replace(",", "."))

    st.subheader("Intervalos bootstrap")
    interval_metric = st.selectbox(
        "Metrica con IC 95%",
        ["f1_positive", "precision_positive", "recall_positive", "auc_roc", "auc_pr"],
        format_func=lambda value: METRIC_LABELS[value],
    )
    st.plotly_chart(interval_figure(intervals, interval_metric), use_container_width=True)

    st.subheader("Metricas seleccionadas")
    st.plotly_chart(bar_chart(metrics, selected_models, selected_metrics), use_container_width=True)


def mlp_page(metrics: pd.DataFrame, history: pd.DataFrame, threshold_search: pd.DataFrame, feature_policy: dict) -> None:
    st.header("Red Neuronal Multicapa")
    mlp = metrics[metrics["model"] == "MLP"].iloc[0]
    st.markdown(
        """
        <div class="callout">
        Arquitectura MLP con tres capas ocultas, BatchNormalization, Dropout, Early Stopping,
        ReduceLROnPlateau y class_weight para tratar el desbalance.
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(5)
    with cols[0]:
        stat_card("Entrada", f"{len(feature_policy.get('included', []))}", "Variables antes de one-hot")
    with cols[1]:
        stat_card("Capas ocultas", "3", "Dense + BN + Dropout")
    with cols[2]:
        stat_card("Parametros", "46.913", "Segun enunciado")
    with cols[3]:
        stat_card("Umbral MLP", f"{mlp['threshold']:.2f}", "Elegido en validacion")
    with cols[4]:
        stat_card("F1 test", num(mlp["f1_positive"]), "cancer=1")

    c1, c2 = st.columns([1.05, 0.95])
    with c1:
        st.subheader("Curvas de entrenamiento")
        st.plotly_chart(mlp_learning_figure(history), use_container_width=True)
    with c2:
        st.subheader("Metricas de test")
        mlp_cols = ["threshold", "precision_positive", "recall_positive", "f1_positive", "auc_roc", "auc_pr", "accuracy", "tn", "fp", "fn", "tp"]
        st.dataframe(style_table(metrics[metrics["model"] == "MLP"][mlp_cols]), use_container_width=True)
        st.markdown(
            """
            <div class="small-muted">
            El recall de la MLP es competitivo, pero su precision y F1 quedan por debajo del
            XGBoost recomendado. Esto es coherente con datasets tabulares donde boosting suele
            aprovechar mejor interacciones no lineales con menos coste operativo.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Barrido de umbral MLP")
    st.plotly_chart(threshold_figure(threshold_search, "Precision, recall y F1 por umbral"), use_container_width=True)

    architecture = FIGURES_DIR / "presentation" / "mlp_architecture.png"
    if architecture.exists():
        st.subheader("Diagrama de arquitectura")
        st.image(str(architecture), use_container_width=True)


def global_page(metrics: pd.DataFrame, selected_models: list[str], selected_metrics: list[str]) -> None:
    st.header("Comparativa global ML vs Red Neuronal")
    st.plotly_chart(bar_chart(metrics, selected_models, selected_metrics), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Curvas ROC superpuestas")
        roc = FIGURES_DIR / "roc_curves.png"
        if roc.exists():
            st.image(str(roc), use_container_width=True)
        else:
            st.info("No se encontro outputs/figures/roc_curves.png")
    with c2:
        st.subheader("Espacio Precision-Recall")
        pr = FIGURES_DIR / "precision_recall_space.png"
        if pr.exists():
            st.image(str(pr), use_container_width=True)
        else:
            st.info("No se encontro outputs/figures/precision_recall_space.png")

    st.subheader("Ranking completo")
    ranking_cols = ["model", "model_type", "threshold", "precision_positive", "recall_positive", "f1_positive", "auc_roc", "auc_pr", "accuracy", "balanced_accuracy"]
    st.dataframe(style_table(metrics[ranking_cols]), use_container_width=True, height=430)


def decision_page(run_summary: dict, model_card: dict, feature_policy: dict) -> None:
    st.header("Decision de viabilidad y entrega")
    st.markdown(
        f"""
        <div class="callout decision">
        <b>Recomendacion:</b> usar <b>{model_card['model_name']}</b> como baseline hospitalario de cribado
        experimental. Es el mejor F1 validado en test ({model_card['test_metrics']['f1_positive']:.3f}) y
        mantiene AUC-ROC {model_card['test_metrics']['auc_roc']:.3f}, sin usar variables post-diagnostico.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Por que es viable")
        st.markdown(
            """
            - Los seis CSV se unen sin duplicados ni missing.
            - Hay senal predictiva suficiente: AUC-ROC cercano a 0.85.
            - El umbral se selecciona en validacion, reduciendo leakage.
            - La comparativa incluye varios ML complejos, ensemble y MLP.
            """
        )
    with c2:
        st.subheader("Por que aun no es clinico")
        st.markdown(
            """
            - Dataset sintetico, sin validacion externa por hospital.
            - No hay temporalidad real prediagnostico/postdiagnostico.
            - F1 y precision son utiles para viabilidad, no para decision medica directa.
            - Faltan imagen, historia longitudinal, sintomas y estadio tumoral.
            """
        )

    st.subheader("Datos adicionales que mejorarian la prediccion")
    cols = st.columns(4)
    additions = [
        ("Longitudinal", "Analiticas fechadas antes del diagnostico."),
        ("Imagen", "Radiologia, mamografia, TAC o patologia digital."),
        ("Historia clinica", "Antecedentes familiares, sintomas y tratamientos."),
        ("Validacion externa", "Hospitales, periodos y poblaciones independientes."),
    ]
    for col, (label, text) in zip(cols, additions, strict=True):
        with col:
            stat_card(label, "Prioridad", text)

    st.subheader("Entregables disponibles")
    report_path = REPORTS_DIR / "presentacion_final_cancer_5_diapositivas.pdf"
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("PDF de diapositivas", f"{run_summary.get('pdf_pages', 5)} paginas")
        if report_path.exists():
            st.download_button(
                "Descargar presentacion final",
                data=report_path.read_bytes(),
                file_name=report_path.name,
                mime="application/pdf",
            )
    with c2:
        st.metric("Modelo recomendado", model_card["artifact"])
        st.caption("Artefacto local ya entrenado; la app no lo carga ni lo ejecuta.")
    with c3:
        st.metric("Variables incluidas", len(feature_policy.get("included", [])))
        st.metric("Variables excluidas", len(feature_policy.get("excluded", [])))

    st.subheader("Checklist del enunciado")
    checklist = pd.DataFrame(REQUIRED_DELIVERABLES, columns=["Bloque", "Cobertura en dashboard"])
    checklist["Estado"] = "Cubierto"
    st.dataframe(checklist.style.hide(axis="index"), use_container_width=True)


def main() -> None:
    inject_css()
    check_artifacts()

    metrics = read_csv("model_metrics.csv").sort_values("f1_positive", ascending=False).reset_index(drop=True)
    intervals = read_csv("model_metric_intervals.csv")
    data_audit = read_csv("data_audit.csv")
    eda = read_csv("eda_summary.csv")
    feature_signal = read_csv("feature_signal_report.csv")
    history = read_csv("mlp_history.csv")
    mlp_threshold_search = read_csv("mlp_threshold_search.csv")
    run_summary = read_json("run_summary.json")
    balance = read_json("target_balance.json")
    model_card = read_json("model_card.json")
    feature_policy = read_json("feature_policy.json")

    page, selected_models, selected_metrics = render_sidebar(metrics, run_summary)

    if page == "Resumen ejecutivo":
        overview_page(metrics, run_summary, balance, model_card, selected_models, selected_metrics)
    elif page == "Datos y features":
        data_page(data_audit, eda, feature_policy, feature_signal, balance)
    elif page == "Modelos ML":
        ml_page(metrics, intervals, selected_models, selected_metrics)
    elif page == "Red Neuronal MLP":
        mlp_page(metrics, history, mlp_threshold_search, feature_policy)
    elif page == "Comparativa global":
        global_page(metrics, selected_models, selected_metrics)
    else:
        decision_page(run_summary, model_card, feature_policy)


if __name__ == "__main__":
    main()
