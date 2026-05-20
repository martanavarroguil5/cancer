"""Graficas reproducibles para metricas y diapositivas."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, precision_recall_curve, roc_curve


PALETTE = ["#1b6ca8", "#2a9d8f", "#e76f51", "#6d597a", "#f4a261", "#4d908e"]


def plot_confusion_matrix(cm: np.ndarray, model_name: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred. no cancer", "Pred. cancer"],
        yticklabels=["Real no cancer", "Real cancer"],
        ax=ax,
    )
    ax.set_title(f"Matriz de confusion - {model_name}")
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_mlp_learning_curves(history: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = np.arange(1, len(history) + 1)
    axes[0].plot(epochs, history["loss"], label="train", color=PALETTE[0])
    axes[0].plot(epochs, history["val_loss"], label="validacion", color=PALETTE[2])
    axes[0].set_title("Loss MLP")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Binary crossentropy")
    axes[0].legend()
    if "val_f1_best" in history.columns:
        axes[1].plot(epochs, history["val_f1_best"], label="val_f1_best", color=PALETTE[2])
        axes[1].set_title("Seleccion de umbral MLP")
        axes[1].set_ylabel("F1 validacion")
    else:
        axes[1].plot(epochs, history["accuracy"], label="train", color=PALETTE[0])
        axes[1].plot(epochs, history["val_accuracy"], label="validacion", color=PALETTE[2])
        axes[1].set_title("Accuracy MLP")
        axes[1].set_ylabel("Accuracy")
    axes[1].set_xlabel("Epoca")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_roc_curves(y_true, probabilities: dict[str, np.ndarray], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5.2))
    for idx, (name, proba) in enumerate(probabilities.items()):
        fpr, tpr, _ = roc_curve(y_true, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})", color=PALETTE[idx % len(PALETTE)])
    ax.plot([0, 1], [0, 1], linestyle="--", color="#6c757d", linewidth=1)
    ax.set_title("Curvas ROC en test")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_metric_comparison(metrics: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metric_columns = ["f1_positive", "recall_positive", "precision_positive", "auc_roc", "auc_pr"]
    metric_columns = [column for column in metric_columns if column in metrics.columns]
    plot_df = metrics[["model"] + metric_columns].melt(id_vars="model", var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    sns.barplot(data=plot_df, x="model", y="value", hue="metric", palette=PALETTE[: len(metric_columns)], ax=ax)
    ax.set_ylim(0, 1.02)
    ax.set_title("Comparativa de metricas en test")
    ax.set_xlabel("")
    ax.set_ylabel("Valor")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_precision_recall_space(y_true, probabilities: dict[str, np.ndarray], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5.2))
    baseline = float(np.mean(y_true))
    for idx, (name, proba) in enumerate(probabilities.items()):
        precision, recall, _ = precision_recall_curve(y_true, proba)
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, label=f"{name} (AUC-PR={pr_auc:.3f})", color=PALETTE[idx % len(PALETTE)])
    ax.axhline(baseline, linestyle="--", linewidth=1, color="#6c757d", label=f"Prevalencia={baseline:.3f}")
    ax.set_title("Espacio Precision-Recall en test")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
