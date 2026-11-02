from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(matrix: np.ndarray, labels: list[str], output_path: str | Path, *, normalize: bool = False) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if normalize:
        matrix = matrix.astype(np.float64)
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        matrix = matrix / row_sums
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix" if not normalize else "Normalized confusion matrix")
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            text = f"{value:.2f}" if normalize else f"{int(value)}"
            ax.text(col_index, row_index, text, ha="center", va="center", color="black" if value < 0.7 else "white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_training_curves(history: dict[str, list[float]], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for key in ("train_loss", "val_loss", "train_accuracy", "val_accuracy"):
        if not history.get(key):
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(history[key], label=key)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(key)
        ax.set_title(f"{key} curve")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{key}.png", dpi=200)
        plt.close(fig)


def plot_metric_comparison(results: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(results.keys())
    values = [float(results[label].get("macro_f1", 0.0)) if isinstance(results[label], dict) else 0.0 for label in labels]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values)
    ax.set_ylabel("Macro F1")
    ax.set_title("Model comparison")
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


__all__ = ["plot_confusion_matrix", "plot_training_curves", "plot_metric_comparison"]
