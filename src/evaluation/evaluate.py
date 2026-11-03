
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, SequentialSampler

from src.models.dataset import CLASS_NAMES, CLASS_TO_INDEX


def compute_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("y_true and y_pred must not be empty.")

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    if class_names is None:
        class_names = list(CLASS_NAMES)

    if not class_names or len(set(class_names)) != len(class_names):
        raise ValueError("class_names must be non-empty and contain unique names.")

    labels = [
        CLASS_TO_INDEX[name] if name in CLASS_TO_INDEX else idx
        for idx, name in enumerate(class_names)
    ]

    if len(set(labels)) != len(labels):
        raise ValueError("Each class name must map to a unique label index.")

    valid_labels = set(labels)
    if not set(y_true.tolist()).issubset(valid_labels):
        raise ValueError("y_true contains a label not represented in class_names.")

    if not set(y_pred.tolist()).issubset(valid_labels):
        raise ValueError("y_pred contains a label not represented in class_names.")

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    results = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "macro_precision": float(report.get("macro avg", {}).get("precision", 0.0)),
        "macro_recall": float(report.get("macro avg", {}).get("recall", 0.0)),
        "macro_f1": float(report.get("macro avg", {}).get("f1-score", 0.0)),
        "per_class": {},
    }

    for name in class_names:
        class_stats = report.get(name, {})
        results["per_class"][name] = {
            "precision": float(class_stats.get("precision", 0.0)),
            "recall": float(class_stats.get("recall", 0.0)),
            "f1": float(class_stats.get("f1-score", 0.0)),
            "support": int(class_stats.get("support", 0)),
        }

    return results


def _get_sample_ids(dataset, count: int) -> list[str]:
    """
    Get real sample IDs when available and safely aligned with evaluation order.
    Otherwise, generate fallback IDs.
    """
    base_dataset = dataset

    if isinstance(dataset, DataLoader):
        # IDs can only be matched safely when the loader preserves dataset order.
        if not isinstance(dataset.sampler, SequentialSampler):
            return [f"sample_{i}" for i in range(count)]

        base_dataset = dataset.dataset

    samples = getattr(base_dataset, "samples", None)

    if samples is not None and len(samples) == count:
        ids = [
            getattr(sample, "sample_id", None)
            for sample in samples
        ]

        if all(isinstance(sample_id, str) and sample_id for sample_id in ids):
            return ids

    return [f"sample_{i}" for i in range(count)]


def evaluate_model(
    model: torch.nn.Module,
    dataset,
    *,
    device: str = "cpu",
    output_dir: str | Path | None = None,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    if class_names is None:
        class_names = list(CLASS_NAMES)

    device_obj = torch.device(device)
    model.to(device_obj)
    model.eval()

    if not isinstance(dataset, DataLoader):
        dataset = DataLoader(dataset, batch_size=32, shuffle=False)

    true_labels: list[int] = []
    predicted_labels: list[int] = []
    confidences: list[float] = []

    with torch.no_grad():
        for batch in dataset:
            if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                raise TypeError("Dataset must yield (inputs, targets) tuples.")

            inputs, targets = batch
            inputs = inputs.to(device_obj)
            targets = targets.to(device_obj)

            logits = model(inputs)

            if logits.ndim != 2 or logits.shape[1] != len(class_names):
                raise ValueError(
                    "Model output must have shape [batch_size, number_of_classes]."
                )

            probs = torch.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1)

            true_labels.extend(targets.cpu().tolist())
            predicted_labels.extend(predictions.cpu().tolist())
            confidences.extend(probs.max(dim=1).values.cpu().tolist())

    metrics = compute_metrics(
        true_labels,
        predicted_labels,
        class_names=class_names,
    )

    sample_ids = _get_sample_ids(dataset, len(true_labels))

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=[
            CLASS_TO_INDEX[name] if name in CLASS_TO_INDEX else idx
            for idx, name in enumerate(class_names)
        ],
    )

    results = {
        **metrics,
        "class_names": class_names,
        "labels": true_labels,
        "predictions": predicted_labels,
        "confusion_matrix": matrix.tolist(),
    }

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        (output_path / "classification_report.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        (output_path / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        with (output_path / "predictions.csv").open(
            "w", encoding="utf-8", newline=""
        ) as csv_stream:
            writer = csv.writer(csv_stream)
            writer.writerow(
                ["sample_id", "ground_truth", "predicted", "confidence"]
            )

            for sample_id, y_true, y_pred, confidence in zip(
                sample_ids,
                true_labels,
                predicted_labels,
                confidences,
            ):
                writer.writerow(
                    [
                        sample_id,
                        class_names[y_true],
                        class_names[y_pred],
                        f"{confidence:.6f}",
                    ]
                )

        with (output_path / "confusion_matrix.csv").open(
            "w", encoding="utf-8", newline=""
        ) as csv_stream:
            writer = csv.writer(csv_stream)
            writer.writerow(["true/predicted", *class_names])

            for name, row in zip(class_names, matrix):
                writer.writerow([name, *row.tolist()])

        from src.evaluation.plots import plot_confusion_matrix

        plot_confusion_matrix(
            matrix,
            class_names,
            output_path / "confusion_matrix.png",
        )

        plot_confusion_matrix(
            matrix,
            class_names,
            output_path / "confusion_matrix_normalized.png",
            normalize=True,
        )

    return results


__all__ = ["compute_metrics", "evaluate_model"]
