from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from src.models.dataset import CLASS_NAMES, CLASS_TO_INDEX


def compute_metrics(y_true: list[int] | np.ndarray, y_pred: list[int] | np.ndarray, class_names: list[str] | None = None) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if class_names is None:
        class_names = list(CLASS_NAMES)
    labels = [CLASS_TO_INDEX.get(name, idx) for idx, name in enumerate(class_names)]
    report = classification_report(y_true, y_pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0)
    results = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(report.get("macro avg", {}).get("precision", 0.0)),
        "macro_recall": float(report.get("macro avg", {}).get("recall", 0.0)),
        "macro_f1": float(report.get("macro avg", {}).get("f1-score", 0.0)),
        "per_class": {},
    }
    for name in class_names:
        idx = CLASS_TO_INDEX.get(name, 0)
        class_stats = report.get(name, {})
        if not class_stats:
            class_stats = report.get("0", {})
        results["per_class"][name] = {
            "precision": float(class_stats.get("precision", 0.0)),
            "recall": float(class_stats.get("recall", 0.0)),
            "f1": float(class_stats.get("f1-score", 0.0)),
            "support": int(class_stats.get("support", 0)),
        }
    return results


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
    sample_ids: list[str] = []

    with torch.no_grad():
        for batch in dataset:
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                inputs, targets = batch
                inputs = inputs.to(device_obj)
                logits = model(inputs)
                probs = torch.softmax(logits, dim=1)
                prediction = logits.argmax(dim=1)
                true_labels.extend(targets.tolist())
                predicted_labels.extend(prediction.tolist())
                confidences.extend(probs.max(dim=1).values.cpu().tolist())
            else:
                raise TypeError("Dataset must yield (inputs, targets) tuples")

    metrics = compute_metrics(true_labels, predicted_labels, class_names=class_names)
    results = {**metrics, "class_names": class_names, "labels": true_labels, "predictions": predicted_labels}
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "classification_report.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (output_path / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        with (output_path / "predictions.csv").open("w", encoding="utf-8", newline="") as csv_stream:
            writer = csv.writer(csv_stream)
            writer.writerow(["sample_id", "ground_truth", "predicted", "confidence"])
            for index, (y_true, y_pred, conf) in enumerate(zip(true_labels, predicted_labels, confidences)):
                writer.writerow([index, class_names[y_true], class_names[y_pred], f"{conf:.6f}"])
    return results


__all__ = ["compute_metrics", "evaluate_model"]
