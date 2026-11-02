
from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.dataset import CLASS_NAMES


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer_state: dict[str, Any],
    epoch: int,
    metric: float,
    config: dict[str, Any],
    class_to_index: dict[str, int],
    seed: int,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer_state,
        "epoch": epoch,
        "best_val_metric": metric,
        "model_config": config,
        "class_to_index": class_to_index,
        "seed": seed,
    }
    torch.save(checkpoint, path)

def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(inputs)
            loss = criterion(logits, targets)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("DataLoader contains no samples.")

    return (
        total_loss / total_samples,
        total_correct / total_samples,
    )


def train_model(
    model: nn.Module,
    train_dataset: TensorDataset | DataLoader,
    val_dataset: TensorDataset | DataLoader | None = None,
    *,
    model_name: str,
    output_dir: str | Path,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    seed: int = 42,
    patience: int = 5,
    class_to_index: dict[str, int] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:

    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if patience < 1:
        raise ValueError("patience must be at least 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    set_seed(seed)

    device_obj = torch.device(device)
    model.to(device_obj)

    if class_to_index is None:
        class_to_index = {
            label: i for i, label in enumerate(CLASS_NAMES)
        }

    if isinstance(train_dataset, DataLoader):
        train_loader = train_dataset
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )

    if len(train_loader.dataset) == 0:
        raise ValueError("Training dataset is empty.")

    if val_dataset is None:
        val_loader = None
    elif isinstance(val_dataset, DataLoader):
        val_loader = val_dataset
    else:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

    if val_loader is not None and len(val_loader.dataset) == 0:
        raise ValueError("Validation dataset is empty.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_state = copy.deepcopy(model.state_dict())
    best_optimizer_state = copy.deepcopy(optimizer.state_dict())
    best_metric = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    last_epoch = 0
    last_state = copy.deepcopy(model.state_dict())
    last_optimizer_state = copy.deepcopy(optimizer.state_dict())

    for epoch in range(1, epochs + 1):
        last_epoch = epoch

        train_loss, train_accuracy = _run_epoch(
            model,
            train_loader,
            criterion,
            device_obj,
            optimizer,
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)

        if val_loader is not None:
            val_loss, val_accuracy = _run_epoch(
                model,
                val_loader,
                criterion,
                device_obj,
            )

            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_accuracy)

            if val_loss < best_metric:
                best_metric = val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                best_optimizer_state = copy.deepcopy(optimizer.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                break

        last_state = copy.deepcopy(model.state_dict())
        last_optimizer_state = copy.deepcopy(optimizer.state_dict())

    config_payload = {
        "model_name": model_name,
        "epochs": last_epoch,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "device": str(device_obj),
        "seed": seed,
        "class_to_index": class_to_index,
        "patience": patience,
    }

    if config:
        config_payload.update(config)

    (output_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    (output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    model.load_state_dict(best_state)
    _save_checkpoint(
        output_dir / "best_model.pt",
        model,
        best_optimizer_state,
        best_epoch,
        best_metric,
        config_payload,
        class_to_index,
        seed,
    )

    model.load_state_dict(last_state)
    _save_checkpoint(
        output_dir / "last_model.pt",
        model,
        last_optimizer_state,
        last_epoch,
        best_metric,
        config_payload,
        class_to_index,
        seed,
    )

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_metric,
        "last_epoch": last_epoch,
        "output_dir": str(output_dir),
        "class_to_index": class_to_index,
        "model_name": model_name,
    }


__all__ = ["set_seed", "train_model"]
