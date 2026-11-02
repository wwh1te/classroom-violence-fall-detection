from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class MLPClassifier(nn.Module):
    """Simple multilayer perceptron for pose-based action classification."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] | int = (128, 64),
        output_dim: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims,)
        hidden_dims = tuple(int(value) for value in hidden_dims)
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 3:
            features = features.mean(dim=1)
        return self.network(features)


__all__ = ["MLPClassifier"]
