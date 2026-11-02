from __future__ import annotations

import torch
from torch import nn


class SequenceLSTMClassifier(nn.Module):
    """Sequence classifier using an LSTM over temporal pose features."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_dim: int = 3,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        self.output_dim = int(output_dim)
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout if self.num_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
        )
        factor = 2 if self.bidirectional else 1
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size * factor, self.output_dim),
        )

    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if sequence.dim() == 2:
            sequence = sequence.unsqueeze(1)
        if sequence.dim() != 3:
            raise ValueError("sequence must have shape (batch, seq_len, features)")
        batch_size, seq_len, _ = sequence.shape
        if lengths is not None:
            lengths = lengths.clamp(min=1, max=seq_len)
            packed = nn.utils.rnn.pack_padded_sequence(
                sequence, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            logits, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=seq_len)
            last_states = logits[torch.arange(batch_size, device=sequence.device), lengths - 1]
        else:
            output, _ = self.lstm(sequence)
            last_states = output[:, -1, :]
        return self.head(last_states)


__all__ = ["SequenceLSTMClassifier"]
