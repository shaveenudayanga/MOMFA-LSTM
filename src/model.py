import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_layers: int,
        units: int,
        dropout: float,
    ):
        super().__init__()
        # PyTorch raises a UserWarning if dropout > 0 with num_layers == 1
        # because there are no inter-layer connections to drop out.
        lstm_dropout = dropout if n_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=units,
            num_layers=n_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(units, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # last timestep only
        return self.fc(out).squeeze(-1)


def build_model(
    n_features: int,
    n_layers: int,
    units: int,
    dropout: float,
) -> LSTMModel:
    return LSTMModel(n_features, n_layers, units, dropout)


def count_parameters(model: nn.Module) -> int:
    """Total trainable parameter count - used as f2 (complexity objective)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
