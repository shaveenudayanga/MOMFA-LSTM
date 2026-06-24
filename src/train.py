import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils.metrics import compute_all


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _build_optimizer(
    model: nn.Module,
    optimizer_name: str,
    lr: float,
) -> torch.optim.Optimizer:
    name = optimizer_name.lower()
    if name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif name == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), lr=lr)
    elif name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer '{optimizer_name}'. Choose adam/rmsprop/sgd.")


def train_fold(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    target_scaler,
    lr: float,
    optimizer_name: str,
    max_epochs: int = 50,
    batch_size: int = 32,
    patience: int = 10,
    device: torch.device = None,
) -> dict:
    """Train one LSTM on one walk-forward fold and return validation metrics.

    Args:
        model:          LSTMModel instance (already built for this fold's n_features).
        X_train:        (n_seq, lookback, n_features) float64, scaled.
        y_train:        (n_seq,) float64, scaled.
        X_val:          (n_seq, lookback, n_features) float64, scaled.
        y_val:          (n_seq,) float64, scaled.
        target_scaler:  MinMaxScaler fitted on training targets - used to inverse-transform.
        lr:             Learning rate.
        optimizer_name: 'adam' | 'rmsprop' | 'sgd'.
        max_epochs:     Hard cap (50 per spec).
        batch_size:     Mini-batch size (32 per spec).
        patience:       Early stopping patience on val MSE (10 per spec).
        device:         Torch device. Defaults to get_device().

    Returns dict:
        val_rmse:   float  - primary fitness objective f1 (inverse-transformed prices)
        val_mae:    float
        val_mape:   float
        val_da:     float
        val_preds:  np.ndarray  - inverse-transformed predictions
        val_true:   np.ndarray  - inverse-transformed ground truth
        best_epoch: int
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    optimizer = _build_optimizer(model, optimizer_name, lr)
    criterion = nn.MSELoss()

    # ── DataLoader for training ───────────────────────────────────────────────
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=batch_size,
        shuffle=True,
    )

    X_va = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_va = torch.tensor(y_val, dtype=torch.float32).to(device)

    # ── Training loop with early stopping ────────────────────────────────────
    best_val_loss = float('inf')
    best_weights  = copy.deepcopy(model.state_dict())
    best_epoch    = 0
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_va), y_va).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights  = copy.deepcopy(model.state_dict())
            best_epoch    = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    # ── Restore best weights and predict ─────────────────────────────────────
    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_va).cpu().numpy()

    val_true_scaled = y_val  # already numpy

    # Inverse-transform to real price units before computing metrics
    val_preds = target_scaler.inverse_transform(
        preds_scaled.reshape(-1, 1)
    ).ravel()
    val_true = target_scaler.inverse_transform(
        val_true_scaled.reshape(-1, 1)
    ).ravel()

    metrics = compute_all(val_true, val_preds)

    return {
        'val_rmse':   metrics['rmse'],
        'val_mae':    metrics['mae'],
        'val_mape':   metrics['mape'],
        'val_da':     metrics['da'],
        'val_preds':  val_preds,
        'val_true':   val_true,
        'best_epoch': best_epoch,
    }
