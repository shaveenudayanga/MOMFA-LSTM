import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

FEATURE_COLS = [
    # Momentum (7)
    'rsi', 'macd', 'macds', 'stoch_k', 'stoch_d', 'roc', 'willr',
    # Trend (7)
    'ema9', 'ema21', 'ema50', 'sma20', 'adx', 'cci', 'psar',
    # Volatility (5)
    'bb_upper', 'bb_lower', 'bb_width', 'atr', 'stddev',
    # Volume (6)
    'obv', 'mfi', 'vwap', 'vol_sma', 'vol_roc', 'cmf',
]
TARGET_COL = 'target'
N_FEATURES = 25

assert len(FEATURE_COLS) == N_FEATURES


def load_ticker(ticker: str, data_dir: str = "data/preprocessed") -> pd.DataFrame:
    """Load a cleaned ticker CSV. Returns DataFrame indexed by datetime."""
    path = os.path.join(data_dir, f"CLEANED_{ticker}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cleaned data not found at '{path}'. "
            "Run src/data_loader.py first."
        )
    df = pd.read_csv(path, parse_dates=['datetime'], index_col='datetime')
    return df[FEATURE_COLS + [TARGET_COL]]


def make_sequences(X: np.ndarray, y: np.ndarray, lookback: int):
    """Sliding window -> LSTM (input, target) pairs.

    Args:
        X:        (n_rows, n_features) scaled feature array
        y:        (n_rows,) scaled target array
        lookback: number of past days fed to the LSTM at each step

    Returns:
        X_seq: (n_rows - lookback, lookback, n_features)
        y_seq: (n_rows - lookback,)  -- target at step lookback+i for window i
    """
    n = len(X) - lookback
    if n <= 0:
        raise ValueError(
            f"lookback={lookback} leaves no sequences (only {len(X)} rows available)."
        )
    X_seq = np.stack([X[i: i + lookback] for i in range(n)])
    y_seq = y[lookback:]
    return X_seq, y_seq


def get_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    val_size: int = 126,
) -> list:
    """Walk-forward expanding-window folds with per-fold Min-Max scaling.

    Fold layout (expanding train, fixed-size val):
        fold 0: train = rows[0 : min_train],          val = rows[min_train       : min_train +   val_size]
        fold 1: train = rows[0 : min_train+val_size], val = rows[min_train+val_size : min_train + 2*val_size]
        ...

    Scaler contract (no leakage):
        - feature_scaler and target_scaler are fit on the training rows of EACH fold only.
        - Validation rows are transformed with that fold's scaler, never fit on it.

    Args:
        df:       Output of load_ticker() -- 25 feature cols + target col.
        n_splits: Number of folds.
        val_size: Validation rows per fold (default 126 ~= 6 trading months).

    Returns:
        List of dicts, one per fold:
            X_train:        (n_train, 25)  scaled to [0, 1]
            y_train:        (n_train,)     scaled to [0, 1]
            X_val:          (n_val, 25)    scaled to [0, 1]
            y_val:          (n_val,)       scaled to [0, 1]
            feature_scaler: MinMaxScaler   fitted on X_train rows
            target_scaler:  MinMaxScaler   fitted on y_train rows

    Usage with variable lookback (from LSTM hyperparameter):
        for fold in get_folds(df):
            X_seq, y_seq = make_sequences(fold['X_train'], fold['y_train'], lookback)
            Xv_seq, yv_seq = make_sequences(fold['X_val'], fold['y_val'], lookback)
            # inverse_transform predictions: fold['target_scaler'].inverse_transform(preds)
    """
    X_all = df[FEATURE_COLS].values.astype(np.float64)
    y_all = df[TARGET_COL].values.astype(np.float64)
    n = len(X_all)

    min_train = n - n_splits * val_size
    if min_train <= 0:
        raise ValueError(
            f"Dataset has {n} rows but n_splits={n_splits} x val_size={val_size} "
            f"= {n_splits * val_size} leaves {min_train} rows for the first training window. "
            "Reduce n_splits or val_size."
        )

    folds = []
    for i in range(n_splits):
        train_end = min_train + i * val_size
        val_end   = train_end + val_size

        X_tr = X_all[:train_end]
        y_tr = y_all[:train_end]
        X_va = X_all[train_end:val_end]
        y_va = y_all[train_end:val_end]

        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        feature_scaler.fit(X_tr)

        target_scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaler.fit(y_tr.reshape(-1, 1))

        folds.append({
            'X_train':        feature_scaler.transform(X_tr),
            'y_train':        target_scaler.transform(y_tr.reshape(-1, 1)).ravel(),
            'X_val':          feature_scaler.transform(X_va),
            'y_val':          target_scaler.transform(y_va.reshape(-1, 1)).ravel(),
            'feature_scaler': feature_scaler,
            'target_scaler':  target_scaler,
        })

    return folds
