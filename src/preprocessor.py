import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import TEST_FRACTION, N_SPLITS, VAL_SIZE

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
CLOSE_COL = 'close'     # today's close, Close[t]
TARGET_COL = 'target'   # next-day close, Close[t+1]
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
    return df[FEATURE_COLS + [CLOSE_COL, TARGET_COL]]


def split_dev_test(
    df: pd.DataFrame,
    test_fraction: float = TEST_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split one ticker chronologically into development and final test periods (spec 9.1).

    Development rows feed the walk-forward folds used by every optimizer.
    Test rows are kept aside for the one-time final evaluation only.
    """
    n_test = int(round(len(df) * test_fraction))
    if not 0 < n_test < len(df):
        raise ValueError(f"test_fraction={test_fraction} gives {n_test} test rows out of {len(df)}.")
    return df.iloc[:-n_test], df.iloc[-n_test:]


def fold_boundaries(
    n_rows: int,
    n_splits: int = N_SPLITS,
    val_size: int = VAL_SIZE,
) -> list[tuple[int, int]]:
    """Row indices (train_end, val_end) of each expanding-window fold.

    Fold i trains on rows [0, train_end) and validates on rows [train_end, val_end).
    The last fold's validation window ends at the last development row.
    """
    first_train_end = n_rows - n_splits * val_size
    if first_train_end <= 0:
        raise ValueError(
            f"{n_rows} rows cannot hold {n_splits} folds x {val_size} validation rows. "
            "Reduce n_splits or val_size."
        )
    return [
        (first_train_end + i * val_size, first_train_end + (i + 1) * val_size)
        for i in range(n_splits)
    ]


def make_sequences(X: np.ndarray, y: np.ndarray, lookback: int):
    """Sliding windows of `lookback` rows, each paired with the target of its last row.

    y[t] already holds Close[t+1] (data_loader shifts the target), so the window
    ending at row t predicts the next day's close using rows <= t only (spec 9.2).

    Returns:
        X_seq: (n_rows - lookback + 1, lookback, n_features)
        y_seq: (n_rows - lookback + 1,)
    """
    n_windows = len(X) - lookback + 1
    if n_windows <= 0:
        raise ValueError(
            f"lookback={lookback} leaves no sequences (only {len(X)} rows available)."
        )
    X_seq = np.stack([X[i: i + lookback] for i in range(n_windows)])
    y_seq = y[lookback - 1:]
    return X_seq, y_seq


def make_fold_sequences(fold: dict, lookback: int, feature_idx: list | None = None):
    """Train and validation sequences for one fold.

    Validation windows take their first lookback-1 input rows from the end of the
    training period, so every candidate is scored on exactly the same validation
    dates whatever its lookback (spec 9.3). Those rows are past data, not leakage.

    Args:
        fold:        One element of get_folds().
        lookback:    Window length in days.
        feature_idx: Columns to keep (the feature mask). None keeps all 25.

    Returns:
        X_train_seq, y_train_seq, X_val_seq, y_val_seq
        (y_val_seq always equals fold['y_val'])
    """
    X_train, X_val = _select(fold['X_train'], feature_idx), _select(fold['X_val'], feature_idx)

    X_tr, y_tr = make_sequences(X_train, fold['y_train'], lookback)
    X_va, y_va = _windows_with_history(X_train, fold['y_train'], X_val, fold['y_val'], lookback)
    return X_tr, y_tr, X_va, y_va


def make_test_sequences(split: dict, lookback: int, feature_idx: list | None = None):
    """Final-test-period sequences for the output of get_final_split().

    Every test date is scored; input history comes from the end of the
    development period (spec 9.3 applied to the test period).

    Returns:
        X_test_seq, y_test_seq  (y_test_seq always equals split['y_test'])
    """
    X_dev = np.concatenate([split['X_train'], split['X_val']])
    y_dev = np.concatenate([split['y_train'], split['y_val']])
    return _windows_with_history(
        _select(X_dev, feature_idx), y_dev,
        _select(split['X_test'], feature_idx), split['y_test'],
        lookback,
    )


def _select(X: np.ndarray, feature_idx: list | None) -> np.ndarray:
    return X if feature_idx is None else X[:, feature_idx]


def _windows_with_history(X_history, y_history, X_eval, y_eval, lookback: int):
    """One window per X_eval row; the first lookback-1 input rows come from the history."""
    start = len(X_history) - (lookback - 1)
    X = np.concatenate([X_history[start:], X_eval])
    y = np.concatenate([y_history[start:], y_eval])
    return make_sequences(X, y, lookback)


def next_day_return(df: pd.DataFrame) -> np.ndarray:
    """Return from today's close to tomorrow's close, as a column: Close[t+1] / Close[t] - 1.

    This is what the LSTM learns to predict (DECISIONS.md D-009). Prices are rebuilt
    with returns_to_prices(), so every metric is still on the next-day close price.
    """
    return (df[TARGET_COL].values / df[CLOSE_COL].values - 1).reshape(-1, 1)


def returns_to_prices(scaled_returns: np.ndarray, target_scaler, close_today: np.ndarray) -> np.ndarray:
    """Turn scaled predicted returns back into next-day close prices (LKR)."""
    returns = target_scaler.inverse_transform(scaled_returns.reshape(-1, 1)).ravel()
    return close_today * (1 + returns)


def _scale(train_df: pd.DataFrame, **other_parts: pd.DataFrame) -> dict:
    """Min-Max scale every part with scalers fit on train_df only (spec 9.2).

    Returns, for 'train' and each other part <name>:
        X_<name>      scaled features
        y_<name>      scaled next-day return (the LSTM target)
        close_<name>  today's close in LKR (to rebuild the predicted price)
        price_<name>  actual next-day close in LKR (to score predictions)
    plus feature_scaler and target_scaler (the latter scales returns).
    """
    feature_scaler = MinMaxScaler(feature_range=(0, 1)).fit(train_df[FEATURE_COLS].values)
    target_scaler = MinMaxScaler(feature_range=(0, 1)).fit(next_day_return(train_df))

    scaled = {'feature_scaler': feature_scaler, 'target_scaler': target_scaler}
    for name, part in {'train': train_df, **other_parts}.items():
        scaled[f'X_{name}'] = feature_scaler.transform(part[FEATURE_COLS].values)
        scaled[f'y_{name}'] = target_scaler.transform(next_day_return(part)).ravel()
        scaled[f'close_{name}'] = part[CLOSE_COL].values
        scaled[f'price_{name}'] = part[TARGET_COL].values
    return scaled


def get_folds(
    dev_df: pd.DataFrame,
    n_splits: int = N_SPLITS,
    val_size: int = VAL_SIZE,
) -> list:
    """Walk-forward expanding-window folds with per-fold Min-Max scaling (spec 3.B.4, 9.1).

    Pass the DEVELOPMENT period only (first output of split_dev_test), so no
    fold ever touches the final test period.

    Fold layout (expanding train, fixed-size val): see fold_boundaries().

    Scaler contract (no leakage, spec 9.2):
        - feature_scaler and target_scaler are fit on the training rows of EACH fold only.
        - Validation rows are transformed with that fold's scaler, never fit on it.

    Args:
        dev_df:   Development period of load_ticker() -- 25 feature cols + target col.
        n_splits: Number of folds.
        val_size: Validation rows per fold.

    Returns:
        List of dicts, one per fold:
            X_train:        (n_train, 25)  scaled to [0, 1]
            y_train:        (n_train,)     scaled next-day return
            X_val:          (n_val, 25)    scaled to [0, 1]
            y_val:          (n_val,)       scaled next-day return
            close_train / close_val:       today's close (LKR)
            price_train / price_val:       actual next-day close (LKR)
            feature_scaler: MinMaxScaler   fitted on training features
            target_scaler:  MinMaxScaler   fitted on training returns

    Usage:
        dev_df, _ = split_dev_test(load_ticker(ticker))
        for fold in get_folds(dev_df):
            X_tr, y_tr, X_va, y_va = make_fold_sequences(fold, lookback, feature_idx)
            # prices: returns_to_prices(preds, fold['target_scaler'], fold['close_val'])
    """
    return [
        _scale(dev_df.iloc[:train_end], val=dev_df.iloc[train_end:val_end])
        for train_end, val_end in fold_boundaries(len(dev_df), n_splits, val_size)
    ]


def get_final_split(df: pd.DataFrame, stop_size: int = VAL_SIZE) -> dict:
    """Data for the final retrained model of any method (spec 9.1, 9.1.1).

    The development period is cut into training rows and an early-stopping
    slice (its last `stop_size` rows). The model trains on the training rows,
    stops early on the slice, and is scored once on the final test period.
    Scalers are fit on the training rows only.

    The slice is stored as X_val / y_val so make_fold_sequences() works unchanged.
    If the development period is shorter than 3 * stop_size, the slice shrinks
    to a third of it; the size actually used is returned as 'stop_size'.

    Args:
        df: Full output of load_ticker().

    Returns:
        Dict with X_/y_ train, val (early-stopping slice) and test arrays,
        feature_scaler, target_scaler, stop_size and test_dates.
    """
    dev_df, test_df = split_dev_test(df)
    stop_size = min(stop_size, len(dev_df) // 3)

    split = _scale(
        dev_df.iloc[:-stop_size],
        val=dev_df.iloc[-stop_size:],
        test=test_df,
    )
    split['stop_size'] = stop_size
    split['test_dates'] = test_df.index
    return split
