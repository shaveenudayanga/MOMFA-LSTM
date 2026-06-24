import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Guard against zero actuals (rare in price data, but CSE stocks can be low)
    mask = y_true != 0
    return float(np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100)


def directional_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prev: np.ndarray,
) -> float:
    """Fraction of steps where predicted direction matches actual direction.

    DA = sign(y_hat_t - y_{t-1}) == sign(y_t - y_{t-1})

    Args:
        y_true: actual closes at time t         shape (n,)
        y_pred: predicted closes at time t      shape (n,)
        y_prev: actual closes at time t-1       shape (n,)

    All arrays must be inverse-transformed (real price units).
    """
    actual_dir    = np.sign(y_true - y_prev)
    predicted_dir = np.sign(y_pred - y_prev)
    return float(np.mean(actual_dir == predicted_dir))


def wilcoxon_test(scores_a: np.ndarray, scores_b: np.ndarray) -> tuple:
    """Wilcoxon signed-rank test: is method A significantly better than B?

    Used to compare MOMFA (a) against each baseline (b) on 30-run RMSE values.
    Lower RMSE = better, so we test if a < b.

    Args:
        scores_a: (30,) RMSE values for method A (e.g. MOMFA)
        scores_b: (30,) RMSE values for method B (e.g. Vanilla LSTM)

    Returns:
        (statistic, p_value) - p < 0.05 means A is significantly better than B.
    """
    from scipy.stats import wilcoxon
    stat, p = wilcoxon(scores_a, scores_b, alternative='less')
    return float(stat), float(p)


def compute_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute all four metrics at once.

    DA is computed on aligned pairs: y_true[1:] vs y_true[:-1] as y_prev.
    This drops the first prediction (no previous actual available).

    All inputs must be inverse-transformed (real price units).
    """
    return {
        'rmse': rmse(y_true, y_pred),
        'mae':  mae(y_true, y_pred),
        'mape': mape(y_true, y_pred),
        'da':   directional_accuracy(y_true[1:], y_pred[1:], y_true[:-1]),
    }
