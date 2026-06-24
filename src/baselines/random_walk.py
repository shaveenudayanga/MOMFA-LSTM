"""Baseline 8: Random Walk Persistence

Predicts tomorrow's close = today's close. No model, no training, no hyperparameters.
This is the sanity floor - every other method must beat this to be publishable.

Deterministic: no seeds, no stochasticity. Results are identical across runs.
For Wilcoxon test compatibility the single result is stored as 30 identical values.

Run: python -m baselines.random_walk
Results saved to: results/baselines/random_walk.json
"""

import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from preprocessor import load_ticker, FEATURE_COLS, TARGET_COL
from utils.metrics import compute_all

N_SPLITS    = 5
VAL_SIZE    = 126
N_RUNS      = 30   # repeated for Wilcoxon test compatibility (all identical)

TICKERS = [
    'JKH.N0000', 'COMB.N0000', 'DIAL.N0000', 'HNB.N0000',  'LOLC.N0000',
    'SAMP.N0000', 'NTB.N0000', 'HHL.N0000',  'DIST.N0000', 'HAYL.N0000',
]

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'results', 'baselines'
)
OUTPUT_FILE = os.path.join(RESULTS_DIR, 'random_walk.json')


def run_one_ticker(ticker: str) -> dict:
    """Compute persistence-forecast metrics for one ticker across all folds.

    Persistence rule: y_hat_t = y_{t-1}
    In the cleaned CSV, `target[i]` = next-day close price (raw LKR).
    So y_hat[k] = target[train_end + k - 1]  (previous row's target = today's close)
    and y_true[k] = target[train_end + k]     (actual next-day close)

    The "previous" for the very first val step comes from the last training row,
    so there is no boundary gap - the persistence chain is unbroken.
    """
    df = load_ticker(ticker)
    prices = df[TARGET_COL].values   # raw LKR, not scaled
    n = len(prices)

    min_train = n - N_SPLITS * VAL_SIZE

    fold_rmse, fold_mae, fold_mape, fold_da = [], [], [], []

    for i in range(N_SPLITS):
        train_end = min_train + i * VAL_SIZE
        val_end   = train_end + VAL_SIZE

        # y_true: actual next-day close for each val step
        y_true = prices[train_end : val_end]
        # y_hat:  previous day's next-day close = today's close (persistence)
        # train_end - 1 gives the last row of the training window
        y_hat  = prices[train_end - 1 : val_end - 1]

        m = compute_all(y_true, y_hat)
        fold_rmse.append(m['rmse'])
        fold_mae.append(m['mae'])
        fold_mape.append(m['mape'])
        fold_da.append(m['da'])

    rmse_mean  = float(np.mean(fold_rmse))
    mae_mean   = float(np.mean(fold_mae))
    mape_mean  = float(np.mean(fold_mape))
    da_mean    = float(np.mean(fold_da))

    return {
        'deterministic': True,
        'rmse_runs':  [rmse_mean] * N_RUNS,
        'mae_runs':   [mae_mean]  * N_RUNS,
        'mape_runs':  [mape_mean] * N_RUNS,
        'da_runs':    [da_mean]   * N_RUNS,
        'rmse_mean':  rmse_mean,
        'rmse_std':   0.0,
        'mae_mean':   mae_mean,
        'mape_mean':  mape_mean,
        'da_mean':    da_mean,
    }


def run_all(tickers=None):
    tickers = tickers or TICKERS
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = {}
    for ticker in tickers:
        results[ticker] = run_one_ticker(ticker)
        r = results[ticker]
        print(
            f'[Random Walk] {ticker:<20}  '
            f'RMSE {r["rmse_mean"]:.4f}  |  '
            f'MAE {r["mae_mean"]:.4f}  |  '
            f'MAPE {r["mape_mean"]:.2f}%  |  '
            f'DA {r["da_mean"]:.4f}'
        )

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\nResults saved to {OUTPUT_FILE}')
    return results


if __name__ == '__main__':
    run_all()
