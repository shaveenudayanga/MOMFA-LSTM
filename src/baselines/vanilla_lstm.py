"""Baseline 1: Vanilla LSTM

Fixed hyperparameters, all 25 features, no feature selection, no HPO.
Sets the lower-bound baseline that every other method must beat.

Fixed config (spec 9.9 defaults):
    n_layers  = 1
    units     = 64
    lr        = 0.001
    dropout   = 0.2
    lookback  = 20  (one trading month)
    optimizer = adam

Run: python -m baselines.vanilla_lstm
Results saved to: results/baselines/vanilla_lstm.json
"""

import json
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import TICKERS, N_SPLITS
from preprocessor import load_ticker, split_dev_test, get_folds, get_final_split
from evaluation import cross_validate, evaluate_on_test, summarize_runs, METRICS
from model import build_model, count_parameters
from train import get_device
from utils.seed import set_seed
from utils.logger import print_method_summary

# ── Fixed hyperparameters ─────────────────────────────────────────────────────

CONFIG = {
    'n_layers':  1,
    'units':     64,
    'lr':        0.001,
    'dropout':   0.2,
    'lookback':  20,
    'optimizer': 'adam',
}

N_FEATURES  = 25   # all features, no mask
N_RUNS      = 30

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'results', 'baselines'
)
OUTPUT_FILE = os.path.join(RESULTS_DIR, 'vanilla_lstm.json')


def run_one_ticker(ticker: str, seeds: list) -> dict:
    """Run every seed for one ticker.

    Per seed: 5 walk-forward folds on the development period (validation metrics),
    then one final model scored on the untouched test period (spec 9.1, 9.1.1).
    """
    df = load_ticker(ticker)
    dev_df, _ = split_dev_test(df)
    folds = get_folds(dev_df)
    final_split = get_final_split(df)

    val_runs = {m: [] for m in METRICS}
    test_runs = {m: [] for m in METRICS}

    # Seed with the best VALIDATION RMSE supplies the plotted predictions.
    # Choosing it by test RMSE would be cherry-picking on the test period.
    best_seed_rmse = float('inf')
    best_val, best_test = None, None

    for seed in seeds:
        set_seed(seed)
        val = cross_validate(folds, CONFIG)
        test = evaluate_on_test(final_split, CONFIG)

        for m in METRICS:
            val_runs[m].append(val[m])
            test_runs[m].append(test[f'test_{m}'])

        if val['rmse'] < best_seed_rmse:
            best_seed_rmse = val['rmse']
            best_val, best_test = val, test

    return {
        'config':   CONFIG,
        'n_params': count_parameters(
            build_model(N_FEATURES, CONFIG['n_layers'], CONFIG['units'], CONFIG['dropout'])
        ),
        **summarize_runs(val_runs, test_runs),
        'test_stop_size':  final_split['stop_size'],
        'best_val_preds':  best_val['last_fold_preds'].tolist(),
        'best_val_true':   best_val['last_fold_true'].tolist(),
        'best_test_preds': best_test['test_preds'].tolist(),
        'best_test_true':  best_test['test_true'].tolist(),
    }


def run_all(tickers=None, n_runs=N_RUNS):
    tickers = tickers or TICKERS
    seeds   = list(range(n_runs))   # seeds 0..29

    device = get_device()
    device_label = 'GPU (CUDA)' if device.type == 'cuda' else 'CPU'
    print(f'[Vanilla LSTM] Using {device_label} | runs={n_runs}  folds={N_SPLITS}  tickers={len(tickers)}')

    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = {}
    for ticker in tickers:
        print(f'\n[Vanilla LSTM] {ticker}')
        results[ticker] = run_one_ticker(ticker, seeds)
        r = results[ticker]
        print(
            f'  RMSE  {r["rmse_mean"]:.4f} ± {r["rmse_std"]:.4f}  |  '
            f'MAE {r["mae_mean"]:.4f}  |  '
            f'MAPE {r["mape_mean"]:.2f}%  |  '
            f'DA {r["da_mean"]:.4f}'
        )
        print(f'  TEST RMSE {r["test_rmse_mean"]:.4f} ± {r["test_rmse_std"]:.4f}')

        # Save after each ticker so a crash doesn't lose all results
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(results, f, indent=2)

    print(f'\nResults saved to {OUTPUT_FILE}')

    print_method_summary('vanilla_lstm', results)
    print('Figures and tables: python plots/make_figures.py')
    return results


if __name__ == '__main__':
    run_all()
