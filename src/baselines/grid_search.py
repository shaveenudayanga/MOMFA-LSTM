"""Baseline 2: Grid Search LSTM (spec 4.A.2, 9.9)

Tries every combination in a fixed hyperparameter grid, all 25 features,
no feature selection. The combination with the lowest mean validation RMSE
on the walk-forward folds is retrained once and scored on the test period.

Grid (spec 9.9 default), 2 x 3 x 2 x 1 x 3 x 1 = 36 configurations:
    n_layers  {1, 2}
    units     {32, 64, 128}
    lr        {0.001, 0.0001}
    dropout   {0.2}
    lookback  {10, 20, 30}
    optimizer {adam}

Run: python -m baselines.grid_search
Results saved to: results/baselines/grid_search.json
"""

import itertools
import json
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import TICKERS, N_SPLITS
from preprocessor import load_ticker, split_dev_test, get_folds, get_final_split
from evaluation import cross_validate, evaluate_on_test, summarize_runs, METRICS
from train import get_device
from utils.seed import set_seed
from utils.logger import print_method_summary

GRID_VALUES = {
    'n_layers':  [1, 2],
    'units':     [32, 64, 128],
    'lr':        [0.001, 0.0001],
    'dropout':   [0.2],
    'lookback':  [10, 20, 30],
    'optimizer': ['adam'],
}
GRID = [dict(zip(GRID_VALUES, combo)) for combo in itertools.product(*GRID_VALUES.values())]

N_RUNS = 30

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'results', 'baselines'
)
OUTPUT_FILE = os.path.join(RESULTS_DIR, 'grid_search.json')


def search_grid(folds: list, grid: list = GRID) -> tuple[dict, dict, list]:
    """Cross-validate every configuration and return the one with the lowest validation RMSE.

    Returns:
        best_hp, best_val (cross_validate output), all_rmse (one value per grid entry)
    """
    scores = [cross_validate(folds, hp) for hp in grid]
    all_rmse = [s['rmse'] for s in scores]
    best = int(np.argmin(all_rmse))
    return grid[best], scores[best], all_rmse


def run_one_ticker(ticker: str, seeds: list, grid: list = GRID) -> dict:
    """Run the full grid search once per seed for one ticker.

    Training is stochastic, so the chosen configuration can differ between seeds.
    """
    df = load_ticker(ticker)
    dev_df, _ = split_dev_test(df)
    folds = get_folds(dev_df)
    final_split = get_final_split(df)

    val_runs = {m: [] for m in METRICS}
    test_runs = {m: [] for m in METRICS}
    chosen_configs, grid_rmse = [], []

    for seed in seeds:
        set_seed(seed)
        best_hp, best_val, all_rmse = search_grid(folds, grid)
        test = evaluate_on_test(final_split, best_hp)

        for m in METRICS:
            val_runs[m].append(best_val[m])
            test_runs[m].append(test[f'test_{m}'])
        chosen_configs.append(best_hp)
        grid_rmse.append(all_rmse)

    return {
        'grid_size':      len(grid),
        'chosen_configs': chosen_configs,
        'grid_val_rmse':  grid_rmse,   # [seed][grid index], same order as GRID
        **summarize_runs(val_runs, test_runs),
        'test_stop_size': final_split['stop_size'],
    }


def run_all(tickers=None, n_runs=N_RUNS):
    tickers = tickers or TICKERS
    seeds = list(range(n_runs))

    device = get_device()
    device_label = 'GPU (CUDA)' if device.type == 'cuda' else 'CPU'
    print(f'[Grid Search] Using {device_label} | grid={len(GRID)}  runs={n_runs}  '
          f'folds={N_SPLITS}  tickers={len(tickers)}')

    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = {}
    for ticker in tickers:
        print(f'\n[Grid Search] {ticker}', flush=True)
        results[ticker] = run_one_ticker(ticker, seeds)
        r = results[ticker]
        print(f'  VAL  RMSE {r["rmse_mean"]:.4f} ± {r["rmse_std"]:.4f}  |  '
              f'TEST RMSE {r["test_rmse_mean"]:.4f} ± {r["test_rmse_std"]:.4f}', flush=True)

        # Save after each ticker so a crash doesn't lose all results
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(results, f, indent=2)

    print(f'\nResults saved to {OUTPUT_FILE}')
    print_method_summary('grid_search', results)
    print('Figures and tables: python plots/make_figures.py')
    return results


if __name__ == '__main__':
    run_all()
