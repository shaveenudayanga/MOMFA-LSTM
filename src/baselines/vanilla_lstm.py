"""Baseline 1: Vanilla LSTM

Fixed hyperparameters, all 25 features, no feature selection, no HPO.
Sets the lower-bound baseline that every other method must beat.

Fixed config (chosen as sensible literature defaults):
    n_layers  = 2
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

from preprocessor import load_ticker, get_folds, make_sequences
from model import build_model, count_parameters
from train import train_fold, get_device
from utils.seed import set_seed
from utils.logger import save_method_table, print_method_summary
from utils.visualizer import plot_rmse_comparison, plot_rmse_boxplots, plot_predictions

# ── Fixed hyperparameters ─────────────────────────────────────────────────────

CONFIG = {
    'n_layers':  2,
    'units':     64,
    'lr':        0.001,
    'dropout':   0.2,
    'lookback':  20,
    'optimizer': 'adam',
}

N_FEATURES  = 25   # all features, no mask
N_SPLITS    = 5
VAL_SIZE    = 126
N_RUNS      = 30
MAX_EPOCHS  = 50
BATCH_SIZE  = 32
PATIENCE    = 10

TICKERS = [
    'JKH.N0000', 'COMB.N0000', 'DIAL.N0000', 'HNB.N0000',  'LOLC.N0000',
    'SAMP.N0000', 'NTB.N0000', 'HHL.N0000',  'DIST.N0000', 'HAYL.N0000',
]

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'results', 'baselines'
)
OUTPUT_FILE = os.path.join(RESULTS_DIR, 'vanilla_lstm.json')


def run_one_ticker(ticker: str, seeds: list) -> dict:
    """Run 30 seeds × 5 folds for one ticker. Returns per-seed averaged metrics."""
    df    = load_ticker(ticker)
    folds = get_folds(df, n_splits=N_SPLITS, val_size=VAL_SIZE)

    # Pre-build sequences for each fold (lookback is fixed, so do this once)
    fold_seqs = []
    for fold in folds:
        X_tr, y_tr = make_sequences(fold['X_train'], fold['y_train'], CONFIG['lookback'])
        X_va, y_va = make_sequences(fold['X_val'],   fold['y_val'],   CONFIG['lookback'])
        fold_seqs.append((X_tr, y_tr, X_va, y_va, fold['target_scaler']))

    per_seed_rmse = []
    per_seed_mae  = []
    per_seed_mape = []
    per_seed_da   = []

    # Track best-seed predictions on the last fold (for predictions plot)
    best_seed_rmse = float('inf')
    best_val_preds = None
    best_val_true  = None

    for seed in seeds:
        set_seed(seed)

        fold_rmse, fold_mae, fold_mape, fold_da = [], [], [], []
        seed_preds, seed_true = None, None

        for (X_tr, y_tr, X_va, y_va, scaler) in fold_seqs:
            model = build_model(
                n_features=N_FEATURES,
                n_layers=CONFIG['n_layers'],
                units=CONFIG['units'],
                dropout=CONFIG['dropout'],
            )
            result = train_fold(
                model, X_tr, y_tr, X_va, y_va,
                target_scaler=scaler,
                lr=CONFIG['lr'],
                optimizer_name=CONFIG['optimizer'],
                max_epochs=MAX_EPOCHS,
                batch_size=BATCH_SIZE,
                patience=PATIENCE,
            )
            fold_rmse.append(result['val_rmse'])
            fold_mae.append(result['val_mae'])
            fold_mape.append(result['val_mape'])
            fold_da.append(result['val_da'])
            # Keep last fold's predictions for this seed
            seed_preds = result['val_preds']
            seed_true  = result['val_true']

        seed_mean_rmse = float(np.mean(fold_rmse))
        per_seed_rmse.append(seed_mean_rmse)
        per_seed_mae.append(float(np.mean(fold_mae)))
        per_seed_mape.append(float(np.mean(fold_mape)))
        per_seed_da.append(float(np.mean(fold_da)))

        # Save predictions from the best-performing seed
        if seed_mean_rmse < best_seed_rmse:
            best_seed_rmse = seed_mean_rmse
            best_val_preds = seed_preds
            best_val_true  = seed_true

    return {
        'config':    CONFIG,
        'n_params':  count_parameters(
            build_model(N_FEATURES, CONFIG['n_layers'], CONFIG['units'], CONFIG['dropout'])
        ),
        'rmse_runs':      per_seed_rmse,
        'mae_runs':       per_seed_mae,
        'mape_runs':      per_seed_mape,
        'da_runs':        per_seed_da,
        'rmse_mean':      float(np.mean(per_seed_rmse)),
        'rmse_std':       float(np.std(per_seed_rmse)),
        'mae_mean':       float(np.mean(per_seed_mae)),
        'mape_mean':      float(np.mean(per_seed_mape)),
        'da_mean':        float(np.mean(per_seed_da)),
        'best_val_preds': best_val_preds.tolist() if best_val_preds is not None else [],
        'best_val_true':  best_val_true.tolist()  if best_val_true  is not None else [],
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

        # Save after each ticker so a crash doesn't lose all results
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(results, f, indent=2)

    print(f'\nResults saved to {OUTPUT_FILE}')

    # ── Evaluation outputs ────────────────────────────────────────────────────
    print_method_summary('vanilla_lstm', results)
    save_method_table('vanilla_lstm', results)

    # Load random walk for partial comparison (already run)
    rw_path = os.path.join(RESULTS_DIR, 'random_walk.json')
    all_results = {'vanilla_lstm': results}
    if os.path.exists(rw_path):
        with open(rw_path) as f:
            all_results['random_walk'] = json.load(f)

    plot_rmse_comparison(all_results, methods_order=['random_walk', 'vanilla_lstm'])
    plot_rmse_boxplots(all_results,   methods_order=['random_walk', 'vanilla_lstm'])

    # Predictions plot for each ticker (best seed, last fold)
    for ticker in tickers:
        r = results.get(ticker, {})
        preds = r.get('best_val_preds', [])
        true  = r.get('best_val_true',  [])
        if preds and true:
            plot_predictions(
                np.array(true), np.array(preds),
                ticker=ticker, fold=N_SPLITS - 1
            )

    return results


if __name__ == '__main__':
    run_all()
