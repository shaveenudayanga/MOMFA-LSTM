"""Build every figure and table from the saved result files (spec 10, 11).

Reads results/baselines/*.json (and results/momfa/*.json once MOMFA has run)
plus the cleaned data files, and writes:
    results/figures/data/          F1 fold splits (every ticker), F2 indicator correlation
    results/figures/predictions/   F3 actual vs predicted on the test period (every ticker)
    results/figures/comparison/    F4 errors, F5 RMSE and MAE vs random walk, F6 MAPE,
                                   F7 DA, and all four metrics in one overview
    results/tables/                method summary (T1), per-ticker test and validation tables

Nothing is taken from memory, so the whole set can be regenerated at any time.

Run from the repo root:
    venv/bin/python plots/make_figures.py
"""

import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))

from config import TICKERS
from preprocessor import load_ticker, split_dev_test, CLOSE_COL, TARGET_COL
from utils.logger import load_results, ratio_runs, write_tables
from utils import visualizer as viz

DATA_DIR = os.path.join(REPO_ROOT, 'data', 'preprocessed')


def test_period(ticker: str):
    """Dates, actual next-day close and today's close for the final test period."""
    _, test_df = split_dev_test(load_ticker(ticker, DATA_DIR))
    return test_df.index, test_df[TARGET_COL].values, test_df[CLOSE_COL].values


def data_figures() -> list[str]:
    paths = []
    for ticker in TICKERS:
        paths.append(viz.plot_fold_splits(load_ticker(ticker, DATA_DIR), ticker))
    dev_df, _ = split_dev_test(load_ticker('JKH.N0000', DATA_DIR))
    paths.append(viz.plot_indicator_correlation(dev_df, 'JKH.N0000'))
    return paths


def prediction_figures(results: dict) -> tuple[list[str], dict]:
    """F3 per ticker. Also returns pooled % errors per method for F4.

    Learned methods show the seed with the best VALIDATION RMSE (chosen before
    looking at the test period). Random walk's prediction is today's close.
    """
    paths, errors = [], {}
    learned = [m for m in results if 'best_test_preds' in next(iter(results[m].values()))]

    for ticker in TICKERS:
        dates, actual, close_today = test_period(ticker)
        predictions = {}
        for method in learned:
            if ticker in results[method]:
                saved_true = np.array(results[method][ticker]['best_test_true'])
                assert np.allclose(saved_true, actual), f'{method} {ticker}: test dates differ'
                predictions[method] = np.array(results[method][ticker]['best_test_preds'])
        if 'random_walk' in results:
            predictions['random_walk'] = close_today

        for method, preds in predictions.items():
            errors.setdefault(method, []).append((preds - actual) / actual * 100)
        paths.append(viz.plot_test_predictions(ticker, dates, actual, predictions))

    return paths, {m: np.concatenate(e) for m, e in errors.items()}


def direction_stats() -> tuple[dict, dict]:
    """Per ticker: DA of always predicting the more common direction, and share of flat days."""
    majority, flat = {}, {}
    for ticker in TICKERS:
        _, actual, close_today = test_period(ticker)
        change = actual - close_today
        majority[ticker] = max(np.mean(change > 0), np.mean(change < 0))
        flat[ticker] = np.mean(change == 0)
    return majority, flat


def comparison_figures(results: dict, errors: dict) -> list[str]:
    """Figures for all four accuracy measures (spec 4.B): RMSE, MAE, MAPE, DA."""
    learned = [m for m in results if m != 'random_walk']
    per_ticker = lambda key: {m: {t: results[m][t][key] for t in TICKERS if t in results[m]}
                              for m in results}
    paths = [viz.plot_error_distribution(errors)]

    # RMSE and MAE are in LKR, so they are compared as a ratio to random walk
    ratio_means, ratio_stds = {}, {}
    for metric in ('rmse', 'mae'):
        ratios = {m: {t: ratio_runs(results, m, t, metric) for t in TICKERS if t in results[m]}
                  for m in learned}
        ratio_means[metric] = {m: {t: float(np.mean(r)) for t, r in per_t.items()}
                               for m, per_t in ratios.items()}
        ratio_stds[metric] = {m: {t: float(np.std(r)) for t, r in per_t.items()}
                              for m, per_t in ratios.items()}
        paths.append(viz.plot_vs_random_walk_boxplots(ratios, metric))
        paths.append(viz.plot_vs_random_walk_heatmap(ratio_means[metric], metric))

    paths.append(viz.plot_mape_heatmap(per_ticker('test_mape_mean')))
    majority, flat = direction_stats()
    paths.append(viz.plot_directional_accuracy(per_ticker('test_da_mean'), majority, flat))

    std_of = lambda key: {m: {t: float(np.std(results[m][t][key])) for t in TICKERS
                              if t in results[m]} for m in results}
    percent = lambda d: {m: {t: 100 * v for t, v in per_t.items()} for m, per_t in d.items()}
    values = {**ratio_means, 'mape': per_ticker('test_mape_mean'),
              'da': percent(per_ticker('test_da_mean'))}
    spreads = {**ratio_stds, 'mape': std_of('test_mape_runs'),
               'da': percent(std_of('test_da_runs'))}
    paths.append(viz.plot_metric_overview(values, spreads))
    return paths


def main():
    results = load_results()
    print('Methods with saved results:', ', '.join(results) or 'none')

    written = data_figures()
    prediction_paths, errors = prediction_figures(results)
    written += prediction_paths
    written += comparison_figures(results, errors)
    written += write_tables(results)

    print(f'\nWrote {len(written)} files:')
    for path in written:
        print('  ' + os.path.relpath(path, REPO_ROOT))


if __name__ == '__main__':
    main()
