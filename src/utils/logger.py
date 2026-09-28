"""Result tables: load saved results and write Markdown / CSV / LaTeX tables (spec 10, 11 T1).

Every number comes from the saved JSON files in results/, never from memory.

Tables written to results/tables/:
    method_summary        one row per method, test period (the thesis T1 table)
    test_by_ticker        every method x ticker, test period
    validation_by_ticker  every method x ticker, walk-forward validation

RMSE and MAE are in LKR, so they are never averaged across tickers (LOLC trades
near 300 LKR, JKH near 20). Cross-ticker summaries use scale-free numbers:
RMSE relative to random walk, MAPE and directional accuracy.
"""

import csv
import glob
import json
import os
import numpy as np

from config import TICKERS, METHODS, METHOD_LABELS

RESULTS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'results'))
TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')


def load_results() -> dict:
    """{method: {ticker: result dict}} for every method with a saved result file."""
    paths = glob.glob(os.path.join(RESULTS_DIR, 'baselines', '*.json'))
    paths += glob.glob(os.path.join(RESULTS_DIR, 'momfa', '*.json'))
    found = {}
    for path in paths:
        method = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            found[method] = json.load(f)
    return {m: found[m] for m in METHODS if m in found}


def _ticker_label(ticker: str) -> str:
    return ticker.replace('.N0000', '')


def ratio_runs(results: dict, method: str, ticker: str, metric: str = 'rmse') -> list:
    """Per-seed test RMSE or MAE divided by random walk's (below 1 = beats random walk).

    RMSE and MAE are in LKR, so this ratio is how they are compared across tickers.
    """
    rw = results['random_walk'][ticker][f'test_{metric}_mean']
    return [r / rw for r in results[method][ticker][f'test_{metric}_runs']]


def _by_ticker_rows(results: dict, prefix: str) -> list[dict]:
    rows = []
    for method, per_ticker in results.items():
        for ticker in TICKERS:
            if ticker not in per_ticker:
                continue
            r = per_ticker[ticker]
            row = {
                'method':    METHOD_LABELS.get(method, method),
                'ticker':    _ticker_label(ticker),
                'rmse_mean': r[f'{prefix}rmse_mean'],
                'rmse_std':  r[f'{prefix}rmse_std'],
                'mae':       r[f'{prefix}mae_mean'],
                'mape':      r[f'{prefix}mape_mean'],
                'da':        r[f'{prefix}da_mean'],
            }
            if prefix == 'test_' and 'random_walk' in results:
                row['rmse_vs_rw'] = float(np.mean(ratio_runs(results, method, ticker, 'rmse')))
                row['mae_vs_rw'] = float(np.mean(ratio_runs(results, method, ticker, 'mae')))
            rows.append(row)
    return rows


def _summary_rows(results: dict) -> list[dict]:
    rows = []
    for method, per_ticker in results.items():
        tickers = [t for t in TICKERS if t in per_ticker]
        rmse_ratios = [np.mean(ratio_runs(results, method, t, 'rmse')) for t in tickers]
        mae_ratios = [np.mean(ratio_runs(results, method, t, 'mae')) for t in tickers]
        rows.append({
            'method':         METHOD_LABELS.get(method, method),
            'tickers':        len(tickers),
            'beats_rw':       sum(r < 1 for r in rmse_ratios),
            'rmse_vs_rw_med': float(np.median(rmse_ratios)),
            'mae_vs_rw_med':  float(np.median(mae_ratios)),
            'mape':           float(np.mean([per_ticker[t]['test_mape_mean'] for t in tickers])),
            'da':             float(np.mean([per_ticker[t]['test_da_mean'] for t in tickers])),
        })
    return rows


def _write_md(path: str, title: str, note: str, header: list, lines: list) -> None:
    with open(path, 'w') as f:
        f.write(f'# {title}\n\n{note}\n\n')
        f.write('| ' + ' | '.join(header) + ' |\n')
        f.write('|' + '|'.join('---' for _ in header) + '|\n')
        for line in lines:
            f.write('| ' + ' | '.join(line) + ' |\n')


def _write_csv(path: str, rows: list[dict]) -> None:
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_tables(results: dict) -> list[str]:
    """Write every result table; returns the paths written."""
    os.makedirs(TABLES_DIR, exist_ok=True)
    written = []

    # ── Method summary (thesis T1) ──
    summary = _summary_rows(results)
    base = os.path.join(TABLES_DIR, 'method_summary')
    _write_md(
        base + '.md', 'Method summary - final test period',
        'Mean over 30 seeds per ticker, then over tickers. RMSE and MAE are in LKR, so they are '
        'shown divided by the random-walk value (below 1.00 beats random walk), median across '
        'tickers. "Beats random walk" counts tickers where the mean RMSE is below random walk.',
        ['Method', 'Beats random walk (RMSE)', 'RMSE vs random walk', 'MAE vs random walk',
         'MAPE (%)', 'DA (%)'],
        [[r['method'], f"{r['beats_rw']}/{r['tickers']} tickers", f"{r['rmse_vs_rw_med']:.3f}",
          f"{r['mae_vs_rw_med']:.3f}", f"{r['mape']:.2f}", f"{100 * r['da']:.1f}"]
         for r in summary],
    )
    _write_csv(base + '.csv', summary)
    with open(base + '.tex', 'w') as f:
        f.write('\\begin{table}[h]\n\\centering\n')
        f.write('\\caption{Final test period, 10 CSE tickers, 30 independent runs per method.}\n')
        f.write('\\label{tab:method_summary}\n\\begin{tabular}{lccccc}\n\\toprule\n')
        f.write('Method & Beats RW & RMSE / RW & MAE / RW & MAPE (\\%) & DA (\\%) \\\\\n'
                '\\midrule\n')
        for r in summary:
            f.write(f"{r['method']} & {r['beats_rw']}/{r['tickers']} & {r['rmse_vs_rw_med']:.3f} "
                    f"& {r['mae_vs_rw_med']:.3f} & {r['mape']:.2f} & {100 * r['da']:.1f} \\\\\n")
        f.write('\\bottomrule\n\\end{tabular}\n\\end{table}\n')
    written += [base + ext for ext in ('.md', '.csv', '.tex')]

    # ── Per ticker: test and validation ──
    for prefix, name, title in (('test_', 'test_by_ticker', 'Final test period by ticker'),
                                ('', 'validation_by_ticker', 'Walk-forward validation by ticker')):
        rows = _by_ticker_rows(results, prefix)
        header = ['Method', 'Ticker', 'RMSE (LKR, mean ± std)', 'MAE (LKR)', 'MAPE (%)', 'DA (%)']
        if prefix == 'test_':
            header += ['RMSE vs random walk', 'MAE vs random walk']
        lines = []
        for r in rows:
            line = [r['method'], r['ticker'], f"{r['rmse_mean']:.4f} ± {r['rmse_std']:.4f}",
                    f"{r['mae']:.4f}", f"{r['mape']:.2f}", f"{100 * r['da']:.1f}"]
            if 'rmse_vs_rw' in r:
                line += [f"{r['rmse_vs_rw']:.3f}", f"{r['mae_vs_rw']:.3f}"]
            lines.append(line)
        base = os.path.join(TABLES_DIR, name)
        _write_md(base + '.md', title, 'Mean over 30 seeds (random walk is deterministic).',
                  header, lines)
        _write_csv(base + '.csv', rows)
        written += [base + '.md', base + '.csv']

    return written


def print_method_summary(method_name: str, results: dict) -> None:
    """Console summary for one method: validation and test RMSE per ticker."""
    label = METHOD_LABELS.get(method_name, method_name)
    print(f'\n{"=" * 62}\n  {label}\n{"=" * 62}')
    print(f'  {"Ticker":<8} {"Val RMSE":>12} {"Test RMSE":>12} {"Test MAPE%":>11} {"Test DA":>8}')
    for ticker in TICKERS:
        if ticker not in results:
            continue
        r = results[ticker]
        print(f'  {_ticker_label(ticker):<8} {r["rmse_mean"]:>12.4f} {r["test_rmse_mean"]:>12.4f} '
              f'{r["test_mape_mean"]:>11.2f} {r["test_da_mean"]:>8.3f}')
    print(f'{"=" * 62}\n')
