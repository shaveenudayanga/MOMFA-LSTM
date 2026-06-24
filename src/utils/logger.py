"""Results logging: JSON → Markdown tables, CSV, and LaTeX.

Every baseline and MOMFA run saves a JSON file first (source of truth).
Call these functions to produce human-readable and thesis-ready outputs.

Usage:
    from utils.logger import save_method_table, save_comparison_table
    save_method_table('random_walk', results_dict)
    save_comparison_table(all_results_dict)
"""

import csv
import json
import os
import numpy as np

TABLES_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'results', 'tables'
)

METHODS_ORDER = [
    'random_walk',
    'vanilla_lstm',
    'grid_search',
    'pso_lstm',
    'ga_lstm',
    'gwo_lstm',
    'woa_lstm',
    'sequential_fa',
    'momfa',
]

METHOD_LABELS = {
    'random_walk':    'Random Walk',
    'vanilla_lstm':   'Vanilla LSTM',
    'grid_search':    'Grid Search LSTM',
    'pso_lstm':       'PSO-LSTM',
    'ga_lstm':        'GA-LSTM',
    'gwo_lstm':       'GWO-LSTM',
    'woa_lstm':       'WOA-LSTM',
    'sequential_fa':  'Sequential FA-LSTM',
    'momfa':          'MOMFA (ours)',
}

TICKERS = [
    'JKH.N0000', 'COMB.N0000', 'DIAL.N0000', 'HNB.N0000',  'LOLC.N0000',
    'SAMP.N0000', 'NTB.N0000', 'HHL.N0000',  'DIST.N0000', 'HAYL.N0000',
]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_json(method_name: str) -> dict:
    path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'results', 'baselines',
        f'{method_name}.json'
    )
    if method_name == 'momfa':
        path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'results', 'momfa', 'momfa.json'
        )
    with open(path) as f:
        return json.load(f)


def _ticker_label(ticker: str) -> str:
    return ticker.replace('.N0000', '')


def _fmt(mean: float, std: float | None = None) -> str:
    if std is not None and std > 0:
        return f'{mean:.4f} ± {std:.4f}'
    return f'{mean:.4f}'


def _extract_metrics(ticker_data: dict) -> dict:
    """Extract mean/std metrics from a per-ticker result dict."""
    rmse_runs = ticker_data.get('rmse_runs', [ticker_data.get('rmse_mean', 0)])
    return {
        'rmse_mean': float(np.mean(rmse_runs)),
        'rmse_std':  float(np.std(rmse_runs)),
        'mae_mean':  ticker_data.get('mae_mean',  0.0),
        'mape_mean': ticker_data.get('mape_mean', 0.0),
        'da_mean':   ticker_data.get('da_mean',   0.0),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def save_method_table(method_name: str, results: dict) -> str:
    """Save a per-ticker summary table for one method.

    Args:
        method_name: e.g. 'random_walk', 'vanilla_lstm'
        results:     dict keyed by ticker string, matching the JSON format

    Returns:
        Path to the saved markdown file.
    """
    os.makedirs(TABLES_DIR, exist_ok=True)

    label = METHOD_LABELS.get(method_name, method_name)
    lines = [
        f'# Results: {label}',
        '',
        '| Ticker | RMSE (mean ± std) | MAE | MAPE (%) | DA |',
        '|--------|-------------------|-----|----------|----|',
    ]

    rmse_vals, mae_vals, mape_vals, da_vals = [], [], [], []

    for ticker in TICKERS:
        if ticker not in results:
            continue
        m = _extract_metrics(results[ticker])
        rmse_vals.append(m['rmse_mean'])
        mae_vals.append(m['mae_mean'])
        mape_vals.append(m['mape_mean'])
        da_vals.append(m['da_mean'])
        lines.append(
            f'| {_ticker_label(ticker):<6} '
            f'| {_fmt(m["rmse_mean"], m["rmse_std"]):<19} '
            f'| {m["mae_mean"]:.4f} '
            f'| {m["mape_mean"]:.2f} '
            f'| {m["da_mean"]:.4f} |'
        )

    if rmse_vals:
        lines.append(
            f'| **Avg** '
            f'| **{np.mean(rmse_vals):.4f}** '
            f'| **{np.mean(mae_vals):.4f}** '
            f'| **{np.mean(mape_vals):.2f}** '
            f'| **{np.mean(da_vals):.4f}** |'
        )

    out_path = os.path.join(TABLES_DIR, f'{method_name}.md')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f'[logger] Table saved → {out_path}')
    return out_path


def save_comparison_table(all_results: dict) -> dict:
    """Build the master comparison table from all available method results.

    Args:
        all_results: dict keyed by method_name → per-ticker results dict.
                     Only methods present in the dict are included.

    Returns:
        Dict of paths: {'md': ..., 'csv': ..., 'tex': ...}
    """
    os.makedirs(TABLES_DIR, exist_ok=True)

    # Aggregate: for each method, average across all tickers
    rows = []
    for method in METHODS_ORDER:
        if method not in all_results:
            continue
        ticker_data = all_results[method]
        rmse_all, mae_all, mape_all, da_all = [], [], [], []
        for ticker in TICKERS:
            if ticker not in ticker_data:
                continue
            m = _extract_metrics(ticker_data[ticker])
            rmse_all.append(m['rmse_mean'])
            mae_all.append(m['mae_mean'])
            mape_all.append(m['mape_mean'])
            da_all.append(m['da_mean'])
        if not rmse_all:
            continue
        rows.append({
            'method':    method,
            'label':     METHOD_LABELS.get(method, method),
            'rmse_mean': float(np.mean(rmse_all)),
            'rmse_std':  float(np.std(rmse_all)),
            'mae_mean':  float(np.mean(mae_all)),
            'mape_mean': float(np.mean(mape_all)),
            'da_mean':   float(np.mean(da_all)),
        })

    if not rows:
        print('[logger] No results to compare yet.')
        return {}

    # Find best value per column (lower is better for RMSE/MAE/MAPE, higher for DA)
    best_rmse  = min(r['rmse_mean'] for r in rows)
    best_mae   = min(r['mae_mean']  for r in rows)
    best_mape  = min(r['mape_mean'] for r in rows)
    best_da    = max(r['da_mean']   for r in rows)

    paths = {}

    # ── Markdown ──────────────────────────────────────────────────────────────
    md_lines = [
        '# Full Method Comparison',
        '',
        'Values averaged across all tickers and 30 independent runs.',
        '**Bold** = best per column.',
        '',
        '| Method | RMSE (mean ± std) | MAE | MAPE (%) | DA |',
        '|--------|-------------------|-----|----------|----|',
    ]
    for r in rows:
        def bold(val, best, fmt):
            s = fmt.format(val)
            return f'**{s}**' if abs(val - best) < 1e-9 else s
        md_lines.append(
            f'| {r["label"]} '
            f'| {_fmt(r["rmse_mean"], r["rmse_std"])} '
            f'| {bold(r["mae_mean"],  best_mae,  "{:.4f}")} '
            f'| {bold(r["mape_mean"], best_mape, "{:.2f}")} '
            f'| {bold(r["da_mean"],   best_da,   "{:.4f}")} |'
        )
    md_path = os.path.join(TABLES_DIR, 'comparison_all_methods.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines) + '\n')
    paths['md'] = md_path

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = os.path.join(TABLES_DIR, 'comparison_all_methods.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'Method', 'RMSE_mean', 'RMSE_std', 'MAE_mean', 'MAPE_mean', 'DA_mean'
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow({
                'Method':    r['label'],
                'RMSE_mean': f'{r["rmse_mean"]:.6f}',
                'RMSE_std':  f'{r["rmse_std"]:.6f}',
                'MAE_mean':  f'{r["mae_mean"]:.6f}',
                'MAPE_mean': f'{r["mape_mean"]:.6f}',
                'DA_mean':   f'{r["da_mean"]:.6f}',
            })
    paths['csv'] = csv_path

    # ── LaTeX ─────────────────────────────────────────────────────────────────
    tex_lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\caption{Comparison of all methods averaged across 10 CSE tickers and 30 independent runs.}',
        r'\label{tab:comparison}',
        r'\begin{tabular}{lcccc}',
        r'\toprule',
        r'Method & RMSE & MAE & MAPE (\%) & DA \\',
        r'\midrule',
    ]
    for r in rows:
        rmse_str = f'{r["rmse_mean"]:.4f} $\\pm$ {r["rmse_std"]:.4f}'
        label = r['label'].replace('(ours)', r'\textbf{(ours)}')
        if r['method'] == 'momfa':
            label = r'\textbf{' + label + r'}'
        tex_lines.append(
            f'{label} & {rmse_str} & {r["mae_mean"]:.4f} '
            f'& {r["mape_mean"]:.2f} & {r["da_mean"]:.4f} \\\\'
        )
    tex_lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]
    tex_path = os.path.join(TABLES_DIR, 'comparison_all_methods.tex')
    with open(tex_path, 'w') as f:
        f.write('\n'.join(tex_lines) + '\n')
    paths['tex'] = tex_path

    for k, p in paths.items():
        print(f'[logger] {k.upper()} saved → {p}')

    return paths


def print_method_summary(method_name: str, results: dict) -> None:
    """Print a quick console summary table for one method."""
    label = METHOD_LABELS.get(method_name, method_name)
    print(f'\n{"="*65}')
    print(f'  {label}')
    print(f'{"="*65}')
    print(f'  {"Ticker":<8}  {"RMSE":>12}  {"MAE":>8}  {"MAPE%":>7}  {"DA":>7}')
    print(f'  {"-"*55}')
    rmse_all, mae_all, mape_all, da_all = [], [], [], []
    for ticker in TICKERS:
        if ticker not in results:
            continue
        m = _extract_metrics(results[ticker])
        rmse_all.append(m['rmse_mean'])
        mae_all.append(m['mae_mean'])
        mape_all.append(m['mape_mean'])
        da_all.append(m['da_mean'])
        print(
            f'  {_ticker_label(ticker):<8}  '
            f'{_fmt(m["rmse_mean"], m["rmse_std"]):>12}  '
            f'{m["mae_mean"]:>8.4f}  '
            f'{m["mape_mean"]:>7.2f}  '
            f'{m["da_mean"]:>7.4f}'
        )
    if rmse_all:
        print(f'  {"-"*55}')
        print(
            f'  {"Average":<8}  '
            f'{np.mean(rmse_all):>12.4f}  '
            f'{np.mean(mae_all):>8.4f}  '
            f'{np.mean(mape_all):>7.2f}  '
            f'{np.mean(da_all):>7.4f}'
        )
    print(f'{"="*65}\n')
