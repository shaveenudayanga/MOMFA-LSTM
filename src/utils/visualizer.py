"""Thesis-quality visualizations for MOMFA-LSTM.

All functions save a PNG to results/figures/ and return the file path.
Call plt.show() afterwards if running interactively in a notebook.

Usage:
    from utils.visualizer import plot_fold_splits, plot_rmse_comparison, ...
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi':      150,
    'font.size':       11,
    'axes.titlesize':  12,
    'axes.labelsize':  11,
    'legend.fontsize': 10,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
})

FIGURES_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'results', 'figures'
)

TICKERS = [
    'JKH.N0000', 'COMB.N0000', 'DIAL.N0000', 'HNB.N0000',  'LOLC.N0000',
    'SAMP.N0000', 'NTB.N0000', 'HHL.N0000',  'DIST.N0000', 'HAYL.N0000',
]

METHOD_LABELS = {
    'random_walk':    'Random Walk',
    'vanilla_lstm':   'Vanilla LSTM',
    'grid_search':    'Grid Search',
    'pso_lstm':       'PSO-LSTM',
    'ga_lstm':        'GA-LSTM',
    'gwo_lstm':       'GWO-LSTM',
    'woa_lstm':       'WOA-LSTM',
    'sequential_fa':  'Sequential FA',
    'momfa':          'MOMFA (ours)',
}

FEATURE_COLS = [
    'RSI', 'MACD', 'MACD Signal', 'Stoch %K', 'Stoch %D', 'ROC', 'Williams %R',
    'EMA9', 'EMA21', 'EMA50', 'SMA20', 'ADX', 'CCI', 'Parabolic SAR',
    'BB Upper', 'BB Lower', 'BB Width', 'ATR', 'Std Dev',
    'OBV', 'MFI', 'VWAP', 'Vol SMA', 'Vol ROC', 'CMF',
]

FEATURE_GROUPS = {
    'Momentum (7)':   list(range(0,  7)),
    'Trend (7)':      list(range(7,  14)),
    'Volatility (5)': list(range(14, 19)),
    'Volume (6)':     list(range(19, 25)),
}

VAL_COLOURS = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db']


def _save(fig, subdir: str, filename: str) -> str:
    out_dir = os.path.join(FIGURES_DIR, subdir)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'[visualizer] Saved → {path}')
    return path


def _ticker_label(ticker: str) -> str:
    return ticker.replace('.N0000', '')


# ── F-01: Fold splits ─────────────────────────────────────────────────────────

def plot_fold_splits(df, ticker: str, n_splits: int = 5, val_size: int = 126) -> str:
    """F-01: Stock price time series with walk-forward validation windows.

    Args:
        df:     DataFrame from preprocessor.load_ticker() (has 'target' column).
        ticker: Ticker string e.g. 'JKH.N0000'.
    """
    prices = df['target'].values
    dates  = df.index
    n      = len(prices)
    min_train = n - n_splits * val_size

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, prices, color='#2c3e50', linewidth=0.8, label='Close price')

    for i in range(n_splits):
        train_end = min_train + i * val_size
        val_end   = train_end + val_size
        ax.axvspan(dates[train_end], dates[val_end - 1],
                   alpha=0.25, color=VAL_COLOURS[i],
                   label=f'Val fold {i+1}')

    ax.set_title(f'Walk-Forward Validation Splits - {_ticker_label(ticker)}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close price (LKR)')
    ax.legend(loc='upper left', ncol=3, framealpha=0.7)
    fig.tight_layout()

    return _save(fig, 'data', f'fold_splits_{_ticker_label(ticker)}.png')


# ── F-02: Indicator correlation heatmap ───────────────────────────────────────

def plot_indicator_correlation(df, ticker: str = 'JKH') -> str:
    """F-02: Pearson correlation matrix of all 25 technical indicators."""
    raw_cols = [c for c in df.columns if c != 'target']
    corr = df[raw_cols].corr()

    fig, ax = plt.subplots(figsize=(13, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        ax=ax,
        mask=mask,
        cmap='RdYlGn',
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.3,
        xticklabels=FEATURE_COLS,
        yticklabels=FEATURE_COLS,
        annot=False,
        cbar_kws={'label': 'Pearson r'},
    )
    ax.set_title(f'Technical Indicator Correlation Matrix - {ticker}', pad=14)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    fig.tight_layout()

    return _save(fig, 'data', 'indicator_correlation.png')


# ── F-03: RMSE bar comparison ─────────────────────────────────────────────────

def plot_rmse_comparison(all_results: dict, methods_order: list = None) -> str:
    """F-03: Grouped bar chart - mean RMSE per method, averaged across tickers.

    Args:
        all_results: {method_name: {ticker: {rmse_mean, rmse_std, ...}}}
        methods_order: list of method keys in display order.
    """
    if methods_order is None:
        methods_order = list(all_results.keys())

    labels, means, stds = [], [], []
    for method in methods_order:
        if method not in all_results:
            continue
        ticker_data = all_results[method]
        rmse_vals = [
            np.mean(ticker_data[t]['rmse_runs'])
            for t in TICKERS if t in ticker_data
        ]
        if not rmse_vals:
            continue
        labels.append(METHOD_LABELS.get(method, method))
        means.append(np.mean(rmse_vals))
        stds.append(np.std(rmse_vals))

    colours = ['#95a5a6'] * (len(labels) - 1) + ['#e74c3c']  # MOMFA in red

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, color=colours, width=0.6,
                  capsize=4, error_kw={'linewidth': 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right')
    ax.set_ylabel('Mean RMSE (LKR)')
    ax.set_title('RMSE Comparison - All Methods (averaged across 10 CSE tickers)')

    # Annotate bars with value
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    return _save(fig, 'baselines', 'rmse_bar_comparison.png')


# ── F-04: RMSE box plots ──────────────────────────────────────────────────────

def plot_rmse_boxplots(all_results: dict, methods_order: list = None) -> str:
    """F-04: Box plots of 30-run RMSE distributions per method."""
    if methods_order is None:
        methods_order = list(all_results.keys())

    plot_data, labels = [], []
    for method in methods_order:
        if method not in all_results:
            continue
        ticker_data = all_results[method]
        # Collect per-seed averages across tickers
        per_seed = []
        n_seeds = len(next(iter(ticker_data.values()))['rmse_runs'])
        for seed in range(n_seeds):
            seed_vals = [
                ticker_data[t]['rmse_runs'][seed]
                for t in TICKERS if t in ticker_data
                and seed < len(ticker_data[t]['rmse_runs'])
            ]
            if seed_vals:
                per_seed.append(np.mean(seed_vals))
        if per_seed:
            plot_data.append(per_seed)
            labels.append(METHOD_LABELS.get(method, method))

    fig, ax = plt.subplots(figsize=(11, 5))
    bp = ax.boxplot(plot_data, patch_artist=True, notch=False,
                    medianprops=dict(color='black', linewidth=2))

    colours = ['#bdc3c7'] * (len(labels) - 1) + ['#e74c3c']
    for patch, colour in zip(bp['boxes'], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.7)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=25, ha='right')
    ax.set_ylabel('Mean RMSE across tickers (LKR)')
    ax.set_title('30-Run RMSE Distribution - All Methods')
    fig.tight_layout()

    return _save(fig, 'baselines', 'rmse_boxplots.png')


# ── F-05: Wilcoxon p-value heatmap ───────────────────────────────────────────

def plot_wilcoxon_heatmap(p_values: dict) -> str:
    """F-05: Heatmap of Wilcoxon p-values (MOMFA vs each baseline per ticker).

    Args:
        p_values: {method_name: {ticker: p_value}}
    """
    from scipy.stats import wilcoxon as _wilcoxon

    methods = [m for m in p_values if m != 'momfa']
    tickers = [t for t in TICKERS if t in next(iter(p_values.values()))]

    matrix = np.array([
        [p_values[m].get(t, 1.0) for t in tickers]
        for m in methods
    ])

    fig, ax = plt.subplots(figsize=(12, len(methods) * 0.7 + 1.5))
    sns.heatmap(
        matrix, ax=ax,
        xticklabels=[_ticker_label(t) for t in tickers],
        yticklabels=[METHOD_LABELS.get(m, m) for m in methods],
        cmap='RdYlGn_r',
        vmin=0, vmax=0.1,
        annot=True, fmt='.3f',
        linewidths=0.5,
        cbar_kws={'label': 'p-value (MOMFA vs baseline)'},
    )
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_title('Wilcoxon Signed-Rank Test - p-values (green = significant at p < 0.05)')
    fig.tight_layout()

    return _save(fig, 'baselines', 'wilcoxon_heatmap.png')


# ── F-06: Pareto front ────────────────────────────────────────────────────────

def plot_pareto_front(archive_solutions: list, ticker: str) -> str:
    """F-06: Pareto front scatter - RMSE vs parameter count.

    Args:
        archive_solutions: list of dicts with keys rmse, n_params, mask.
        ticker: ticker string.
    """
    rmse     = [s['rmse']     for s in archive_solutions]
    n_params = [s['n_params'] for s in archive_solutions]
    n_feats  = [sum(s['mask']) for s in archive_solutions]

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(n_params, rmse, c=n_feats, cmap='viridis', s=60, zorder=3)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Selected features')

    # Annotate best RMSE and smallest model
    best_rmse_idx   = int(np.argmin(rmse))
    best_params_idx = int(np.argmin(n_params))
    for idx, label in [(best_rmse_idx, 'Best RMSE'), (best_params_idx, 'Smallest model')]:
        ax.annotate(label, xy=(n_params[idx], rmse[idx]),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=9, arrowprops=dict(arrowstyle='->', color='grey'))

    ax.set_xlabel('Trainable parameters (model complexity f₂)')
    ax.set_ylabel('Val RMSE - LKR (prediction error f₁)')
    ax.set_title(f'MOMFA Pareto Front - {_ticker_label(ticker)}')
    fig.tight_layout()

    return _save(fig, 'momfa', f'pareto_front_{_ticker_label(ticker)}.png')


# ── F-07: Hypervolume convergence ─────────────────────────────────────────────

def plot_hypervolume_convergence(hv_per_iter: np.ndarray, ticker: str) -> str:
    """F-07: Mean ± std hypervolume per iteration across 30 runs.

    Args:
        hv_per_iter: (n_runs, n_iter) array of hypervolume values.
        ticker: ticker string.
    """
    mean_hv = hv_per_iter.mean(axis=0)
    std_hv  = hv_per_iter.std(axis=0)
    iters   = np.arange(1, len(mean_hv) + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(iters, mean_hv, color='#2980b9', linewidth=2, label='Mean HV')
    ax.fill_between(iters, mean_hv - std_hv, mean_hv + std_hv,
                    alpha=0.25, color='#2980b9', label='±1 std')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Hypervolume')
    ax.set_title(f'Hypervolume Convergence - {_ticker_label(ticker)} (30 runs)')
    ax.legend()
    fig.tight_layout()

    return _save(fig, 'momfa', f'hypervolume_{_ticker_label(ticker)}.png')


# ── F-08: Feature selection heatmap ──────────────────────────────────────────

def plot_feature_selection_heatmap(selection_freq: np.ndarray) -> str:
    """F-08: Feature selection frequency across tickers.

    Args:
        selection_freq: (25, n_tickers) array - fraction of Pareto solutions
                        that selected each feature per ticker.
    """
    ticker_labels = [_ticker_label(t) for t in TICKERS[:selection_freq.shape[1]]]

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        selection_freq, ax=ax,
        xticklabels=ticker_labels,
        yticklabels=FEATURE_COLS,
        cmap='YlOrRd',
        vmin=0, vmax=1,
        annot=True, fmt='.2f',
        linewidths=0.4,
        cbar_kws={'label': 'Selection frequency'},
    )
    # Draw group separator lines
    for label, indices in FEATURE_GROUPS.items():
        ax.axhline(y=indices[-1] + 1, color='#2c3e50', linewidth=1.5)

    # Group labels on y-axis
    group_mid = {
        'Momentum': 3.5, 'Trend': 10.5, 'Volatility': 16.5, 'Volume': 22.0
    }
    for label, pos in group_mid.items():
        ax.text(-0.5, pos, label, ha='right', va='center',
                fontsize=9, fontweight='bold', color='#2c3e50')

    ax.set_title('Feature Selection Frequency - MOMFA Pareto-Optimal Solutions')
    ax.set_xlabel('Ticker')
    ax.set_ylabel('Technical Indicator')
    fig.tight_layout()

    return _save(fig, 'momfa', 'feature_selection_heatmap.png')


# ── F-09: Predictions vs actual ───────────────────────────────────────────────

def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                     ticker: str, fold: int = 4) -> str:
    """F-09: Best MOMFA prediction vs actual close price.

    Args:
        y_true: Actual close prices (inverse-transformed, LKR).
        y_pred: Predicted close prices (inverse-transformed, LKR).
        ticker: Ticker string.
        fold:   Which validation fold is shown (for the title).
    """
    fig, ax = plt.subplots(figsize=(11, 4))
    steps = np.arange(len(y_true))
    ax.plot(steps, y_true, color='#2c3e50', linewidth=1.5, label='Actual')
    ax.plot(steps, y_pred, color='#e74c3c', linewidth=1.2,
            linestyle='--', label='MOMFA prediction')
    ax.set_xlabel('Trading day (validation fold)')
    ax.set_ylabel('Close price (LKR)')
    ax.set_title(f'Predicted vs Actual - {_ticker_label(ticker)} (Fold {fold + 1})')
    ax.legend()
    fig.tight_layout()

    return _save(fig, 'predictions', f'predictions_{_ticker_label(ticker)}.png')
