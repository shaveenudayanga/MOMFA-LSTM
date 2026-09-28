"""Thesis figures for MOMFA-LSTM (spec 11).

Every function draws one figure, saves a PNG under results/figures/ and returns
its path. The functions take plain arrays and dicts; plots/make_figures.py reads
the saved result files and calls them, so every figure can be regenerated from
disk (spec 10).
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import seaborn as sns

from config import TICKERS, METHOD_LABELS

# ── Style ─────────────────────────────────────────────────────────────────────
# Ink and grid are neutral so the data carries the colour.

INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#8a8984'
GRID = '#e6e5e0'

plt.rcParams.update({
    'figure.dpi':        150,
    'savefig.dpi':       200,
    'font.size':         10,
    'axes.titlesize':    11.5,
    'axes.titleweight':  'bold',
    'axes.titlelocation': 'left',
    'axes.labelsize':    10,
    'axes.labelcolor':   INK_2,
    'axes.edgecolor':    MUTED,
    'axes.linewidth':    0.8,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'axes.axisbelow':    True,
    'grid.color':        GRID,
    'grid.linewidth':    0.6,
    'xtick.color':       INK_2,
    'ytick.color':       INK_2,
    'text.color':        INK,
    'legend.fontsize':   9,
    'legend.frameon':    False,
    'figure.facecolor':  'white',
    'savefig.facecolor': 'white',
})

# Fixed colour per method (validated categorical palette, slot order).
# Random walk is the reference, so it stays neutral grey.
METHOD_COLORS = {
    'momfa':         '#2a78d6',
    'vanilla_lstm':  '#eb6834',
    'grid_search':   '#1baf7a',
    'pso_lstm':      '#eda100',
    'ga_lstm':       '#e87ba4',
    'gwo_lstm':      '#008300',
    'woa_lstm':      '#4a3aa7',
    'sequential_fa': '#e34948',
    'random_walk':   MUTED,
}

# Sequential (magnitude) and diverging (better / worse than a midpoint) maps
SEQUENTIAL = LinearSegmentedColormap.from_list(
    'seq_blue', ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
DIVERGING = LinearSegmentedColormap.from_list(
    'div_blue_red', ['#184f95', '#6da7ec', '#f0efec', '#ef8a89', '#b3302f'])

INDICATOR_NAMES = [
    'RSI', 'MACD', 'MACD signal', 'Stoch %K', 'Stoch %D', 'ROC', 'Williams %R',
    'EMA9', 'EMA21', 'EMA50', 'SMA20', 'ADX', 'CCI', 'Parabolic SAR',
    'BB upper', 'BB lower', 'BB width', 'ATR', 'Std dev',
    'OBV', 'MFI', 'VWAP', 'Volume SMA', 'Volume ROC', 'CMF',
]
FEATURE_COLS = INDICATOR_NAMES   # used by the MOMFA feature-selection heatmap
FEATURE_GROUPS = {
    'Momentum (7)':   list(range(0,  7)),
    'Trend (7)':      list(range(7,  14)),
    'Volatility (5)': list(range(14, 19)),
    'Volume (6)':     list(range(19, 25)),
}

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'figures')


def _save(fig, subdir: str, filename: str) -> str:
    out_dir = os.path.join(FIGURES_DIR, subdir)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.normpath(os.path.join(out_dir, filename))
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path


def _ticker_label(ticker: str) -> str:
    return ticker.replace('.N0000', '')


def _label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def _date_axis(ax):
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _annotated_heatmap(ax, matrix, row_labels, col_labels, cmap, norm, fmt):
    """Heatmap with the value written in each cell, ink chosen for contrast."""
    image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect='auto')
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isnan(value):
                continue
            r, g, b, _ = cmap(norm(value))
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(j, i, fmt.format(value), ha='center', va='center', fontsize=9,
                    color='white' if luminance < 0.5 else INK)
    return image


# ── F1: Development folds and final test period ──────────────────────────────

def plot_fold_splits(df, ticker: str) -> str:
    """Close price with the 5 walk-forward validation windows and the test period (spec 11 F1)."""
    from preprocessor import split_dev_test, fold_boundaries

    dev_df, test_df = split_dev_test(df)
    dates = df.index
    fold_colors = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#1c5cab']

    fig, ax = plt.subplots(figsize=(11, 3.6))
    for i, (train_end, val_end) in enumerate(fold_boundaries(len(dev_df))):
        ax.axvspan(dates[train_end], dates[val_end - 1], color=fold_colors[i],
                   alpha=0.55, lw=0, label=f'Validation fold {i + 1}')
    ax.axvspan(test_df.index[0], test_df.index[-1], facecolor='none', edgecolor=MUTED,
               hatch='///', lw=0, label='Final test period (never seen by optimizers)')
    ax.plot(dates, df['target'].values, color=INK, lw=1.1)

    ax.set_title(f'{_ticker_label(ticker)}: development folds and final test period')
    ax.set_ylabel('Close price (LKR)')
    _date_axis(ax)
    ax.legend(loc='upper left', ncol=3)
    return _save(fig, 'data', f'fold_splits_{_ticker_label(ticker)}.png')


# ── F2: Indicator correlation ────────────────────────────────────────────────

def plot_indicator_correlation(df, ticker: str) -> str:
    """Pearson correlation of the 25 indicators; shows redundancy (spec 11 F2)."""
    from preprocessor import FEATURE_COLS as indicator_cols

    corr = df[indicator_cols].corr().values
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(10.5, 9))
    sns.heatmap(corr, ax=ax, mask=mask, cmap=DIVERGING, center=0, vmin=-1, vmax=1,
                linewidths=0.4, linecolor='white', square=True,
                xticklabels=INDICATOR_NAMES, yticklabels=INDICATOR_NAMES,
                cbar_kws={'label': 'Pearson correlation', 'shrink': 0.7})
    ax.set_title(f'{_ticker_label(ticker)}: correlation between the 25 technical indicators')
    ax.tick_params(length=0)
    plt.xticks(rotation=60, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    return _save(fig, 'data', f'indicator_correlation_{_ticker_label(ticker)}.png')


# ── F3: Test-period predictions ──────────────────────────────────────────────

def plot_test_predictions(ticker: str, dates, actual: np.ndarray,
                          predictions: dict, zoom_days: int = 30) -> str:
    """Actual vs predicted close on the test period, with a zoomed last-30-day panel (spec 11 F3).

    Args:
        predictions: {method: predicted closes}, one value per test date.
    """
    fig, (full, zoom) = plt.subplots(
        2, 1, figsize=(11, 6.4), gridspec_kw={'height_ratios': [1.6, 1]})

    for ax, start in ((full, 0), (zoom, len(actual) - zoom_days)):
        ax.plot(dates[start:], actual[start:], color=INK, lw=1.6, label='Actual close')
        for method, preds in predictions.items():
            style = {'ls': '--', 'lw': 1.1} if method == 'random_walk' else {'lw': 1.3}
            ax.plot(dates[start:], preds[start:], color=METHOD_COLORS[method],
                    label=_label(method), **style)
        ax.set_ylabel('Close price (LKR)')
        _date_axis(ax)

    full.set_title(f'{_ticker_label(ticker)}: next-day close on the final test period')
    zoom.set_title(f'Last {zoom_days} trading days (zoomed)', fontweight='normal', color=INK_2)
    full.legend(loc='upper left', ncol=len(predictions) + 1)
    fig.tight_layout()
    return _save(fig, 'predictions', f'test_predictions_{_ticker_label(ticker)}.png')


# ── F4: Error distribution ───────────────────────────────────────────────────

def plot_error_distribution(errors: dict) -> str:
    """Distribution of test-period percentage errors per method, pooled over tickers (spec 11 F4).

    Args:
        errors: {method: array of (predicted - actual) / actual * 100}
    """
    bins = np.arange(-6, 6.25, 0.25)
    fig, ax = plt.subplots(figsize=(9, 4))
    for method, err in errors.items():
        clipped = np.clip(err, bins[0], bins[-1])
        ax.hist(clipped, bins=bins, density=True, histtype='step', lw=1.6,
                color=METHOD_COLORS[method],
                label=f'{_label(method)}  (median |error| {np.median(np.abs(err)):.2f}%)')
    ax.axvline(0, color=INK_2, lw=0.8)
    ax.set_xlabel('Prediction error, % of actual close (values beyond ±6% shown at the edge)')
    ax.set_ylabel('Density')
    ax.set_title('Test-period prediction errors, all 10 tickers pooled')
    ax.legend(loc='upper right')
    return _save(fig, 'comparison', 'error_distribution.png')


# ── F5: RMSE / MAE relative to random walk (30 runs) ─────────────────────────

METRIC_NAMES = {'rmse': 'RMSE', 'mae': 'MAE'}


def plot_vs_random_walk_boxplots(ratios: dict, metric: str) -> str:
    """Test RMSE or MAE of every run divided by random walk's, per ticker (spec 11 F5, check S1).

    Dividing by random walk puts tickers with very different prices on one scale.
    Below 1.0 = better than random walk.

    Args:
        ratios: {method: {ticker: [one ratio per seed]}}
        metric: 'rmse' or 'mae'
    """
    name = METRIC_NAMES[metric]
    methods = list(ratios)
    tickers = [t for t in TICKERS if any(t in ratios[m] for m in methods)]
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(11, 4.4))
    for k, method in enumerate(methods):
        positions = [i + (k - (len(methods) - 1) / 2) * width for i in range(len(tickers))]
        data = [ratios[method].get(t, [np.nan]) for t in tickers]
        box = ax.boxplot(data, positions=positions, widths=width * 0.8, patch_artist=True,
                         showfliers=True, medianprops={'color': INK, 'lw': 1.2},
                         whiskerprops={'color': INK_2, 'lw': 0.8},
                         capprops={'color': INK_2, 'lw': 0.8},
                         flierprops={'marker': 'o', 'ms': 3, 'mec': INK_2, 'mfc': 'none'})
        for patch in box['boxes']:
            patch.set(facecolor=METHOD_COLORS[method], edgecolor=INK_2, lw=0.8, alpha=0.85)
        box['boxes'][0].set_label(f'{_label(method)} (30 runs)')

    ax.axhline(1.0, color=INK_2, ls='--', lw=1, label='Random walk = 1.0')
    ax.set_xticks(range(len(tickers)), [_ticker_label(t) for t in tickers])
    ax.set_ylabel(f'Test {name} ÷ random-walk {name}')
    ax.set_title(f'Test {name} relative to random walk (below 1.0 = beats random walk)')
    ax.legend(loc='upper left')
    return _save(fig, 'comparison', f'{metric}_vs_random_walk_boxplots.png')


def plot_vs_random_walk_heatmap(mean_ratios: dict, metric: str) -> str:
    """Mean test RMSE or MAE ÷ random walk's, tickers x methods; blue beats random walk, red loses.

    Args:
        mean_ratios: {method: {ticker: mean ratio over seeds}}
        metric:      'rmse' or 'mae'
    """
    name = METRIC_NAMES[metric]
    methods = list(mean_ratios)
    matrix = np.array([[mean_ratios[m].get(t, np.nan) for m in methods] for t in TICKERS])
    spread = max(0.25, np.nanmax(np.abs(matrix - 1)))
    norm = TwoSlopeNorm(vcenter=1.0, vmin=1 - spread, vmax=1 + spread)

    fig, ax = plt.subplots(figsize=(1.6 + 1.4 * len(methods), 5.6))
    image = _annotated_heatmap(ax, matrix, [_ticker_label(t) for t in TICKERS],
                               [_label(m) for m in methods], DIVERGING, norm, '{:.2f}')
    fig.colorbar(image, ax=ax, shrink=0.8, label=f'{name} ÷ random-walk {name}')
    ax.set_title(f'Test {name} vs random walk\n(< 1.00 blue = better, > 1.00 red = worse)')
    return _save(fig, 'comparison', f'{metric}_vs_random_walk_heatmap.png')


# ── All four metrics in one figure ───────────────────────────────────────────

def plot_metric_overview(values: dict, spreads: dict) -> str:
    """The four accuracy measures of spec 4.B side by side, per ticker and method.

    RMSE and MAE are shown divided by random walk (they are in LKR); MAPE and DA are
    already percentages. Error bars show ±1 std over the 30 runs.

    Args:
        values:  {metric: {method: {ticker: mean}}} for metric in rmse, mae, mape, da
        spreads: same layout, std over seeds (0 for random walk)
    """
    panels = [
        ('rmse', 'RMSE ÷ random-walk RMSE', 'lower is better', 1.0),
        ('mae',  'MAE ÷ random-walk MAE',   'lower is better', 1.0),
        ('mape', 'MAPE (%)',                'lower is better', None),
        ('da',   'Directional accuracy (%)', 'higher is better', 50.0),
    ]
    x = np.arange(len(TICKERS))
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.6), sharex=True)

    for ax, (metric, ylabel, direction, reference) in zip(axes.flat, panels):
        methods = list(values[metric])
        width = 0.8 / len(methods)
        for k, method in enumerate(methods):
            means = np.array([values[metric][method].get(t, np.nan) for t in TICKERS])
            stds = np.array([spreads[metric][method].get(t, 0.0) for t in TICKERS])
            ax.bar(x + (k - (len(methods) - 1) / 2) * width, means, width=width * 0.92,
                   yerr=None if np.allclose(stds, 0) else stds, capsize=2,
                   error_kw={'lw': 0.8, 'ecolor': INK_2},
                   color=METHOD_COLORS[method], edgecolor='white', lw=1, label=_label(method))
        if reference is not None:
            label = 'Random walk = 1.0' if reference == 1.0 else '50% (coin flip)'
            ax.axhline(reference, color=INK_2, ls='--', lw=1, label=label)
        if metric == 'da':
            ax.set_ylim(0, 100)
        ax.set_title(f'{ylabel}  ({direction})', fontsize=10.5)
        ax.set_xticks(x, [_ticker_label(t) for t in TICKERS], fontsize=8.5)
        ax.legend(loc='upper left', fontsize=8)

    fig.suptitle('Test-period accuracy: RMSE, MAE, MAPE and directional accuracy',
                 x=0.01, ha='left', fontweight='bold', fontsize=12)
    fig.tight_layout()
    return _save(fig, 'comparison', 'metric_overview.png')


# ── F6: MAPE heatmap ─────────────────────────────────────────────────────────

def plot_mape_heatmap(mape: dict) -> str:
    """Mean test MAPE (%), tickers x methods (spec 11 F6).

    Args:
        mape: {method: {ticker: mean MAPE}}
    """
    methods = list(mape)
    matrix = np.array([[mape[m].get(t, np.nan) for m in methods] for t in TICKERS])
    norm = plt.Normalize(vmin=0, vmax=np.nanmax(matrix))

    fig, ax = plt.subplots(figsize=(1.6 + 1.4 * len(methods), 5.6))
    image = _annotated_heatmap(ax, matrix, [_ticker_label(t) for t in TICKERS],
                               [_label(m) for m in methods], SEQUENTIAL, norm, '{:.2f}%')
    fig.colorbar(image, ax=ax, shrink=0.8, label='MAPE (%)')
    ax.set_title('Test-period MAPE (%) — lower is better')
    return _save(fig, 'comparison', 'mape_heatmap.png')


# ── F7: Directional accuracy ─────────────────────────────────────────────────

def plot_directional_accuracy(da: dict, majority_rate: dict, flat_share: dict) -> str:
    """Test directional accuracy per ticker and method, against 50% and the
    "always predict the majority direction" rate (spec 11 F7, check S3).

    Args:
        da:            {method: {ticker: mean DA as a fraction}}
        majority_rate: {ticker: DA of always predicting the more common direction}
        flat_share:    {ticker: share of test days with no price change}
    """
    methods = list(da)
    width = 0.8 / len(methods)
    x = np.arange(len(TICKERS))

    fig, ax = plt.subplots(figsize=(11, 4.4))
    for k, method in enumerate(methods):
        values = [100 * da[method].get(t, np.nan) for t in TICKERS]
        ax.bar(x + (k - (len(methods) - 1) / 2) * width, values, width=width * 0.92,
               color=METHOD_COLORS[method], edgecolor='white', lw=1, label=_label(method))

    ax.scatter(x, [100 * majority_rate[t] for t in TICKERS], marker='_', s=420, lw=2.2,
               color=INK, zorder=3, label='Always predict majority direction')
    ax.axhline(50, color=INK_2, ls='--', lw=1, label='50% (coin flip)')
    ax.set_xticks(x, [f'{_ticker_label(t)}\n{100 * flat_share[t]:.0f}% flat'
                      for t in TICKERS])
    ax.set_ylim(0, 100)
    ax.set_ylabel('Directional accuracy (%)')
    ax.set_title('Test-period directional accuracy '
                 '(flat days = no price change, always counted as a miss)')
    ax.legend(loc='upper left', ncol=len(methods) + 2)
    return _save(fig, 'comparison', 'directional_accuracy.png')


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


