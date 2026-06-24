# RESULTS_PLAN.md
# Output Strategy - Figures, Tables, and Human-Readable Results

> Every result produced in this project must be saved in three forms:
>   1. JSON - raw data, machine-readable, source of truth
>   2. Markdown table - human-readable, goes into session logs and PRs
>   3. Figure - visual, goes directly into the thesis document
>
> This file is the single reference for WHAT to produce, WHEN, and WHERE it goes.
> Last updated: 2026-06-25

---

## Output Directory Structure

```
results/
├── baselines/
│   ├── random_walk.json
│   ├── vanilla_lstm.json
│   ├── grid_search.json
│   ├── pso_lstm.json
│   ├── ga_lstm.json
│   ├── gwo_lstm.json
│   ├── woa_lstm.json
│   └── sequential_fa.json
├── momfa/
│   └── momfa.json
├── tables/
│   ├── random_walk.md
│   ├── vanilla_lstm.md
│   ├── comparison_all_methods.md
│   ├── comparison_all_methods.csv    ← for Excel / supervisor review
│   └── comparison_all_methods.tex   ← paste directly into thesis LaTeX
└── figures/
    ├── data/
    │   ├── fold_splits_{ticker}.png         ← one per ticker (show train/val windows)
    │   └── indicator_correlation.png         ← heatmap of 25 indicators (JKH)
    ├── baselines/
    │   ├── rmse_bar_comparison.png           ← mean RMSE across all methods
    │   ├── rmse_boxplots.png                 ← 30-run distribution per method
    │   └── wilcoxon_heatmap.png              ← p-values: methods × tickers
    ├── momfa/
    │   ├── pareto_front_{ticker}.png         ← RMSE vs param count scatter
    │   ├── hypervolume_convergence.png       ← HV per iteration, mean ± std across 30 runs
    │   └── feature_selection_heatmap.png     ← 25 features × 10 tickers selection frequency
    └── predictions/
        └── predictions_{ticker}.png          ← best MOMFA prediction vs actual (last val fold)
```

---

## Figure Specifications

### F-01: Fold Splits (data/fold_splits_{ticker}.png)
- **What:** Stock closing price time series with 5 walk-forward validation windows overlaid
- **When to produce:** After data pipeline is confirmed
- **X-axis:** Date
- **Y-axis:** Close price (LKR)
- **Visual:** Shaded grey = training regions, shaded colour = validation windows (5 colours)
- **Why:** Shows the committee that no data leakage occurred; standard in time-series ML papers

### F-02: Indicator Correlation Heatmap (data/indicator_correlation.png)
- **What:** 25×25 Pearson correlation matrix of all technical indicators (JKH data)
- **When to produce:** After data pipeline is confirmed
- **Why:** Motivates feature selection - high correlations among trend indicators confirm redundancy

### F-03: RMSE Bar Comparison (baselines/rmse_bar_comparison.png)
- **What:** Grouped bar chart - 9 methods × 10 tickers (or averaged across tickers)
- **When to produce:** After all baselines + MOMFA are run
- **Y-axis:** Mean RMSE (LKR)
- **Error bars:** ±1 std across 30 runs
- **Reference line:** Random Walk RMSE (sanity floor)
- **Why:** Primary result figure for thesis Chapter 4

### F-04: RMSE Box Plots (baselines/rmse_boxplots.png)
- **What:** 30-run RMSE distribution per method (one subplot per ticker, or averaged)
- **When to produce:** After all 30-seed runs complete
- **Why:** Shows statistical variability of stochastic methods - directly supports Wilcoxon argument

### F-05: Wilcoxon P-value Heatmap (baselines/wilcoxon_heatmap.png)
- **What:** Heatmap of Wilcoxon signed-rank p-values: rows = methods, cols = tickers
- **Cells:** p-value, coloured green (p < 0.05 = significant) / red (not significant)
- **Compared against:** MOMFA (i.e., each cell = p-value for "MOMFA vs this method on this ticker")
- **When to produce:** After all methods have 30 runs
- **Why:** Core statistical evidence for the thesis claim

### F-06: Pareto Front (momfa/pareto_front_{ticker}.png)
- **What:** Scatter plot of all archive solutions: x = param count, y = val RMSE
- **One plot per ticker** (10 total); also one aggregate plot
- **Colour:** Each point coloured by number of selected features
- **Annotate:** Best RMSE solution and minimum-params solution
- **Why:** The KEY thesis figure - shows the trade-off MOMFA discovers

### F-07: Hypervolume Convergence (momfa/hypervolume_convergence.png)
- **What:** Line plot - mean hypervolume ± std across 30 runs, per iteration (1–30)
- **One subplot per ticker** OR one averaged line
- **Why:** Answers OQ-002 (does alpha decay work?) and OQ-003 (is N=20, T=30 enough?)

### F-08: Feature Selection Heatmap (momfa/feature_selection_heatmap.png)
- **What:** Heatmap - rows = 25 features (grouped by category), cols = 10 tickers
- **Cell value:** Fraction of Pareto-optimal solutions that selected that feature (0.0–1.0)
- **Why:** Shows which indicators matter for CSE stocks - novel empirical finding

### F-09: Predictions vs Actual (predictions/predictions_{ticker}.png)
- **What:** Time series - actual close price (solid), MOMFA best prediction (dashed)
- **Window:** Last validation fold only
- **One plot per ticker** (10 total)
- **Why:** Visual sanity check; standard in forecasting papers

---

## Table Specifications

### T-01: Per-Method Summary Table
Produced by `logger.save_method_table()` after each baseline run.

```
| Ticker    | RMSE (mean±std) | MAE   | MAPE  | DA    |
|-----------|-----------------|-------|-------|-------|
| JKH       |                 |       |       |       |
| ...       |                 |       |       |       |
| Average   |                 |       |       |       |
```

### T-02: Full Comparison Table
Produced by `logger.save_comparison_table()` after all methods are run.

```
| Method            | RMSE       | MAE    | MAPE   | DA     |
|-------------------|------------|--------|--------|--------|
| Random Walk       |            |        |        |        |
| Vanilla LSTM      |            |        |        |        |
| Grid Search LSTM  |            |        |        |        |
| PSO-LSTM          |            |        |        |        |
| GA-LSTM           |            |        |        |        |
| GWO-LSTM          |            |        |        |        |
| WOA-LSTM          |            |        |        |        |
| Sequential FA     |            |        |        |        |
| **MOMFA (ours)**  |            |        |        |        |
```
Values = mean across 30 runs × 10 tickers. Bold = best per column.

Also exported as `.csv` (for supervisor review) and `.tex` (for thesis).

---

## Production Checklist

### After data pipeline:
- [ ] F-01: Fold splits (all 10 tickers)
- [ ] F-02: Indicator correlation heatmap

### After each baseline is run (30 seeds):
- [ ] T-01: Per-method summary markdown table
- [ ] JSON already saved by the baseline script itself

### After ALL baselines + MOMFA are run:
- [ ] T-02: Full comparison table (.md + .csv + .tex)
- [ ] F-03: RMSE bar comparison
- [ ] F-04: RMSE box plots
- [ ] F-05: Wilcoxon p-value heatmap
- [ ] F-06: Pareto fronts (all tickers)
- [ ] F-07: Hypervolume convergence
- [ ] F-08: Feature selection heatmap
- [ ] F-09: Predictions vs actual (best MOMFA solution, all tickers)

---

## Implementation Location

| Utility | File |
|---------|------|
| Markdown/CSV/LaTeX tables | `src/utils/logger.py` |
| All figures | `src/utils/visualizer.py` |
| Wilcoxon test | `src/utils/metrics.py` → `wilcoxon_test()` |
| Hypervolume calculation | `src/utils/metrics.py` → `hypervolume()` |
