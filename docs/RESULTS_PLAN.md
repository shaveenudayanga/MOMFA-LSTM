# RESULTS_PLAN.md
# Output Strategy - Figures, Tables, and Human-Readable Results

> Every result is saved as JSON first (source of truth), then turned into figures and
> tables by ONE script that reads only saved files (spec 10, 11):
>
>     venv/bin/python plots/make_figures.py
>
> Nobody should need to read JSON. Every result must also exist as a figure or table.
> Last updated: 2026-09-29

---

## Output Directory Structure

```
results/
├── baselines/                 ← raw results, one JSON per method (written by the run scripts)
│   ├── random_walk.json
│   ├── vanilla_lstm.json
│   └── grid_search.json ...
├── momfa/                     ← MOMFA results (not run yet)
├── tables/                    ← written by plots/make_figures.py
│   ├── method_summary.md/.csv/.tex     T1: one row per method, test period, all four measures
│   ├── test_by_ticker.md/.csv          every method x ticker, test period
│   └── validation_by_ticker.md/.csv    every method x ticker, walk-forward validation
└── figures/                   ← written by plots/make_figures.py
    ├── data/
    │   ├── fold_splits_{ticker}.png          F1  (all 10 tickers)
    │   └── indicator_correlation_JKH.png     F2
    ├── predictions/
    │   └── test_predictions_{ticker}.png     F3  actual vs predicted, test period + 30-day zoom
    └── comparison/
        ├── metric_overview.png               all four measures (RMSE, MAE, MAPE, DA) in one figure
        ├── error_distribution.png            F4  % errors, all tickers pooled
        ├── rmse_vs_random_walk_boxplots.png  F5  30-run test RMSE ÷ random-walk RMSE
        ├── rmse_vs_random_walk_heatmap.png   F5  mean ratio, tickers x methods (S1 at a glance)
        ├── mae_vs_random_walk_boxplots.png   same for MAE
        ├── mae_vs_random_walk_heatmap.png    same for MAE
        ├── mape_heatmap.png                  F6  tickers x methods
        └── directional_accuracy.png          F7  vs 50% and majority-direction rate (S3)
```

MOMFA-specific figures (spec 11 D: Pareto front F8, hypervolume F9, feature selection F10,
hyperparameter distribution F11) and the Wilcoxon table (T2) are added once MOMFA and the
statistics step have run. Plot functions for F8-F10 and the Wilcoxon heatmap already exist
in src/utils/visualizer.py.

---

## Rules

- **Test period is the headline.** Tables and comparison figures show the final test period.
  Validation numbers are in validation_by_ticker only.
- **All four measures of spec 4.B (RMSE, MAE, MAPE, DA) appear in the tables and figures.**
- **Never average RMSE or MAE across tickers.** They are in LKR and prices differ ~15x between
  tickers. Cross-ticker comparisons use RMSE ÷ random-walk RMSE, MAPE and DA.
- **Plotted predictions** use the seed with the best VALIDATION RMSE, never the best test RMSE.
- **Fixed colour per method** (src/utils/visualizer.py METHOD_COLORS); random walk is neutral grey.
- **Regenerate after every run:** figures are cheap to rebuild and must always match the JSON.

---

## Implementation Location

| Utility | File |
|---------|------|
| Build all figures + tables | `plots/make_figures.py` |
| Plot functions | `src/utils/visualizer.py` |
| Load results, write tables | `src/utils/logger.py` |
| Wilcoxon test | `src/utils/metrics.py` → `wilcoxon_test()` |
