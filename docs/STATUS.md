# STATUS.md
# Current Implementation State

> Update this file at the START and END of every Claude session.
> A new session should read CONTEXT.md first, then this file, and know exactly what to do.
> Last updated: 2026-06-24

---

## Current Phase

**Phase 1: Data Pipeline**

---

## What Is Done

| Component | Status | Notes |
|---|---|---|
| Repo created | DONE | https://github.com/shaveenudayanga/MOMFA-LSTM |
| Requirements file | DONE | requirements.txt present |
| README | DONE | Basic description |
| 25 technical indicators calculation | DONE | Using pandas-ta |
| Data fetching from TradingView | DONE | 10 tickers in data/, 2021-2025 |
| Preprocessing pipeline (forward fill, Min-Max) | DONE | src/preprocessor.py - per-fold scaler, no leakage |
| Walk-forward validation splitter | DONE | get_folds() in preprocessor.py, 5 folds × 126 rows |
| LSTM model builder | NOT STARTED | |
| Training protocol (epochs, early stop, MSE) | NOT STARTED | |
| Fitness function (RMSE + param count) | NOT STARTED | |
| Fitness cache | NOT STARTED | |
| Sigmoid transfer function | NOT STARTED | |
| Levy flight step | NOT STARTED | |
| Firefly position update rule | NOT STARTED | |
| Pareto archive (phenotype storage) | NOT STARTED | |
| Full MOMFA loop | NOT STARTED | |
| Baseline 1: Vanilla LSTM | NOT STARTED | |
| Baseline 2: Grid Search | NOT STARTED | |
| Baseline 3: PSO-LSTM | NOT STARTED | |
| Baseline 4: GA-LSTM | NOT STARTED | |
| Baseline 5: GWO-LSTM | NOT STARTED | |
| Baseline 6: WOA-LSTM | NOT STARTED | |
| Baseline 7: Sequential FA-LSTM | NOT STARTED | Most important - do not skip |
| Baseline 8: Random Walk Persistence | NOT STARTED | Simplest - do early as sanity check |
| Evaluation metrics (RMSE, MAE, MAPE, DA, HV) | NOT STARTED | |
| Wilcoxon test runner (30 seeds) | NOT STARTED | |
| Results logging | NOT STARTED | |

---

## What To Do Next (Immediate)

1. **`src/utils/metrics.py`** - RMSE, MAE, MAPE, Directional Accuracy (all on inverse-transformed values). Small, needed by fitness function and all baselines.
2. **`src/utils/seed.py`** - reproducible seed setter for numpy + tensorflow. Required before any training run.
3. **`src/model.py`** - LSTM builder: takes (n_layers, units, lr, dropout, lookback, optimizer_name, n_features) and returns a compiled Keras model. Unlocks the fitness function.
4. **`src/train.py`** - training loop with early stopping (patience=10, monitor val RMSE, max 50 epochs, batch 32, MSE loss). Returns history + val RMSE.

**Before touching the MOMFA algorithm:** model.py and train.py must be solid, because every fitness evaluation calls them.

---

## Current Blockers

- None confirmed yet. First session should start by reading what is in `src/`.

---

## File Structure (last known)

```
MOMFA-LSTM/
├── src/
│   └── (unknown contents - check first)
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Session Log

| Date | What happened | Who |
|---|---|---|
| 2026-06-16 | Project started. Context documents created. Feature calculation reportedly done. | Udayanga + Claude |

---

## Notes for Next Session

- Read CONTEXT.md completely before doing anything
- First task: `ls -la src/` equivalent - understand what files exist
- Check whether features are calculated for all 10 tickers or just one
- Confirm output format of feature calculation (CSV? DataFrame saved to disk? In-memory only?)
- The output format will determine how the LSTM pipeline reads data
