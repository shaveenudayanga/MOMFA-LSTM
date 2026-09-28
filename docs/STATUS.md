# STATUS.md
# Current Implementation State

> Read this first in every new session, then the files it points to.
> Update it at the END of every session.
> Last updated: 2026-09-29

---

## How the docs fit together (read in this order)

| File | Role |
|---|---|
| `MOMFA_research_proposal.txt` | **The specification.** Sections 0-8 = approved proposal, 9-12 = implementation addendum, 13 = working rules. Notes marked `[DECIDED date]` record author-approved changes. |
| `docs/DECISIONS.md` | Why each non-obvious choice was made (D-001 ...). Where it differs from the spec text, the spec has a `[DECIDED]` note pointing here. |
| `docs/FINDINGS.md` | What actually happened when code ran: bugs (B-xxx), findings (F-xxx), negative results (N-xxx), open questions (OQ-xxx), benchmarks. |
| `docs/RESULTS_PLAN.md` | Which figures and tables exist and how to regenerate them. |
| `docs/CONTEXT.md` | Short summary of the method for orientation. Not authoritative. |
| `CLAUDE.md` | Rules for AI assistants. |

If two documents disagree: spec (including its `[DECIDED]` notes) > DECISIONS.md > everything else.
Fix the disagreement in the same session rather than leaving it.

---

## Current Phase

**Build order (spec 13 R7) step 5 complete. Next: step 6, grid search full run.**

| Step | What | Status |
|---|---|---|
| 1 | Data download + storage | DONE (10 tickers, 2021-03-12 to 2025-12-29, 1,252 rows each) |
| 2 | Indicators + leakage tests | DONE (25 indicators; tests/test_data_split.py, 17 tests pass) |
| 3 | Walk-forward splitter | DONE (80/20 dev/test split, 5 folds x 126 days in dev) |
| 4 | LSTM model + trainer | DONE (PyTorch; return target, D-009) |
| 5 | Random walk + vanilla LSTM | DONE (30 seeds x 10 tickers, results in results/baselines/) |
| 6 | Grid search | CODE DONE, smoke-tested; full run NOT STARTED |
| 7 | PSO / GA / GWO / WOA | NOT STARTED (files exist but are empty) |
| 8 | MOMFA | CODE EXISTS (encoding, Levy, sigmoid, archive, cache); never run end to end on the full budget; results logging per spec 10 missing |
| 9 | Results logging (spec 10 per-seed layout) | NOT STARTED (current format: one JSON per method) |
| 10 | Plots | DONE for baselines (plots/make_figures.py); MOMFA figures pending |
| 11 | Statistics (Wilcoxon, Holm) | NOT STARTED |
| 12 | Ablation | NOT STARTED |

---

## Key results so far (final test period)

| Method | Beats random walk (RMSE) | RMSE ÷ RW (median) | MAE ÷ RW (median) | MAPE | DA |
|---|---|---|---|---|---|
| Random walk | - | 1.000 | 1.000 | 1.01% | 17.9% |
| Vanilla LSTM | 0/10 tickers | 1.104 | 1.224 | 1.30% | 40.3% |

Full tables: `results/tables/`. Figures: `results/figures/`. See FINDINGS N-001, F-003, F-004.

---

## How to run things (from the repo root)

```bash
venv/bin/python -m unittest discover -s tests -v          # leakage / alignment tests
PYTHONPATH=src venv/bin/python -m baselines.random_walk   # seconds
PYTHONPATH=src venv/bin/python -m baselines.vanilla_lstm  # ~45 min on the laptop GPU
PYTHONPATH=src venv/bin/python -m baselines.grid_search   # ~27 h on the laptop GPU
venv/bin/python plots/make_figures.py                     # all figures + tables from saved JSON
```

Run scripts from the repo root: `load_ticker()` reads `data/preprocessed/` relative to it.
Shared settings (split, training protocol, tickers, method order) live in `src/config.py`.
Every method scores configurations through `src/evaluation.py` (`cross_validate`, `evaluate_on_test`).

---

## Open items waiting for the author / supervisor

| Item | Where | Blocks |
|---|---|---|
| Price-level indicators (EMA, SMA, BB, PSAR, VWAP) leave the training range on the test period | FINDINGS F-002; progress report 01, discussion point 1 | Should be decided before the expensive runs (steps 7-8) |
| Report DA on non-flat days as an extra measure? | FINDINGS F-003; progress report 01, discussion point 2 | Nothing (reporting only) |
| Compute plan for steps 7-8 (laptop GPU vs paid cloud) | FINDINGS "Performance Benchmarks" | Steps 7-8 |
| Pareto archive size cap (spec 9.7 [OPEN], default 50 + crowding) | Not implemented in src/firefly/pareto.py | Step 8 |
| Sequential FA-LSTM baseline (D-004) | No code yet | Step 7 |

## Known issues (not blocking)

- `notebooks/02_results_analysis.ipynb` calls table/plot functions that were removed; use
  `plots/make_figures.py` instead or update the notebook.
- `notebooks/01_data_exploration.ipynb` calls `get_folds(df, ...)` on the full data (includes the
  test period). Exploration only; real runs use `split_dev_test()` first.
- Sanity check S2 (lag check) is not implemented (FINDINGS F-004).

---

## Session Log

| Date | What happened |
|---|---|
| 2026-06-16 | Project started. Context documents created. |
| 2026-06-24/25 | Preprocessing, splitter, model, trainer, MOMFA code, vanilla/random-walk baselines written. |
| 2026-09-28/29 | Audit against the spec. Added final test period (9.1), same-date evaluation (9.3), fixed one-day target offset, return target (D-009), final-model early stopping (9.1.1, D-008), vanilla 1 layer, lr log scale, shared evaluation/config modules, 17 leakage tests. Full random walk + vanilla runs. All figures/tables rebuilt from saved files (all four metrics). Modal timing benchmark. Progress report 01 (reports/progress_report_01/). Bugs B-001..B-009 logged. |
