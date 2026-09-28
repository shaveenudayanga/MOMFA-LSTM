# FINDINGS.md
# Empirical Findings Log

> Record what the code ACTUALLY does when run.
> This is separate from what the proposal SAYS it should do.
> Surprises, bugs fixed, parameter behaviour, unexpected results - all go here.
> Purpose: build up institutional memory that cannot be reconstructed from code.

---

## Finding Template

```
## F-XXX: Short title

**Date:** YYYY-MM-DD
**Phase:** Data / LSTM / MOMFA / Baselines / Evaluation
**Type:** Bug / Observation / Insight / Warning
**Status:** OPEN (still relevant) / RESOLVED / INCORPORATED INTO DESIGN

**What happened:**

**Why it matters:**

**What we did about it:**

**Open questions raised:**
```

---

## Open Theoretical Questions (from CONTEXT.md, to be answered here)

These questions from the proposal's open questions list will be resolved by experiments.
Update each entry when you have empirical evidence.

### OQ-001: Levy saturation on binary dimensions

**Question:** Does Levy flight on binary dimensions (pre-sigmoid) saturate the sigmoid,
collapsing feature selection diversity to near-deterministic 0/1?

**Status:** OPEN - not yet tested

**What to look for:** Plot the distribution of sigmoid inputs (real-valued binary dimensions)
after several iterations. If most values are < -5 or > 5, saturation is occurring.
Also plot feature selection frequency across the population over iterations - if all
fireflies converge to the same mask within 5-10 iterations, saturation is confirmed.

**Mitigation if confirmed:** Apply Levy flight only to the 6 continuous hyperparameter
dimensions. Use a smaller-scaled random walk for the 25 binary dimensions.
Record this as a new DECISION if mitigation is applied.

---

### OQ-002: Does alpha decay do meaningful work in T=30?

**Question:** alpha_0=0.5, delta=0.97, T=30 gives alpha_30 ≈ 0.20.
Range is only 0.5 to 0.20. Does "broad early, refine late" actually manifest?

**Status:** OPEN - not yet tested

**What to look for:** Plot RMSE of best archive solution per iteration.
If convergence happens in the first 5-10 iterations and later iterations produce
no improvement, the decay is not contributing. If there is clear late-stage refinement,
it is working.

**Mitigation if confirmed as not working:** Increase T to 50 or steepen delta to 0.93.
Record as new DECISION if changed.

---

### OQ-003: Is N=20, T=30 sufficient for 31-D multi-objective search?

**Question:** Population of 20 over 30 iterations = 600 total evaluations for 31-D space.
Is Pareto coverage adequate?

**Status:** OPEN - not yet tested

**What to look for:** Plot hypervolume per iteration. If hypervolume is still increasing
sharply at iteration 30, budget is insufficient. If it plateaus before iteration 20,
budget is adequate.

**Mitigation if confirmed as insufficient:** This requires a real conversation with
Prof. Fernando because increasing N or T significantly changes the compute budget.
Do NOT silently increase these values. Record as new DECISION.

---

### OQ-004: Random archive guide + distance-based attraction interaction

**Question:** When a firefly moves toward a randomly selected archive member,
the distance-based attractiveness term still applies. But all archive members are
non-dominated, so brightness comparison is irrelevant. Is this coherent?

**Status:** OPEN - theoretical, may not need empirical resolution

**What to look for:** Compare archive diversity between runs where all fireflies move
toward the same archive guide vs. random guide. If random guide produces more diverse
Pareto fronts, the mechanism is working even if not formally justified.

---

## Findings Log (populated as experiments run)

## F-001: Predicting the price LEVEL fails when test prices leave the training range

**Date:** 2026-09-28
**Phase:** Baselines (vanilla LSTM, first full run with the untouched test period)
**Type:** Warning -> methodological fix
**Status:** INCORPORATED INTO DESIGN (DECISIONS.md D-009, spec 9.2.1)

**What happened:**
The first vanilla LSTM run with a real final test period (spec 9.1) gave test RMSE far
worse than the random-walk floor on several tickers (30 seeds, mean):

| Ticker | Vanilla test RMSE (level target) | Random walk test RMSE |
|--------|------------------|------------------|
| JKH    | 0.51  | 0.28 |
| COMB   | 17.93 | 2.34 |
| DIAL   | 5.55  | 0.32 |
| HNB    | 57.54 | 5.00 |
| LOLC   | 24.02 | (see random_walk.json) |

Cause: CSE prices rose strongly in the test year, far above anything in training
(COMB training max 111 LKR vs test up to 214; HNB 205 vs 424; DIAL 13.3 vs 32).
The Min-Max scaler is fit on training rows only (correct, no leakage), so test targets
become 2-3 instead of <= 1. The LSTM never saw such values and stays near the old
maximum. This is a property of "level target + Min-Max scaling", not a code bug, and
MOMFA would have suffered from it too.
Evidence log: logs/vanilla_lstm_level_target_BEFORE_D009.log

**Why it matters:**
Any comparison would mostly measure how badly each method extrapolates, not how well
it forecasts. Every learned model would fail sanity check S1 by a wide margin.

**What we did about it:**
The LSTM now predicts the next-day return r = Close[t+1]/Close[t] - 1; the forecast
price is Close[t] x (1 + r_hat). All metrics are still computed on the next-day close
price in LKR. Same method, seed 0:

| Ticker | Test RMSE before | Test RMSE after | Random walk |
|--------|-------|-------|-------|
| JKH    | 0.51  | 0.29  | 0.28 |
| COMB   | 17.93 | 2.84  | 2.34 |
| DIAL   | 5.55  | 0.41  | 0.32 |
| HNB    | 57.54 | 5.53  | 5.00 |

After the fix the LSTM is close to, but still slightly worse than, random walk.

**Open questions raised:**
- Several input indicators are also price levels (EMA9/21/50, SMA20, PSAR, BB upper/lower,
  VWAP, OBV, vol_sma) and are also out of range on the test period. Not changed yet
  (they are the spec's feature pool); watch whether this hurts. See F-002.
- Directional accuracy is low (0.30-0.45), see F-003.

## F-002: Price-level input indicators are out of the training range on the test period

**Date:** 2026-09-28
**Phase:** Data / Baselines
**Type:** Warning
**Status:** OPEN

**What happened:** Same cause as F-001, but on the input side: level-type indicators
(EMA9, EMA21, EMA50, SMA20, PSAR, BB-U, BB-L, VWAP, OBV, VSMA) scale above 1 on the test
period. D-009 fixed the target only.

**Why it matters:** The LSTM may treat these inputs unreliably on the test period.
Methods that select fewer level-type indicators could gain an advantage for this reason.

**What we did about it:** Nothing yet; needs an author/supervisor decision before
changing the spec's feature pool or its scaling.

## F-003: Directional accuracy is pulled down by flat days

**Date:** 2026-09-28
**Phase:** Evaluation
**Type:** Observation
**Status:** OPEN

**What happened:** Random walk DA is 0.14-0.35 and vanilla LSTM test DA 0.30-0.45.
Many days have zero price change (forward-filled holidays, and illiquid days with the
same close). On those days the actual direction is 0 and a learned model's direction
is almost never exactly 0, so they count as misses.

**Why it matters:** DA below 50% looks like "worse than a coin flip" but is partly an
artefact of flat days. Sanity check S3 must report this honestly.

**What we did about it:** Nothing yet. Options (needs decision): report DA on non-flat
days only in addition to the spec formula; or keep the spec formula and explain it.

---

## Performance Benchmarks (populated during experiments)

| Ticker | Method | Approx. time per run | Hardware | Notes |
|---|---|---|---|---|
| (empty) | | | | |

---

## Bug Registry

| ID | Date | Description | Root cause | Fix | Status |
|---|---|---|---|---|---|
| B-001 | 2026-09-28 | No final test period: folds ran to the last row, so every reported number was a validation number (spec 9.1) | get_folds() used the full series | split_dev_test() holds out the last 20% (250 rows); folds use the development period only | FIXED |
| B-002 | 2026-09-28 | Candidates with different lookbacks were scored on different dates (lookback 5 → 121 days, 60 → 66 days per fold) (spec 9.3) | Validation windows were built from validation rows only | make_fold_sequences() takes the first lookback-1 input rows from the training period; every candidate scores all 126 dates | FIXED |
| B-003 | 2026-09-28 | Model predicted Close[t+2] instead of Close[t+1] (spec 9.2) | target is already shifted by data_loader, and make_sequences() shifted it again (y[lookback:]) | Window ending at row t is paired with y[t] (y[lookback-1:]) | FIXED |
| B-004 | 2026-09-28 | Vanilla LSTM used 2 layers; spec 9.9 default is 1 | Config not checked against spec | n_layers = 1 | FIXED |

| B-005 | 2026-09-28 | MOMFA decoded learning rate on a linear scale; spec 9.6 says log scale. With linear decoding ~90% of candidates had lr > 0.001, so 1e-4..1e-3 was barely searched | Spec not followed, not logged | lr = 10^(-4 + 2u) in _decode_hyperparams | FIXED |
| B-006 | 2026-09-29 | Prediction plots were labelled "MOMFA prediction" although they showed vanilla LSTM output (MOMFA has never run) | Legend text hard-coded in plot_predictions() | Plots take the method from the results file; label = method name | FIXED |
| B-007 | 2026-09-29 | Tables and RMSE plots showed VALIDATION metrics, not the final test period; the RMSE bar chart averaged RMSE across tickers with different price scales (LOLC ~300 LKR vs JKH ~20); the last bar was always coloured as "MOMFA"; random_walk.md, fold_splits_JKH.png and indicator_correlation.png were left over from before the test split existed | logger/visualizer read the old keys and were never updated after D-009/9.1.1; figures were drawn inside training scripts instead of from saved files (spec 10) | plots/make_figures.py rebuilds every figure and table from saved JSON; primary tables use test metrics; cross-ticker summaries use RMSE ÷ random-walk RMSE, MAPE and DA only; fixed colour per method; stale files moved out | FIXED |
| B-008 | 2026-09-29 | Summary table, figures and progress report showed RMSE (and MAPE) but not all four measures required by spec 4.B; MAE had no figure and was missing from the summary table and report | Only RMSE plots existed; report table was cut to save space | Summary table has RMSE, MAE, MAPE, DA; MAE gets the same ratio-to-random-walk box plot and heatmap as RMSE; new metric_overview.png shows all four; report has two tables and the overview figure | FIXED |
| B-009 | 2026-09-29 | Cache key rounded lr to 6 decimal places, not 6 significant figures as D-003 says; with lr down to 1e-4 that keeps only 2-3 significant digits | round(lr, 6) | lr formatted with 6 significant figures in make_cache_key | FIXED |
Tests: tests/test_data_split.py (17 tests, all pass). Results produced before these fixes (results/baselines/*.json) are invalid and must be re-run.

---

## Feature Calculation Verification

When the feature pipeline is confirmed working, fill this table:

| Ticker | Rows | Missing before fill | Missing after fill | All 25 indicators present | Verified by |
|---|---|---|---|---|---|
| JKH | | | | | |
| COMB | | | | | |
| DIAL | | | | | |
| HNB | | | | | |
| LOLC | | | | | |
| SAMP | | | | | |
| (4 more) | | | | | |

---

## Notes on CSE Data Quality

*Populated as data issues are discovered during pipeline development.*

Known expectations based on CSE characteristics:
- More missing trading days than NYSE-level markets (public holidays, trading suspensions)
- Lower volume on some tickers may cause NaN in volume indicators (VWAP, CMF) on
  zero-volume days - forward fill should handle this but verify
- Price gaps between sessions may be large - check for outliers in ROC and VROC

---

## F-004: Vanilla LSTM predictions track yesterday's close (persistence-like)

**Date:** 2026-09-29
**Phase:** Baselines
**Type:** Observation
**Status:** OPEN - sanity check S2 (lag check) must be computed for every method

**What happened:** In results/figures/predictions/test_predictions_*.png the vanilla LSTM
line almost overlaps the random-walk line (today's close). The predicted return is close
to zero most days, so the model mostly repeats today's price.

**Why it matters:** This is the failure S2 exists to catch. It explains why vanilla sits
just above 1.0 x random-walk RMSE on most tickers (N-001).

**What we did about it:** Nothing yet. S2 (corr(pred_t, actual_{t-1}) vs corr(pred_t,
actual_t)) is not implemented; it belongs with the other sanity checks (spec 12.1).

## Unexpected Positive Results

*To be recorded when something works better than expected.*

---

## Unexpected Negative Results

## N-001: Vanilla LSTM fails sanity check S1 on all 10 tickers

**Date:** 2026-09-29
**Phase:** Baselines
**Status:** OPEN - expected for this baseline, watch whether MOMFA/tuned baselines also fail it

Vanilla LSTM (spec 9.9 fixed config: 1 layer, 64 units, lr 0.001, dropout 0.2,
lookback 20, Adam, all 25 features, return target per D-009) test RMSE, 30 seeds:

| Ticker | Vanilla test RMSE | Random walk test RMSE | Vanilla worse by |
|---|---|---|---|
| JKH  | 0.289 | 0.282 | 2% |
| COMB | 2.687 | 2.335 | 15% |
| DIAL | 0.555 | 0.321 | 73% |
| HNB  | 5.884 | 5.004 | 18% |
| LOLC | 10.024 | 9.097 | 10% |
| SAMP | 1.836 | 1.771 | 4% |
| NTB  | 4.069 | 3.697 | 10% |
| HHL  | 0.474 | 0.428 | 11% |
| DIST | 0.994 | 0.959 | 4% |
| HAYL | 3.945 | 2.634 | 50% |

Sanity check S1 (12.1): 0/10 tickers beat random walk. Reported plainly per spec 12.2,
not hidden.

**Reading this:** an untuned, fixed-hyperparameter LSTM with no feature selection losing
to persistence is a known pattern in the literature (this is exactly the failure mode
S1 exists to catch) and is expected for baseline 1, which is deliberately the lower
bound. It becomes a real problem only if the tuned baselines (PSO/GA/GWO/WOA) and MOMFA
also fail S1 - that would suggest the LSTM architecture or training protocol itself
cannot beat persistence on CSE data regardless of tuning. Cannot be judged yet with
only 2 of 9 methods run.
