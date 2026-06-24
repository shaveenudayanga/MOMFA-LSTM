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

*No findings yet. First entries will be added after pipeline verification.*

---

## Performance Benchmarks (populated during experiments)

| Ticker | Method | Approx. time per run | Hardware | Notes |
|---|---|---|---|---|
| (empty) | | | | |

---

## Bug Registry

| ID | Date | Description | Root cause | Fix | Status |
|---|---|---|---|---|---|
| (empty) | | | | | |

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

## Unexpected Positive Results

*To be recorded when something works better than expected.*

---

## Unexpected Negative Results

*To be recorded when something works worse than expected or fails a sanity check.*
