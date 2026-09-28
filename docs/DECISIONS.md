# DECISIONS.md
# Design Decisions Log

> Record every non-obvious implementation choice here with the reasoning and rejected alternatives.
> Purpose 1: stop relitigating the same questions in future sessions.
> Purpose 2: thesis defense preparation - each entry answers a realistic examiner question.
>   "Why parallel?" → D-006. "Why phenotypes not genotypes?" → D-002. "Why raw param count?" → D-001.
>   Point to the document, not your memory, when asked.
> When a decision is made during a session, add it here before the session ends.

---

## Decision Template

```
## D-XXX: Short title

**Date:** YYYY-MM-DD
**Status:** ACCEPTED / SUPERSEDED BY D-YYY
**Context:** What situation forced this decision?
**Decision:** What we chose.
**Reasoning:** Why this over the alternatives.
**Rejected alternatives:** What else was considered and why not.
**Consequences:** What this decision locks in or makes harder.
```

---

## D-001: Complexity metric = raw trainable parameter count only

**Date:** 2026-06-16 (proposal revision, before implementation)
**Status:** ACCEPTED

**Context:**
The original proposal defined complexity as (number of selected features × number of trainable
parameters). This was flagged as problematic because the number of trainable parameters in the
first LSTM layer already scales with input width (feature count). Multiplying by feature count
again double-counts the influence of input size and produces a metric with no interpretable units.

**Decision:**
Complexity objective f2 = total number of trainable parameters in the LSTM network.
Feature count is NOT included as a multiplier.

**Reasoning:**
Raw parameter count has clear units, is computationally interpretable, and is a standard
proxy for model size in the deep learning literature. It still penalizes wide/deep networks
without the double-counting problem.

**Rejected alternatives:**
- Features × params: double-counts, uninterpretable units
- Inference FLOPs: correct but harder to compute, overkill for this project
- Feature count alone: does not penalize network size at all

**Consequences:**
The Pareto front's x-axis (complexity) now does not explicitly separate "too many features"
from "too large a network." These are conflated in the param count. The visual Pareto front
will show param count only. This is a known trade-off accepted for defensibility.

---

## D-002: Pareto archive stores phenotypes, not genotypes

**Date:** 2026-06-16 (before implementation)
**Status:** ACCEPTED

**Context:**
The sigmoid transfer function (Eq. 5-6 in proposal) is stochastic - the same real-valued
position vector can produce different binary masks on different evaluations. If the archive
stores real-valued position vectors (genotypes) and re-decodes them on access, two things break:
(a) the fitness associated with an archive entry may not match the binary mask that gets
decoded next time, and (b) the Pareto front itself is noisy and unstable.

**Decision:**
The archive stores phenotypes: the fixed decoded binary mask (25 bits) + the real-valued
hyperparameter vector (6 values) + the two fitness values (RMSE, param count).
Once a solution is evaluated and enters the archive, its binary mask is frozen.
The real-valued position vector is kept separately for the purpose of computing
firefly movement, but the archive entry is the phenotype.

**Reasoning:**
A Pareto front whose members change identity between iterations is not a valid Pareto front.
The archive must be stable for Pareto dominance comparisons to be meaningful.

**Rejected alternatives:**
- Store genotypes and re-decode: unstable archive, violates Pareto dominance logic
- Deterministic threshold instead of stochastic sigmoid: changes the algorithm character,
  departs more from the proposal spec

**Consequences:**
The fitness cache and the archive must use the same key structure.
Key for caching = (tuple of binary mask bits, tuple of rounded hyperparameter values).
If two fireflies arrive at the same decoded phenotype, they share a cache entry.

---

## D-003: Fitness cache keyed on decoded phenotype, not real-valued position

**Date:** 2026-06-16 (before implementation)
**Status:** ACCEPTED

**Context:**
The proposal mentions fitness caching but does not specify what constitutes a cache hit.
Two options: cache on real-valued position vector, or cache on decoded phenotype (binary
mask + discretized hyperparameters).

**Decision:**
Cache on decoded phenotype: (binary_mask_tuple, discretized_hyperparams_tuple).
Hyperparameters are rounded to a reasonable precision (e.g., 4 decimal places for LR)
before forming the cache key.

**Reasoning:**
- Two different real-valued position vectors can decode to the same binary mask
  (because sigmoid is stochastic and many continuous values map to the same bit)
- Caching on phenotype catches more real duplicates and actually reduces computation
- Consistent with D-002: phenotype is the stable identity of a solution

**Rejected alternatives:**
- Cache on real-valued vector: misses duplicate phenotypes that cost a full LSTM training
- No caching: computationally infeasible given scale (see CONTEXT.md §3.5)

**Consequences:**
Need a deterministic discretization step for hyperparameters before forming cache keys.
Optimizer is already discrete (Adam/RMSprop/SGD). Layer count is integer.
Units per layer: round to nearest integer. LR: round to 6 significant figures.
Dropout: round to 2 decimal places. Lookback: round to nearest integer.

---

## D-004: Sequential FA-LSTM baseline is baseline 7, not optional

**Date:** 2026-06-16 (proposal revision)
**Status:** ACCEPTED

**Context:**
The original audit flagged that comparing MOMFA only against HPO-only baselines tests
"joint > HPO-only" which is almost a tautology (joint has strictly more search freedom).
The real claim in the introduction is "joint > sequential." Without a sequential baseline,
the headline contribution is not tested.

**Decision:**
Sequential FA-LSTM is a mandatory 7th baseline implemented as:
Step 1: Run FA for feature selection with hyperparameters fixed at default values.
Step 2: Using the selected features from step 1, run FA again for hyperparameter optimization.
Uses the same FA primitives (Levy flight, sigmoid transfer, same budget per phase).

**Reasoning:**
This is the exact strawman attacked in Fig. 1 of the proposal. Without it, a reviewer
could argue all gains come from feature selection alone, not from jointness.

**Rejected alternatives:**
- Skip it and hope reviewers don't notice: too risky for thesis defense
- Use a different algorithm for sequential baseline: would conflate algorithm choice
  with jointness. Same FA primitives isolates the jointness variable cleanly.

**Consequences:**
Adds one more baseline to implement. Total baselines = 8. Budget is higher but manageable.
Budget note: sequential FA has 2 × (N × T) evaluations, so roughly double the cost of
a single-phase FA run. Cache helps here if phase 1 and 2 share solutions.

---

## D-005: Per-ticker modeling, not pooled

**Date:** 2026-06-16 (proposal clarification)
**Status:** ACCEPTED

**Context:**
The proposal was ambiguous about whether one global model is trained across all stocks or
separate models per ticker. This changes param counts, complexity values, and generalization claims.

**Decision:**
Separate LSTM model trained per stock ticker. 10 models total.
MOMFA runs independently per ticker and returns 10 separate Pareto fronts.
Results are reported per ticker. Update 2026-09-29: RMSE and MAE are never averaged across
tickers (they are in LKR and prices differ ~15x); cross-ticker summaries use RMSE/MAE divided
by the random-walk value, MAPE and DA (see docs/RESULTS_PLAN.md).

**Reasoning:**
Each CSE stock has different volatility, liquidity, and sector characteristics.
A pooled model cannot discover stock-specific optimal feature subsets, which is
a core contribution of the joint optimization framework.

**Rejected alternatives:**
- Global pooled model: faster, but defeats the per-stock customization argument
- Cluster then pool: adds complexity without clear benefit for this scope

**Consequences:**
Compute cost scales linearly with ticker count (×10). This is the main factor in the
compute budget. See CONTEXT.md §3.5 for the full estimate.

---

## D-006: Parallel fitness evaluation within each MOMFA iteration

**Date:** 2026-06-24
**Status:** ACCEPTED

**Context:**
Scale estimate: 20 fireflies × 30 iterations × 30 seeds × 10 tickers = 180,000 LSTM
training calls (minus cache hits). Sequential evaluation makes this infeasible on a
single machine within any reasonable thesis timeline.

**Decision:**
Within each iteration, fitness evaluations are parallelised using `joblib.Parallel`.
The main process checks the cache first and dispatches only uncached phenotypes to workers.
Workers are stateless - they receive (mask, hyperparams, folds) and return (rmse, n_params).
The cache is updated in the main process after workers complete.

**Reasoning:**
All N=20 firefly evaluations within one iteration are fully independent: no firefly
needs another's fitness to compute its own. This is a textbook embarrassingly parallel
workload. Parallelising at the iteration level (not the run level) keeps the cache
effective within a run and avoids the complexity of cross-process shared state.

**Rejected alternatives:**
- Sequential evaluation: infeasible at scale
- Parallelise across seeds (30 runs): would require 30× memory for folds/models and
  lose the per-run cache benefit entirely
- Threading backend: GIL prevents true CPU parallelism for PyTorch training

**Update 2026-09-28:** on a machine with CUDA, evaluations run sequentially on the GPU
(`_resolve_n_jobs` in momfa.py), because several processes sharing one GPU conflict and
serialise anyway. Parallel workers are used on CPU-only machines.

**Consequences:**
Worker functions must be picklable (no lambdas, no closures capturing non-picklable state).
folds (numpy arrays + sklearn scalers) are picklable. The cache dict stays in the main process.

---

## D-007: Infeasible solution guard for all-zero binary mask

**Date:** 2026-06-24
**Status:** ACCEPTED

**Context:**
The stochastic sigmoid transfer (Eq. 5-6 in proposal) can produce an all-zero binary
mask, especially early in the search when continuous values for binary dims are very
negative. An all-zero mask means zero input features, which produces an invalid LSTM
with no input layer.

**Decision:**
After sigmoid decoding of binary dimensions 1-25, if `sum(mask) == 0`, randomly force
exactly one bit to 1 before passing to the fitness function. The bit is chosen uniformly
at random from all 25 positions using the run's seeded RNG. This guard lives in
`decode_position()` - at decode time, not inside training.

**Reasoning:**
Standard practice in binary metaheuristic feature selection literature. The fix is
minimal (one line) and preserves the stochastic nature of the search - the forced bit
is random, not biased toward any particular feature. Handling it at decode time keeps
`evaluate_phenotype` and `train_fold` clean and unaware of this edge case.

**Rejected alternatives:**
- Penalise all-zero mask with a very high fitness: still requires calling train_fold
  with an invalid LSTM, which crashes
- Re-sample the position: changes convergence dynamics unpredictably
- Minimum feature constraint during initialisation only: doesn't protect against drift
  during iteration updates

**Consequences:**
All-zero masks are corrected silently. The corrected mask is used for both evaluation
and as the phenotype stored in the cache and archive. This means the phenotype stored
may differ slightly from what the sigmoid transfer would have produced without the guard,
but the alternative is a crash.

---

## D-008: Final model early-stops on the last 126 development days

**Date:** 2026-09-28
**Status:** ACCEPTED (spec 9.1.1)

**Context:**
The final model of every method is retrained before its one-time score on the test
period (spec 9.1). Early stopping (patience 10) needs validation data, and the spec
did not say which data to use.

**Decision:**
Train on the development period minus its last 126 trading days; early-stop on those
126 days; score once on the test period. No second retrain on the full development
period. Same rule for MOMFA's selected models and every baseline
(`src/evaluation.py → evaluate_on_test`).

**Reasoning:**
The test period can never influence training, so it cannot be the stopping set.
126 days matches the walk-forward validation window size, so the stopping signal is
as reliable as during the search.

**Rejected alternatives:**
- Early-stop on the test period: leakage, invalidates the final numbers
- Retrain on all development rows for a fixed epoch count: adds a second rule to
  justify and a second training run per model
- No early stopping for the final model: breaks the fixed training protocol (3.F.3)

**Consequences:**
The final model sees 126 fewer training rows than the development period holds.
The slice size used is stored per result as `test_stop_size`.

---

## D-009: LSTM predicts the next-day return; metrics stay on the next-day close price

**Date:** 2026-09-28
**Status:** ACCEPTED by author (spec 9.2.1)

**Context:**
Predicting the Min-Max-scaled price level failed on the test period because CSE prices
rose far above the training range (FINDINGS.md F-001: vanilla test RMSE 57.5 vs random
walk 5.0 on HNB).

**Decision:**
The LSTM target is the next-day return r_t = Close[t+1] / Close[t] - 1, Min-Max scaled
with a scaler fit on training rows only. The forecast price is Close[t] x (1 + r_hat).
RMSE, MAE, MAPE and DA are computed on that price in LKR, exactly as before. Applies to
MOMFA and every learned baseline through src/evaluation.py.

**Reasoning:**
Returns stay in a similar range when the price level moves to a new regime, so the model
is never asked to output values it has never seen. It is the standard target in financial
forecasting, and the research question (forecasting the next-day close) is unchanged.
Random walk is the special case r_hat = 0, which makes the S1 comparison direct.

**Rejected alternatives:**
- Keep the level target: fails badly on the test period (F-001)
- Normalise each input window by its last close: also works, but harder to explain and
  changes the feature pipeline, not only the target
- Fit the scaler on all data: leakage (spec 9.2)

**Consequences:**
- Cleaned data files now also keep today's close (`close` column).
- Price-level input indicators still go out of range (F-002, open).
- Results produced before this date with the level target are invalid.

---

## D-010: Full compute budget, no reductions (spec 9.8)

**Date:** 2026-09-28
**Status:** ACCEPTED by author

**Context:**
Spec 9.8 estimates the full plan at about 4.65 million LSTM trainings and lists ways
to cut it (fewer folds, fewer seeds or tickers, fewer epochs during search).

**Decision:**
No reduction. Every method runs K = 5 folds, 30 seeds, 10 tickers, N = 20, T = 30
(620 evaluations per run) and max 50 epochs, exactly as in the proposal. The timeline
is extended instead, using the laptop GPU and Modal cloud credit ($30/month).

**Reasoning:**
Any reduction changes the research that was proposed and approved, weakens the
statistics (Wilcoxon on 30 paired seeds per ticker) or makes the search less
thorough than stated. Time is available; changing the method is not necessary.

**Rejected alternatives:**
- Fewer folds / seeds / tickers / epochs during search (spec 9.8 a-c): change the study

**Consequences:**
Experiments take months of compute. Runs must be resumable and save results as they
go, so an interrupted run never loses finished work. The evaluation cache (9.7)
matters for run time.

---

## TEMPLATE FOR NEW DECISIONS

When you make a non-obvious choice during implementation, copy and fill this:

```
## D-00X: Title

**Date:**
**Status:** ACCEPTED

**Context:**

**Decision:**

**Reasoning:**

**Rejected alternatives:**

**Consequences:**
```
