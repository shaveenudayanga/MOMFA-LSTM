# CONTEXT.md
# MOMFA-LSTM: Complete Project Context

> This file is the single source of truth for any Claude session working on this project.
> Read this before touching any code. Do not paraphrase the proposal - this IS the spec.
> Last updated: 2026-06-16

---

## 1. What This Project Is

A research implementation for a final-year thesis at the University of Sri Jayewardenepura,
Department of Computer Science, supervised by Prof. T. G. I. Fernando.

**The core idea:** LSTM stock price forecasting performance depends on two connected decisions -
which technical indicators to use as input features, and how to configure the network. Current
research solves these separately. This project solves them jointly using a modified multi-objective
Firefly Algorithm, producing a Pareto front of accuracy vs. complexity trade-offs.

**Target market:** Colombo Stock Exchange (CSE), Sri Lanka. A frontier market, illiquid, sensitive
to local events, and almost completely unstudied in the deep learning forecasting literature.

**Novelty claim (verified against literature, June 2026):**
- No prior work does joint FS + HPO for LSTM stock forecasting (Gap 1)
- Multi-objective formulations in this domain are rare and incomplete (Gap 2)
- No modern metaheuristic work targets the CSE specifically (Gap 3)

---

## 2. Repository

- URL: https://github.com/shaveenudayanga/MOMFA-LSTM
- Branch: main
- Structure:
  - `src/` - implementation code
  - `requirements.txt` - dependencies
  - `README.md` - public-facing description
  - DOI: 10.5281/zenodo.20132641

---

## 3. Algorithm: MOMFA (Multi-Objective Modified Firefly Algorithm)

### 3.1 Solution Encoding

Each firefly encodes a complete candidate solution as a single vector:

```
x = [ b1, b2, ..., b25 | h1, h2, h3, h4, h5, h6 ]
      Binary mask (25)    LSTM config (6)
```

**Part 1 - Binary feature mask (dimensions 1-25):**
- Each bit = one technical indicator (1 = selected, 0 = excluded)
- The algorithm can select any combination from 1 to all 25

**Part 2 - LSTM hyperparameter vector (dimensions 26-31):**

| # | Hyperparameter | Range |
|---|---|---|
| h1 | Number of layers | 1 to 3 |
| h2 | Units per layer | 16 to 256 |
| h3 | Learning rate | 0.0001 to 0.01 |
| h4 | Dropout rate | 0.10 to 0.50 |
| h5 | Lookback window | 5 to 60 days |
| h6 | Optimizer | Adam / RMSprop / SGD (mapped from continuous) |

Total dimensions: **31**

### 3.2 Two Optimization Objectives

- **f1 (minimize):** RMSE of next-day closing price prediction on validation fold
- **f2 (minimize):** Total number of trainable parameters in the LSTM network

Note: f2 is raw parameter count only (NOT features × params - that formulation was
rejected. See DECISIONS.md entry D-001.)

### 3.3 Algorithm Modifications

**Base:** Firefly Algorithm (Yang, 2009)

Standard update rule:
```
x_i^{t+1} = x_i^t + beta_0 * exp(-gamma * r_ij^2) * (x_j^t - x_i^t) + alpha * epsilon^t
```

**Modification 1 - Levy flight replaces Gaussian random walk:**
```
alpha * epsilon^t  →  alpha * L(lambda),   lambda = 1.5
```
Why: Gaussian steps converge too early in 31-D mixed search space. Levy heavy tail
allows occasional long jumps to escape local optima.

**Adaptive alpha decay:**
```
alpha^t = alpha_0 * delta^t,   alpha_0 = 0.5,   delta = 0.97
```

**Modification 2 - Sigmoid transfer for binary dimensions:**
Applied only to dimensions 1-25 after each position update:
```
S(x_i^d) = 1 / (1 + exp(-x_i^d))
b_i^d = 1 if r < S(x_i^d), else 0     (r ~ Uniform[0,1])
```
Dimensions 26-31 use standard update, clipped to search ranges.

### 3.4 Multi-Objective Handling

- Pareto dominance instead of single fitness
- Non-dominated archive maintained across all iterations
- Fireflies move toward a randomly selected archive member (not single global best)
- Archive is updated every iteration
- **Archive stores phenotypes (fixed binary mask + fitness), NOT genotypes**
  This is critical for cache coherence. See DECISIONS.md D-002.

### 3.5 Algorithm Parameters

| Parameter | Value |
|---|---|
| Population size N | 20 |
| Iterations T | 30 |
| beta_0 | 1.0 (standard) |
| gamma | 1.0 (standard) |
| alpha_0 | 0.5 |
| delta | 0.97 |
| lambda (Levy) | 1.5 |

### 3.6 Full Algorithm Steps

1. Initialize N=20 fireflies with random positions in joint solution space
2. Evaluate each firefly: decode vector, train LSTM, compute RMSE and param count
3. Update non-dominated archive with any non-dominated solutions
4. For each firefly i:
   a. Select guide j randomly from archive
   b. Update position using modified rule with Levy flight
   c. Apply sigmoid transfer to binary dimensions 1-25
   d. Clip real-valued dimensions 26-31 to search ranges
5. Update alpha using adaptive decay rule
6. Repeat steps 2-5 for T=30 iterations
7. Return final Pareto front from archive

---

## 4. Data

### 4.1 Source
- TradingView via tvDatafeed Python library
- Daily OHLCV (Open, High, Low, Close, Volume)
- Period: 5 years
- 10 liquid CSE stocks including: JKH, Commercial Bank, Dialog Axiata, HNB, LOLC, Sampath Bank

### 4.2 Modeling unit
- One separate LSTM model per stock ticker
- No pooled or global model

### 4.3 Target variable
- Next-day closing price level (regression, not returns)
- Normalized to [0,1] via Min-Max scaling

### 4.4 Preprocessing
1. Forward fill missing values (public holidays, trading suspensions)
2. Min-Max normalization - fit on training window ONLY at each fold
3. Same scaler applied to validation and test sets separately
4. All metrics computed on inverse-transformed values

### 4.5 Technical Indicators (25 total)

**Group 1: Momentum (7)**
RSI, MACD, MACD Signal, Stochastic %K, Stochastic %D, Rate of Change, Williams %R

**Group 2: Trend (7)**
EMA9, EMA21, EMA50, SMA20, ADX, CCI, Parabolic SAR

**Group 3: Volatility (5)**
Bollinger Upper, Bollinger Lower, Bollinger Width, ATR, Standard Deviation

**Group 4: Volume (6)**
OBV, MFI, VWAP, Volume SMA, Volume Rate of Change, Chaikin Money Flow

Library: `pandas-ta`

Status: **COMPLETE** (features calculated, verified)

### 4.6 Validation Strategy
Walk-forward validation with expanding training windows.
- Train on past, test on future only - no data leakage
- Final metrics = average across all folds

---

## 5. LSTM Training Protocol (fixed across all methods)

| Setting | Value |
|---|---|
| Max epochs | 50 |
| Batch size | 32 |
| Early stopping patience | 10 epochs (on validation RMSE) |
| Loss function | MSE |
| Fitness caching | Yes - cache on real-valued position vector, return stored phenotype |

---

## 6. Baselines (8 total)

| # | Method | Type | FS | HPO |
|---|---|---|---|---|
| 1 | Vanilla LSTM | Manual | No | No |
| 2 | Grid Search LSTM | Exhaustive | No | Yes |
| 3 | PSO-LSTM | Metaheuristic | No | Yes |
| 4 | GA-LSTM | Metaheuristic | No | Yes |
| 5 | GWO-LSTM | Metaheuristic | No | Yes |
| 6 | WOA-LSTM | Metaheuristic | No | Yes |
| 7 | Sequential FA-LSTM | Metaheuristic | Yes, then | Yes |
| 8 | Random Walk Persistence | Naive | - | - |

**Baseline 7 is critical** - it tests joint vs. sequential (the actual claim in Fig. 1).
Baselines 1-6 test joint vs. HPO-only.
Baseline 8 is the sanity floor.

All baselines use the same fixed training protocol (Section 5 above).

---

## 7. Evaluation

### 7.1 Metrics

| Metric | Formula | Notes |
|---|---|---|
| RMSE | sqrt(mean((y_hat - y)^2)) | Primary metric for Wilcoxon test |
| MAE | mean(abs(y_hat - y)) | |
| MAPE | mean(abs((y_hat - y)/y)) * 100 | Allows cross-stock comparison |
| Directional Accuracy (DA) | sign(y_hat_t - y_{t-1}) == sign(y_t - y_{t-1}) | Computed on inverse-transformed values |
| Hypervolume | Volume dominated by Pareto front vs. reference point | MOMFA only |

Hypervolume reference point: worst observed RMSE and worst observed parameter count across all runs.

### 7.2 Statistical Testing
- 30 independent runs per method, different random seeds
- Wilcoxon signed-rank test on RMSE values (non-parametric, appropriate for stochastic algorithms)
- Significance threshold: p < 0.05

---

## 8. Open Theoretical Questions

These are unresolved as of the start of implementation. See FINDINGS.md for answers
as they are discovered empirically.

**OQ-001:** Does Levy flight on binary dimensions (pre-sigmoid) cause saturation?
With large Levy step values, sigmoid collapses to near-deterministic 0 or 1,
killing feature selection diversity. Unknown whether this is a real problem in practice.

**OQ-002:** Does alpha decay actually do meaningful work in T=30 iterations?
alpha_30 = 0.5 * 0.97^30 ≈ 0.20. The range is 0.5 to 0.20, which is limited.
May need steeper decay or more iterations in practice.

**OQ-003:** Is N=20, T=30 sufficient for 31-D multi-objective search?
Population size is small for the dimensionality. No convergence study done yet.
Monitor hypervolume per iteration during experiments.

**OQ-004:** Interaction between random archive guide selection and distance-based
attraction. When a firefly moves toward a randomly selected archive member,
the attractiveness term beta_0 * exp(-gamma * r_ij^2) is computed against
that random guide. All archive members are non-dominated, so brightness
comparison is moot. The distance term still creates spatial pull toward the
guide. This is an implicit diversity mechanism but is not formally justified.

---

## 9. Key References

- [1] Samarawickrama & Fernando, ICIIS 2017 - CSE baseline (direct predecessor)
- [5] Das et al., Expert Systems X 2019 - FA for FS in stock forecasting (closest prior work)
- [11] Yang, SAGA 2009 - Original Firefly Algorithm
- [14] Ariyaratne, Fernando & Weerakoon, SEC 2019 - ModFA (local precedent for modifications)
- [4] Zeng et al., PLoS ONE 2025 - PSO-LSTM (baseline 3 reference)
- [18] Holland, MIT Press 1992 - GA (baseline 4 reference)
- [19] Mirjalili et al., AES 2014 - GWO (baseline 5 reference)
- [20] Mirjalili & Lewis, AES 2016 - WOA (baseline 6 reference)

---

## 10. What NOT to Change Without Discussion

These were deliberate decisions after audit and Prof. Fernando review:

- **Complexity metric = raw param count** (not features x params - see D-001)
- **8 baselines, not 7** - the sequential FA-LSTM is load-bearing for the thesis claim
- **Walk-forward validation** - not k-fold, not train-test split
- **30 seeds** - do not reduce for speed; this is the statistical validity claim
- **Per-ticker models** - not pooled
- **Fitness caching keyed on real-valued vector** - not on binary mask (see D-002)
