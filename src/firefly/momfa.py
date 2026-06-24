"""MOMFA - Multi-Objective Modified Firefly Algorithm for joint FS + HPO.

Solution encoding (31 dimensions):
    dims  0-24  : continuous values → sigmoid → binary feature mask (25 bits)
    dims 25-30  : continuous values in [0,1] normalised → decoded to hyperparams

Algorithm modifications over base FA (Yang 2009):
    1. Levy flight replaces Gaussian random walk (lambda=1.5, Mantegna algorithm)
    2. Adaptive alpha decay: alpha_t = alpha_0 * delta^t
    3. Sigmoid transfer for binary dimensions (dims 0-24 only)
    4. Pareto dominance + non-dominated archive instead of single global best
    5. Parallel evaluation within each iteration (D-006)
    6. Infeasible guard: all-zero mask corrected at decode time (D-007)

References: CONTEXT.md §3, DECISIONS.md D-001 to D-007.
"""

import os
import sys
import numpy as np
import torch
from joblib import Parallel, delayed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model import build_model, count_parameters
from train import train_fold, get_device
from preprocessor import make_sequences
from utils.seed import set_seed
from firefly.levy import levy_flight
from firefly.pareto import ParetoArchive


def _resolve_n_jobs(n_jobs: int) -> int:
    """Auto-select parallelism strategy based on available hardware.

    GPU available  → n_jobs=1  (sequential, each training uses CUDA fully).
                     Multiple processes sharing one GPU cause CUDA conflicts
                     and serialize anyway - sequential is correct and faster.
    CPU only       → n_jobs as given (default -1 = all cores via joblib).

    This implements D-006: parallel on CPU, sequential on GPU.
    """
    if torch.cuda.is_available():
        return 1
    return n_jobs

# ── Algorithm constants (CONTEXT.md §3.5) ────────────────────────────────────

N_FIREFLIES = 20
N_ITER      = 30
BETA_0      = 1.0
GAMMA       = 1.0
ALPHA_0     = 0.5
DELTA       = 0.97   # alpha decay factor
LAMBDA      = 1.5    # Levy stability index

# ── Hyperparameter bounds (CONTEXT.md §3.1) ──────────────────────────────────
# All 6 hyperparam dims are stored NORMALISED in [0, 1].
# HP_LO and HP_HI define the decode mapping.

HP_LO = np.array([1,      16,     0.0001, 0.10,  5,   0.0])
HP_HI = np.array([3,      256,    0.01,   0.50,  60,  1.0])

# Bin dim init range: sigmoid(-4)≈0.018, sigmoid(4)≈0.982 → good initial diversity
BIN_INIT_LO, BIN_INIT_HI = -4.0, 4.0

# Training protocol (CONTEXT.md §5)
MAX_EPOCHS = 50
BATCH_SIZE = 32
PATIENCE   = 10


# ── Encoding / decoding ───────────────────────────────────────────────────────

def _decode_hyperparams(h_norm: np.ndarray) -> dict:
    """Map normalised [0,1]^6 to actual hyperparameter values."""
    h = HP_LO + np.clip(h_norm, 0.0, 1.0) * (HP_HI - HP_LO)
    opt_val   = float(h[5])
    optimizer = 'adam' if opt_val < 1/3 else ('rmsprop' if opt_val < 2/3 else 'sgd')
    return dict(
        n_layers = int(round(float(h[0]))),
        units    = int(round(float(h[1]))),
        lr       = float(h[2]),
        dropout  = float(round(float(h[3]), 2)),
        lookback = int(round(float(h[4]))),
        optimizer= optimizer,
    )


def decode_position(x: np.ndarray, rng: np.random.Generator) -> tuple:
    """Decode a 31-dim continuous position to (mask_tuple, hyperparams_dict).

    Binary dims (0-24):
        S(x) = 1 / (1 + exp(-x))
        b = 1 if r < S(x) else 0,  r ~ Uniform[0,1]
        Infeasible guard (D-007): if all bits are 0, force one random bit to 1.

    Hyperparam dims (25-30):
        Normalised [0,1], clipped, then decoded to actual ranges.
    """
    # Binary part
    probs = 1.0 / (1.0 + np.exp(-x[:25]))
    mask  = (rng.random(25) < probs).astype(np.int8)

    # Infeasible guard - D-007
    if mask.sum() == 0:
        mask[rng.integers(25)] = 1

    return tuple(mask.tolist()), _decode_hyperparams(x[25:])


def make_cache_key(mask: tuple, hp: dict) -> tuple:
    """Deterministic cache key for a decoded phenotype (D-003).

    Precision rounding follows D-003 specification.
    """
    return (
        mask,
        hp['n_layers'],
        hp['units'],
        round(hp['lr'],       6),
        round(hp['dropout'],  2),
        hp['lookback'],
        hp['optimizer'],
    )


# ── Fitness evaluation ────────────────────────────────────────────────────────

def evaluate_phenotype(mask: tuple, hp: dict, folds: list) -> tuple:
    """Train LSTM on all walk-forward folds and return averaged fitness.

    This function is the unit of parallelism (D-006). It is stateless:
    it receives all data it needs and returns (rmse, n_params).

    Returns:
        (mean_val_rmse, n_trainable_params)
    """
    feat_idx   = [i for i, b in enumerate(mask) if b == 1]
    n_features = len(feat_idx)
    rmse_list  = []

    for fold in folds:
        X_tr, y_tr = make_sequences(
            fold['X_train'][:, feat_idx], fold['y_train'], hp['lookback']
        )
        X_va, y_va = make_sequences(
            fold['X_val'][:,   feat_idx], fold['y_val'],   hp['lookback']
        )
        model  = build_model(n_features, hp['n_layers'], hp['units'], hp['dropout'])
        result = train_fold(
            model, X_tr, y_tr, X_va, y_va,
            target_scaler  = fold['target_scaler'],
            lr             = hp['lr'],
            optimizer_name = hp['optimizer'],
            max_epochs     = MAX_EPOCHS,
            batch_size     = BATCH_SIZE,
            patience       = PATIENCE,
        )
        rmse_list.append(result['val_rmse'])

    n_params = count_parameters(
        build_model(n_features, hp['n_layers'], hp['units'], hp['dropout'])
    )
    return float(np.mean(rmse_list)), int(n_params)


def _evaluate_population(
    positions: np.ndarray,
    cache:     dict,
    folds:     list,
    rng:       np.random.Generator,
    n_jobs:    int,
) -> tuple[list, list, list]:
    """Decode + cache-check + parallel evaluate one population.

    Returns:
        decoded:  list of (mask, hp) for each firefly
        fitnesses: list of (rmse, n_params) for each firefly
        positions_used: positions array (unchanged, returned for archive storage)
    """
    n = len(positions)
    decoded = [decode_position(positions[i], rng) for i in range(n)]
    keys    = [make_cache_key(m, h) for m, h in decoded]

    # Split into cache hits and misses
    miss_idx = [i for i in range(n) if keys[i] not in cache]

    if miss_idx:
        if n_jobs == 1:
            # Sequential - each call uses GPU if available (safe, no CUDA conflicts)
            eval_results = [
                evaluate_phenotype(decoded[i][0], decoded[i][1], folds)
                for i in miss_idx
            ]
        else:
            # Parallel CPU workers - GPU not available, multiprocessing is safe
            eval_results = Parallel(n_jobs=n_jobs, prefer='processes')(
                delayed(evaluate_phenotype)(decoded[i][0], decoded[i][1], folds)
                for i in miss_idx
            )
        for i, res in zip(miss_idx, eval_results):
            cache[keys[i]] = res

    fitnesses = [cache[k] for k in keys]
    return decoded, fitnesses


# ── Main MOMFA loop ───────────────────────────────────────────────────────────

def run_momfa(
    folds:       list,
    seed:        int,
    n_fireflies: int = N_FIREFLIES,
    n_iter:      int = N_ITER,
    n_jobs:      int = -1,
) -> ParetoArchive:
    """One complete MOMFA run. Returns the final Pareto archive.

    Args:
        folds:       Output of preprocessor.get_folds() for one ticker.
        seed:        Random seed for this run (one of 30 independent runs).
        n_fireflies: Population size N (default 20).
        n_iter:      Iterations T (default 30).
        n_jobs:      Parallel workers (-1 = all cores).

    Returns:
        ParetoArchive containing all non-dominated (RMSE, param_count) solutions
        found across all iterations.
    """
    n_jobs = _resolve_n_jobs(n_jobs)
    device = get_device()
    device_label = 'GPU (CUDA)' if device.type == 'cuda' else 'CPU'
    print(f'[MOMFA] Using {device_label} | fireflies={n_fireflies}  iters={n_iter}  n_jobs={n_jobs}')

    set_seed(seed)
    rng = np.random.default_rng(seed)

    # ── Initialise population ─────────────────────────────────────────────
    positions = np.empty((n_fireflies, 31))
    positions[:, :25] = rng.uniform(BIN_INIT_LO, BIN_INIT_HI, (n_fireflies, 25))
    positions[:, 25:] = rng.uniform(0.0, 1.0, (n_fireflies, 6))

    archive = ParetoArchive()
    cache:  dict = {}
    alpha = ALPHA_0

    # ── Initial evaluation ────────────────────────────────────────────────
    decoded, fitnesses = _evaluate_population(positions, cache, folds, rng, n_jobs)

    for i, ((mask, hp), (rmse, n_params)) in enumerate(zip(decoded, fitnesses)):
        archive.update(dict(
            mask=mask, hyperparams=hp, position=positions[i].copy(),
            rmse=rmse, n_params=n_params,
        ))

    # ── Iteration loop ────────────────────────────────────────────────────
    for t in range(n_iter):
        new_positions = positions.copy()

        for i in range(n_fireflies):
            guide = archive.select_guide()
            x_j   = guide['position']          # continuous guide position (D-002)
            x_i   = positions[i]

            r_ij_sq   = float(np.sum((x_j - x_i) ** 2))
            beta      = BETA_0 * np.exp(-GAMMA * r_ij_sq)
            levy_step = levy_flight(31, LAMBDA)

            x_new = x_i + beta * (x_j - x_i) + alpha * levy_step

            # Clip hyperparam dims to [0, 1] normalised range
            x_new[25:] = np.clip(x_new[25:], 0.0, 1.0)
            # Binary dims left unconstrained - sigmoid handles saturation

            new_positions[i] = x_new

        positions = new_positions
        alpha     = ALPHA_0 * (DELTA ** (t + 1))   # adaptive decay

        # Evaluate new positions
        decoded, fitnesses = _evaluate_population(positions, cache, folds, rng, n_jobs)

        for i, ((mask, hp), (rmse, n_params)) in enumerate(zip(decoded, fitnesses)):
            archive.update(dict(
                mask=mask, hyperparams=hp, position=positions[i].copy(),
                rmse=rmse, n_params=n_params,
            ))

    return archive
