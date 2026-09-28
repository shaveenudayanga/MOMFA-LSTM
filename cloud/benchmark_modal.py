"""Measure how long one candidate evaluation takes on Modal (spec 9.8 d).

A candidate evaluation = one random configuration from the MOMFA search space
(random mask + 6 hyperparameters, decoded as in spec 9.6) cross-validated on
the 5 walk-forward folds of one ticker. The same candidates run on CPU and GPU
containers so the two can be compared directly.

Run from the repo root:
    venv/bin/modal run cloud/benchmark_modal.py
"""

import os
import sys
import time

import modal

REMOTE_SRC = '/root/src'
REMOTE_DATA = '/root/data/preprocessed'
TICKER = 'JKH.N0000'

CPU_CONTAINERS = 8
GPU_CONTAINERS = 2
CANDIDATES_PER_CONTAINER = 5

python_libs = ['numpy', 'pandas', 'scikit-learn', 'joblib']

cpu_image = (
    modal.Image.debian_slim(python_version='3.12')
    .pip_install(*python_libs)
    .pip_install('torch', index_url='https://download.pytorch.org/whl/cpu')
    .add_local_dir('src', REMOTE_SRC)
    .add_local_dir('data/preprocessed', REMOTE_DATA)
)
gpu_image = (
    modal.Image.debian_slim(python_version='3.12')
    .pip_install(*python_libs, 'torch')
    .add_local_dir('src', REMOTE_SRC)
    .add_local_dir('data/preprocessed', REMOTE_DATA)
)

app = modal.App('momfa-benchmark')


def _time_candidates(seed: int, n_candidates: int) -> list[dict]:
    """Cross-validate n random search-space candidates and time each one."""
    sys.path.insert(0, REMOTE_SRC)
    import numpy as np
    import torch
    from evaluation import cross_validate
    from firefly.momfa import _decode_hyperparams, BIN_INIT_LO, BIN_INIT_HI
    from preprocessor import load_ticker, split_dev_test, get_folds
    from utils.seed import set_seed

    if not torch.cuda.is_available():
        torch.set_num_threads(1)   # one core per container; avoid oversubscription

    dev_df, _ = split_dev_test(load_ticker(TICKER, REMOTE_DATA))
    folds = get_folds(dev_df)
    rng = np.random.default_rng(seed)

    timings = []
    for i in range(n_candidates):
        probs = 1 / (1 + np.exp(-rng.uniform(BIN_INIT_LO, BIN_INIT_HI, 25)))
        mask = rng.random(25) < probs
        if not mask.any():
            mask[rng.integers(25)] = True
        hp = _decode_hyperparams(rng.random(6))

        set_seed(seed * 1000 + i)
        start = time.perf_counter()
        rmse = cross_validate(folds, hp, list(np.flatnonzero(mask)))['rmse']
        timings.append({
            'seconds': time.perf_counter() - start,
            'n_features': int(mask.sum()),
            'rmse': rmse,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            **hp,
        })
    return timings


@app.function(image=cpu_image, cpu=1.0, memory=2048, timeout=3600)
def time_on_cpu(seed: int) -> list[dict]:
    return _time_candidates(seed, CANDIDATES_PER_CONTAINER)


@app.function(image=gpu_image, gpu='T4', timeout=3600)
def time_on_gpu(seed: int) -> list[dict]:
    return _time_candidates(seed, CANDIDATES_PER_CONTAINER)


@app.local_entrypoint()
def main():
    cpu = [t for batch in time_on_cpu.map(range(CPU_CONTAINERS)) for t in batch]
    gpu = [t for batch in time_on_gpu.map(range(GPU_CONTAINERS)) for t in batch]

    for label, rows in (('CPU (1 core)', cpu), ('GPU (T4)', gpu)):
        secs = [r['seconds'] for r in rows]
        print(f'\n{label}: {len(rows)} candidate evaluations (5 folds each)')
        print(f'  mean {sum(secs) / len(secs):.1f}s   min {min(secs):.1f}s   max {max(secs):.1f}s')
        for r in rows:
            print(f"  {r['seconds']:6.1f}s  layers {r['n_layers']}  units {r['units']:3d}  "
                  f"lookback {r['lookback']:2d}  feats {r['n_features']:2d}  {r['optimizer']}")

    # Same seeds on both devices -> same candidates, compare one to one
    paired = list(zip(cpu[:len(gpu)], gpu))
    speedup = sum(c['seconds'] for c, _ in paired) / sum(g['seconds'] for _, g in paired)
    print(f'\nGPU is {speedup:.2f}x faster than one CPU core on the same {len(paired)} candidates')
