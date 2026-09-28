"""How every method scores one LSTM configuration (spec 3.F.3, 9.1, 9.1.1, R6).

MOMFA, the vanilla LSTM, grid search and the metaheuristic baselines all call
these two functions, so they share exactly the same folds, training protocol
and final test procedure.

    cross_validate()   -> validation score on the walk-forward folds (fitness f1)
    evaluate_on_test() -> one-time score of a chosen configuration on the test period
"""

import numpy as np

from config import MAX_EPOCHS, BATCH_SIZE, PATIENCE
from model import build_model
from preprocessor import make_fold_sequences, make_test_sequences, returns_to_prices
from train import train_fold, predict
from utils.metrics import compute_all

METRICS = ('rmse', 'mae', 'mape', 'da')


def _train(X_tr, y_tr, X_val, y_val, hp: dict) -> dict:
    """Build and train one LSTM with the fixed training protocol."""
    model = build_model(X_tr.shape[2], hp['n_layers'], hp['units'], hp['dropout'])
    return train_fold(
        model, X_tr, y_tr, X_val, y_val,
        lr=hp['lr'],
        optimizer_name=hp['optimizer'],
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        patience=PATIENCE,
    )


def _score(model, X: np.ndarray, data: dict, part: str) -> tuple[dict, np.ndarray, np.ndarray]:
    """Predict next-day closes for one part ('val' or 'test') and score them in LKR.

    The model outputs scaled returns; they are turned into prices from today's close
    (DECISIONS.md D-009), so every metric is on the next-day close price level.
    """
    close_today = data[f'close_{part}']
    true_prices = data[f'price_{part}']
    pred_prices = returns_to_prices(predict(model, X), data['target_scaler'], close_today)
    return compute_all(true_prices, pred_prices, close_today), pred_prices, true_prices


def cross_validate(folds: list, hp: dict, feature_idx: list | None = None) -> dict:
    """Train one model per walk-forward fold and average the validation metrics.

    Args:
        folds:       preprocessor.get_folds() on one ticker's development period.
        hp:          Hyperparameters: n_layers, units, lr, dropout, lookback, optimizer.
        feature_idx: Selected feature columns (the mask). None uses all 25.

    Returns:
        rmse, mae, mape, da   (means over folds, real price units)
        fold_rmse             (one value per fold)
        last_fold_preds, last_fold_true
    """
    per_fold = {m: [] for m in METRICS}
    for fold in folds:
        X_tr, y_tr, X_va, y_va = make_fold_sequences(fold, hp['lookback'], feature_idx)
        trained = _train(X_tr, y_tr, X_va, y_va, hp)
        metrics, preds, true = _score(trained['model'], X_va, fold, 'val')
        for m in METRICS:
            per_fold[m].append(metrics[m])

    return {
        **{m: float(np.mean(per_fold[m])) for m in METRICS},
        'fold_rmse':       per_fold['rmse'],
        'last_fold_preds': preds,
        'last_fold_true':  true,
    }


def evaluate_on_test(split: dict, hp: dict, feature_idx: list | None = None) -> dict:
    """Train the final model for one configuration and score it on the test period.

    Trains on the development period minus its last 126 days, early-stops on
    those 126 days, then scores once on the untouched test period (spec 9.1.1).

    Args:
        split:       preprocessor.get_final_split() for one ticker.
        hp:          Hyperparameters, as in cross_validate().
        feature_idx: Selected feature columns (the mask). None uses all 25.

    Returns:
        test_rmse, test_mae, test_mape, test_da (real price units),
        test_preds, test_true, best_epoch
    """
    X_tr, y_tr, X_stop, y_stop = make_fold_sequences(split, hp['lookback'], feature_idx)
    X_test, _ = make_test_sequences(split, hp['lookback'], feature_idx)

    trained = _train(X_tr, y_tr, X_stop, y_stop, hp)
    metrics, test_preds, test_true = _score(trained['model'], X_test, split, 'test')

    return {
        **{f'test_{m}': metrics[m] for m in METRICS},
        'test_preds': test_preds,
        'test_true':  test_true,
        'best_epoch': trained['best_epoch'],
    }


def summarize_runs(val_runs: dict, test_runs: dict) -> dict:
    """Per-seed metric lists plus mean (and std for RMSE), in the results JSON layout.

    Args:
        val_runs / test_runs: {'rmse': [...], 'mae': [...], 'mape': [...], 'da': [...]},
                              one value per seed.
    """
    summary = {}
    for prefix, runs in (('', val_runs), ('test_', test_runs)):
        for m in METRICS:
            summary[f'{prefix}{m}_runs'] = runs[m]
            summary[f'{prefix}{m}_mean'] = float(np.mean(runs[m]))
        summary[f'{prefix}rmse_std'] = float(np.std(runs['rmse']))
    return summary
