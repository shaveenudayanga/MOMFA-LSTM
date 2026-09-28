"""Leakage and alignment tests for the data split (spec 9.1, 9.2, 9.3, 12.1 S7).

Run from the repo root:
    venv/bin/python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))

from config import TEST_FRACTION, N_SPLITS, VAL_SIZE
from data_loader import load_raw_ohlcv, calculate_indicators
from preprocessor import (
    FEATURE_COLS, TARGET_COL,
    load_ticker, split_dev_test, fold_boundaries, get_folds, get_final_split,
    make_sequences, make_fold_sequences, make_test_sequences,
    next_day_return, returns_to_prices,
)

TICKER = 'JKH.N0000'
RAW_CSV = os.path.join(REPO_ROOT, 'data', f'{TICKER}_data.csv')
CLEANED_DIR = os.path.join(REPO_ROOT, 'data', 'preprocessed')


class TestTarget(unittest.TestCase):

    def test_target_is_next_day_close(self):
        raw = load_raw_ohlcv(RAW_CSV)
        cleaned = load_ticker(TICKER, CLEANED_DIR)
        next_close = raw['close'].shift(-1).loc[cleaned.index]
        np.testing.assert_allclose(cleaned[TARGET_COL].values, next_close.values)

    def test_indicators_use_past_rows_only(self):
        # Cutting off the future must not change any indicator value on earlier dates.
        raw = load_raw_ohlcv(RAW_CSV)
        full = calculate_indicators(raw)
        truncated = calculate_indicators(raw.iloc[:900])
        common = truncated.index
        np.testing.assert_allclose(
            truncated[FEATURE_COLS].values, full.loc[common, FEATURE_COLS].values
        )


class TestReturnTarget(unittest.TestCase):
    """The LSTM predicts the next-day return; prices are rebuilt from it (D-009)."""

    @classmethod
    def setUpClass(cls):
        dev, _ = split_dev_test(load_ticker(TICKER, CLEANED_DIR))
        cls.folds = get_folds(dev)

    def test_true_returns_rebuild_true_prices(self):
        for fold in self.folds:
            for part in ('train', 'val'):
                rebuilt = returns_to_prices(fold[f'y_{part}'], fold['target_scaler'], fold[f'close_{part}'])
                np.testing.assert_allclose(rebuilt, fold[f'price_{part}'])

    def test_return_scaler_fit_on_training_rows_only(self):
        for fold in self.folds:
            train_returns = fold['price_train'] / fold['close_train'] - 1
            self.assertAlmostEqual(fold['target_scaler'].data_max_[0], train_returns.max())
            self.assertAlmostEqual(fold['target_scaler'].data_min_[0], train_returns.min())


class TestDevTestSplit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = load_ticker(TICKER, CLEANED_DIR)
        cls.dev, cls.test = split_dev_test(cls.df)

    def test_test_period_is_last_fraction(self):
        self.assertEqual(len(self.test), round(len(self.df) * TEST_FRACTION))
        self.assertEqual(len(self.dev) + len(self.test), len(self.df))
        self.assertTrue(self.test.index.equals(self.df.index[-len(self.test):]))

    def test_dev_ends_before_test_starts(self):
        self.assertLess(self.dev.index[-1], self.test.index[0])

    def test_folds_never_reach_test_period(self):
        last_val_end = fold_boundaries(len(self.dev))[-1][1]
        self.assertEqual(last_val_end, len(self.dev))


class TestFolds(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dev, _ = split_dev_test(load_ticker(TICKER, CLEANED_DIR))
        cls.dev_features = dev[FEATURE_COLS].values
        cls.boundaries = fold_boundaries(len(dev))
        cls.folds = get_folds(dev)

    def test_fold_count_and_validation_size(self):
        self.assertEqual(len(self.folds), N_SPLITS)
        for fold in self.folds:
            self.assertEqual(len(fold['y_val']), VAL_SIZE)

    def test_scaler_fit_on_training_rows_only(self):
        for fold, (train_end, _) in zip(self.folds, self.boundaries):
            train_rows = self.dev_features[:train_end]
            np.testing.assert_allclose(fold['feature_scaler'].data_max_, train_rows.max(axis=0))
            np.testing.assert_allclose(fold['feature_scaler'].data_min_, train_rows.min(axis=0))

    def test_validation_follows_training_without_gap(self):
        for (train_end, val_end), fold in zip(self.boundaries, self.folds):
            self.assertEqual(len(fold['X_train']), train_end)
            self.assertEqual(len(fold['X_val']), val_end - train_end)


class TestFinalSplit(unittest.TestCase):
    """Final retrained model: train, early-stopping slice, test (spec 9.1.1)."""

    @classmethod
    def setUpClass(cls):
        cls.df = load_ticker(TICKER, CLEANED_DIR)
        cls.dev, cls.test = split_dev_test(cls.df)
        cls.split = get_final_split(cls.df)

    def test_stopping_slice_is_last_126_development_days(self):
        self.assertEqual(self.split['stop_size'], VAL_SIZE)
        self.assertEqual(len(self.split['X_val']), VAL_SIZE)
        self.assertEqual(len(self.split['X_train']) + VAL_SIZE, len(self.dev))

    def test_test_period_is_untouched_test_rows(self):
        self.assertEqual(len(self.split['y_test']), len(self.test))
        self.assertTrue(self.split['test_dates'].equals(self.test.index))

    def test_scaler_fit_on_training_rows_only(self):
        train_rows = self.dev[FEATURE_COLS].values[:-VAL_SIZE]
        np.testing.assert_allclose(self.split['feature_scaler'].data_max_, train_rows.max(axis=0))
        train_returns = next_day_return(self.dev.iloc[:-VAL_SIZE])
        np.testing.assert_allclose(self.split['target_scaler'].data_max_, train_returns.max(axis=0))


class TestSequences(unittest.TestCase):

    def test_window_predicts_day_after_its_last_row(self):
        # Row t holds feature value t and target t+1 (the next day's close).
        n_rows, lookback = 50, 7
        X = np.arange(n_rows, dtype=float).reshape(-1, 1)
        y = np.arange(n_rows, dtype=float) + 1
        X_seq, y_seq = make_sequences(X, y, lookback)
        last_row_in_window = X_seq[:, -1, 0]
        np.testing.assert_array_equal(y_seq, last_row_in_window + 1)
        self.assertEqual(len(X_seq), n_rows - lookback + 1)

    def test_every_lookback_scores_the_same_validation_dates(self):
        dev, _ = split_dev_test(load_ticker(TICKER, CLEANED_DIR))
        fold = get_folds(dev)[0]
        for lookback in (1, 5, 20, 60):
            _, _, X_va, y_va = make_fold_sequences(fold, lookback)
            np.testing.assert_array_equal(y_va, fold['y_val'])
            np.testing.assert_array_equal(X_va[:, -1, :], fold['X_val'])

    def test_every_lookback_scores_every_test_date(self):
        split = get_final_split(load_ticker(TICKER, CLEANED_DIR))
        for lookback in (1, 5, 60):
            X_te, y_te = make_test_sequences(split, lookback)
            np.testing.assert_array_equal(y_te, split['y_test'])
            np.testing.assert_array_equal(X_te[:, -1, :], split['X_test'])

    def test_feature_mask_selects_columns(self):
        dev, _ = split_dev_test(load_ticker(TICKER, CLEANED_DIR))
        fold = get_folds(dev)[0]
        X_tr, _, X_va, _ = make_fold_sequences(fold, lookback=10, feature_idx=[0, 7, 24])
        self.assertEqual(X_tr.shape[2], 3)
        np.testing.assert_array_equal(X_va[:, -1, :], fold['X_val'][:, [0, 7, 24]])


if __name__ == '__main__':
    unittest.main()
