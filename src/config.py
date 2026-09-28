"""Settings shared by MOMFA and every baseline (spec 3.F.3, 9.1, 9.1.1, R6).

Keep these in one place so every method sees the same development period,
the same walk-forward folds, the same untouched final test period and the
same training protocol.
"""

# Data split (spec 9.1)
TEST_FRACTION = 0.2   # last 20% of each ticker: final test period, never seen by any optimizer
N_SPLITS = 5          # walk-forward folds inside the development period
VAL_SIZE = 126        # validation rows per fold (~6 trading months); also the
                      # early-stopping slice of the final retrained model (spec 9.1.1)

# Training protocol, identical for all methods (spec 3.F.3)
MAX_EPOCHS = 50
BATCH_SIZE = 32
PATIENCE = 10         # early stopping on validation loss

# The 10 CSE tickers (spec 3.B.1); one independent optimization per ticker
TICKERS = [
    'JKH.N0000', 'COMB.N0000', 'DIAL.N0000', 'HNB.N0000',  'LOLC.N0000',
    'SAMP.N0000', 'NTB.N0000', 'HHL.N0000',  'DIST.N0000', 'HAYL.N0000',
]

# Methods in fixed display order (spec 4.A). Colours and table rows follow this
# order, so a method keeps the same colour whichever other methods have run.
METHODS = [
    'momfa', 'random_walk', 'vanilla_lstm', 'grid_search',
    'pso_lstm', 'ga_lstm', 'gwo_lstm', 'woa_lstm', 'sequential_fa',
]
METHOD_LABELS = {
    'momfa':         'MOMFA (ours)',
    'random_walk':   'Random walk',
    'vanilla_lstm':  'Vanilla LSTM',
    'grid_search':   'Grid search LSTM',
    'pso_lstm':      'PSO-LSTM',
    'ga_lstm':       'GA-LSTM',
    'gwo_lstm':      'GWO-LSTM',
    'woa_lstm':      'WOA-LSTM',
    'sequential_fa': 'Sequential FA-LSTM',
}
