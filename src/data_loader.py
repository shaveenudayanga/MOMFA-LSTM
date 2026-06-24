import os
import pandas as pd
import pandas_ta as ta
import numpy as np


def calculate_indicators(df):
    """
    Calculates exactly 25 technical indicators matching the proposal.

    Group 1 - Momentum (7):
        RSI, MACD line, MACD Signal, Stochastic %K, Stochastic %D,
        Rate of Change, Williams %R

    Group 2 - Trend (7):
        EMA9, EMA21, EMA50, SMA20, ADX, CCI, Parabolic SAR

    Group 3 - Volatility (5):
        Bollinger Band Upper, Bollinger Band Lower, Bollinger Band Width,
        ATR, Standard Deviation

    Group 4 - Volume (6):
        OBV, MFI, VWAP (20-day rolling), Volume SMA, Volume Rate of Change,
        Chaikin Money Flow
    """
    d = df.copy()

    if not isinstance(d.index, pd.DatetimeIndex):
        if 'datetime' in d.columns:
            d['datetime'] = pd.to_datetime(d['datetime'])
            d.set_index('datetime', inplace=True)
    d.sort_index(inplace=True)

    # ── Group 1: Momentum (7) ─────────────────────────────────────────────────

    d['rsi'] = ta.rsi(d['close'], length=14)

    macd = ta.macd(d['close'], fast=12, slow=26, signal=9)
    d['macd']  = macd.iloc[:, 0]   # MACD line
    d['macds'] = macd.iloc[:, 2]   # Signal line (iloc[:,1] is histogram)

    stoch = ta.stoch(d['high'], d['low'], d['close'], k=14, d=3)
    d['stoch_k'] = stoch.iloc[:, 0]
    d['stoch_d'] = stoch.iloc[:, 1]

    d['roc']   = ta.roc(d['close'], length=10)
    d['willr'] = ta.willr(d['high'], d['low'], d['close'], length=14)

    # ── Group 2: Trend (7) ────────────────────────────────────────────────────

    d['ema9']  = ta.ema(d['close'], length=9)
    d['ema21'] = ta.ema(d['close'], length=21)
    d['ema50'] = ta.ema(d['close'], length=50)
    d['sma20'] = ta.sma(d['close'], length=20)

    adx = ta.adx(d['high'], d['low'], d['close'], length=14)
    d['adx'] = adx.iloc[:, 0]

    d['cci'] = ta.cci(d['high'], d['low'], d['close'], length=14)

    # Parabolic SAR - column name varies by pandas-ta version
    psar_df = ta.psar(d['high'], d['low'], d['close'])
    psar_cols = [c for c in psar_df.columns if c.startswith('PSAR')]
    if len(psar_cols) == 0:
        raise ValueError(
            f"PSAR column not found. Available columns: {list(psar_df.columns)}"
        )
    # Use the long signal column if available, otherwise use the first PSAR column
    psar_long_cols = [c for c in psar_cols if 'l' in c.lower() or 'long' in c.lower()]
    psar_col = psar_long_cols[0] if psar_long_cols else psar_cols[0]
    d['psar'] = psar_df[psar_col]
    # PSAR can have NaN where signal is not active - forward fill within the series
    d['psar'] = d['psar'].ffill()

    # ── Group 3: Volatility (5) ───────────────────────────────────────────────

    bb = ta.bbands(d['close'], length=20, std=2)
    # Column order: lower, mid, upper, bandwidth, percent
    d['bb_upper'] = bb.iloc[:, 2]
    d['bb_lower'] = bb.iloc[:, 0]
    d['bb_width'] = bb.iloc[:, 3]   # Bandwidth not middle band

    d['atr']    = ta.atr(d['high'], d['low'], d['close'], length=14)
    d['stddev'] = ta.stdev(d['close'], length=20)

    # ── Group 4: Volume (6) ───────────────────────────────────────────────────

    d['obv'] = ta.obv(d['close'], d['volume'])
    d['mfi'] = ta.mfi(d['high'], d['low'], d['close'], d['volume'], length=14)

    # VWAP: pandas-ta vwap() is designed for intraday tick data.
    # On daily OHLCV it returns NaN for most rows.
    # Using a 20-day rolling VWAP instead: sum(TP * Volume) / sum(Volume)
    # This is the standard daily equivalent used in financial research.
    typical_price = (d['high'] + d['low'] + d['close']) / 3
    d['vwap'] = (
        (typical_price * d['volume']).rolling(window=20).sum()
        / d['volume'].rolling(window=20).sum()
    )

    d['vol_sma'] = ta.sma(d['volume'], length=20)
    d['vol_roc'] = ta.roc(d['volume'], length=10)
    d['cmf']     = ta.cmf(d['high'], d['low'], d['close'], d['volume'], length=20)

    # ── Final 25 features in fixed proposal order ─────────────────────────────

    features = [
        # Momentum (7)
        'rsi', 'macd', 'macds', 'stoch_k', 'stoch_d', 'roc', 'willr',
        # Trend (7)
        'ema9', 'ema21', 'ema50', 'sma20', 'adx', 'cci', 'psar',
        # Volatility (5)
        'bb_upper', 'bb_lower', 'bb_width', 'atr', 'stddev',
        # Volume (6)
        'obv', 'mfi', 'vwap', 'vol_sma', 'vol_roc', 'cmf'
    ]

    assert len(features) == 25, f"Feature count error: expected 25, got {len(features)}"

    # Target: next business day closing price
    d['target'] = d['close'].shift(-1)

    result = d[features + ['target']].dropna()
    return result


def check_nan_report(df, ticker=""):
    """
    Diagnostic helper. Call this before dropna() to see which
    indicators are causing the most row loss.
    """
    d = df.copy()
    print(f"\n{'='*55}")
    print(f"NaN Report for {ticker} ({len(d)} rows before dropna)")
    print(f"{'='*55}")
    nan_counts = d.isnull().sum()
    nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)
    if nan_counts.empty:
        print("No NaN values found.")
    else:
        for col, count in nan_counts.items():
            print(f"  {col:<15} {count:>5} NaN  ({count/len(d)*100:.1f}%)")
    print(f"  Rows surviving dropna: {d.dropna().shape[0]}")
    print(f"{'='*55}\n")


def process_and_export_all(run_nan_report=False):
    """
    Loads raw OHLCV CSVs, reindexes to business days only (Mon-Fri),
    forward fills only genuine missing trading days (not weekends),
    computes 25 indicators, and saves cleaned files.

    Parameters
    ----------
    run_nan_report : bool
        If True, prints a NaN diagnostic for each stock before saving.
        Useful for debugging row loss. Set False for production runs.
    """
    data_dir   = "data"
    export_dir = os.path.join(data_dir, "preprocessed")
    os.makedirs(export_dir, exist_ok=True)

    raw_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith(".csv") and os.path.isfile(os.path.join(data_dir, f))
    ])

    if not raw_files:
        print("No CSV files found in the data/ directory.")
        return {}

    all_stocks_data = {}
    failed = []

    print(f"Found {len(raw_files)} raw CSV files.\n")

    for file in raw_files:
        ticker = file.replace("_data.csv", "")
        path   = os.path.join(data_dir, file)

        try:
            df = pd.read_csv(path)
            df.columns = [c.lower() for c in df.columns]
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime').sort_index()

            raw_rows = len(df)

            # Business days only: Mon-Fri
            # asfreq('B') inserts rows for every weekday in the date range.
            # ffill() fills only genuine missing trading days such as
            # public holidays or trading halts, never weekends.
            df = df.asfreq('B').ffill()

            if run_nan_report:
                # Run before final dropna to see which columns cause row loss
                temp = df.copy()
                # Calculate indicators without dropna for diagnostic
                check_nan_report(
                    calculate_indicators.__wrapped__(temp)
                    if hasattr(calculate_indicators, '__wrapped__')
                    else temp,
                    ticker=ticker
                )

            processed = calculate_indicators(df)

            if processed.empty:
                print(f"SKIPPED {ticker}: empty after indicator calculation")
                failed.append(ticker)
                continue

            export_path = os.path.join(export_dir, f"CLEANED_{ticker}.csv")
            processed.to_csv(export_path)

            all_stocks_data[ticker] = processed

            kept_pct = len(processed) / raw_rows * 100
            print(
                f"Done: {ticker:<20} "
                f"raw={raw_rows:>5}  "
                f"cleaned={len(processed):>5}  "
                f"kept={kept_pct:.1f}%"
            )

        except Exception as e:
            print(f"ERROR in {ticker}: {e}")
            failed.append(ticker)

    print(f"\n{'='*55}")
    print(f"Finished: {len(all_stocks_data)}/{len(raw_files)} stocks processed.")
    if failed:
        print(f"Failed:   {failed}")
    print(f"Output:   {export_dir}/")
    print(f"{'='*55}")

    return all_stocks_data


if __name__ == "__main__":
    # Set run_nan_report=True if you want to debug row loss per stock
    stocks = process_and_export_all(run_nan_report=False)