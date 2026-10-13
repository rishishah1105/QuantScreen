"""
Technical indicators, computed per stock from the prices table.

Every indicator on day t uses ONLY prices from day t and earlier
(rolling windows and exponential smoothing look backwards), so these
columns are safe to use for backtest signals — no look-ahead bias.

Columns added:
    rsi                  RSI(14), Wilder's smoothing
    sma50, sma200        simple moving averages of close
    high_52w, low_52w    highest high / lowest low over the last 252 trading days
    vol_avg20            average volume of the PREVIOUS 20 days (excludes today)
    golden_cross_recent  50-SMA crossed above 200-SMA within the last 10 trading days

An indicator is NaN until there is enough history (e.g. sma200 needs 200 days).
NaN never passes a filter.
"""

from functools import lru_cache

import pandas as pd

from data.corporate_actions import apply_corporate_actions
from data.database import DB_PATH, get_connection
from data.tickers import BENCHMARK

RSI_WINDOW = 14
SMA_FAST = 50
SMA_SLOW = 200
WINDOW_52W = 252        # ~trading days in a year
VOL_WINDOW = 20
CROSS_LOOKBACK = 10     # "recent" golden cross = within this many trading days


def rsi(close: pd.Series, window: int = RSI_WINDOW) -> pd.Series:
    """Wilder's RSI — the standard definition used by TradingView and most platforms.

    1. Daily change = close - previous close
    2. Split into gains (positive changes) and losses (negative changes, as positives)
    3. Smooth each with Wilder's moving average (an EMA with alpha = 1/window)
    4. RS = avg gain / avg loss;  RSI = 100 - 100 / (1 + RS)
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss
    out = 100 - 100 / (1 + rs)
    # No losses at all in the window -> RSI is 100 by definition (rs = inf gives that already;
    # 0/0 when price is completely flat -> leave as 50, neutral)
    out[(avg_gain == 0) & (avg_loss == 0)] = 50.0
    return out


def add_indicators(g: pd.DataFrame) -> pd.DataFrame:
    """Add indicator columns to ONE stock's price history."""
    g = g.sort_values("date").copy()
    close = g["close"]

    g["rsi"] = rsi(close)
    g["sma50"] = close.rolling(SMA_FAST, min_periods=SMA_FAST).mean()
    g["sma200"] = close.rolling(SMA_SLOW, min_periods=SMA_SLOW).mean()
    g["high_52w"] = g["high"].rolling(WINDOW_52W, min_periods=WINDOW_52W).max()
    g["low_52w"] = g["low"].rolling(WINDOW_52W, min_periods=WINDOW_52W).min()
    # shift(1): compare today's volume against the 20 days BEFORE today
    g["vol_avg20"] = g["volume"].shift(1).rolling(VOL_WINDOW, min_periods=VOL_WINDOW).mean()

    both_valid = g["sma50"].notna() & g["sma200"].notna()
    above = (g["sma50"] > g["sma200"]) & both_valid
    crossed_today = above & ~above.shift(1, fill_value=False) & both_valid.shift(1, fill_value=False)
    g["golden_cross_recent"] = (
        crossed_today.astype(int).rolling(CROSS_LOOKBACK, min_periods=1).max().astype(bool)
    )
    return g


def load_prices(db_path=DB_PATH, include_benchmark: bool = False,
                apply_corrections: bool = True) -> pd.DataFrame:
    """Price table from stocks.db, with corporate-action corrections applied
    (see data/corporate_actions.py) unless apply_corrections=False."""
    with get_connection(db_path) as conn:
        df = pd.read_sql(
            "SELECT date, ticker, open, high, low, close, volume FROM prices ORDER BY ticker, date",
            conn, parse_dates=["date"],
        )
    if not include_benchmark:
        df = df[df["ticker"] != BENCHMARK]
    if apply_corrections:
        df = apply_corporate_actions(df)
    return df.reset_index(drop=True)


def load_benchmark(db_path=DB_PATH) -> pd.DataFrame:
    """Nifty 50 index (^NSEI) prices."""
    with get_connection(db_path) as conn:
        return pd.read_sql(
            "SELECT date, open, close FROM prices WHERE ticker = ? ORDER BY date",
            conn, params=[BENCHMARK], parse_dates=["date"],
        )


def compute_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    """Indicators for every stock in a long (date, ticker, ...) price table."""
    if prices.empty:
        return prices
    parts = [add_indicators(g) for _, g in prices.groupby("ticker", sort=True)]
    return pd.concat(parts, ignore_index=True)


@lru_cache(maxsize=4)
def get_indicator_table(db_path: str = str(DB_PATH)) -> pd.DataFrame:
    """All stocks + indicators, computed once per process and cached.
    Callers must not modify the returned frame (copy it first)."""
    return compute_indicators(load_prices(db_path))
