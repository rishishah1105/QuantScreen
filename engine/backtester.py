"""
Backtester: replays the TA entry signals on the FA-selected stocks and
measures how the trades would have performed.

Rules (see context.md):
  Signal   the full TA condition is true on day t AND was false on day t-1
           (and the indicators were already available on t-1, so the end of
           the warm-up period doesn't create fake signals)
  Entry    open of day t+1   — never the close of the signal day (look-ahead)
  Exit     close after `holding_period` trading days (entry day counts as day 1)
  Limit    one open position per stock; signals while it's open are ignored
  Costs    `cost_pct` charged on entry and again on exit (brokerage + STT + slippage)
  Sizing   equal weight: portfolio return each day = average return of the
           positions held that day; 0% on days with no positions
  Window   signals and trades only inside [start, end]; the first ~252 trading
           days are warm-up for the 200-day MA and 52-week levels

No TA filters at all  ->  "FA basket buy-and-hold": every FA-selected stock is
bought at the window's first open and held to the end.

Run from the project root for a demo (writes CSVs to backtest_output/):
    python -m engine.backtester
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from data.database import DB_PATH
from engine.indicators import get_indicator_table, load_benchmark
from engine.metrics import START_VALUE, equity_curve, summarize
from engine.screener import TAFilters, fa_filter, ta_mask

DEFAULT_HOLDING = 30
DEFAULT_COST_PCT = 0.1     # % per side; ~0.2% round trip
WARMUP_DAYS = 252
RANDOM_RUNS = 100

FA_KEYS = ("pe_max", "roe_min", "de_max", "mcap_min", "eps_min")
TA_KEYS = ("rsi_min", "rsi_max", "ma_condition", "near_high_pct", "near_low_pct", "volume_spike")

TRADE_COLUMNS = ["ticker", "signal_date", "entry_date", "entry_price", "exit_date", "exit_price",
                 "days_held", "return_pct", "closed_at_window_end"]


# ------------------------------------------------------------- signals

def required_columns(tf: TAFilters) -> list[str]:
    """Indicator columns a TA filter set depends on."""
    cols = []
    if tf.rsi_min is not None or tf.rsi_max is not None:
        cols.append("rsi")
    if tf.ma_condition is not None:
        cols += ["sma50", "sma200"]
    if tf.near_high_pct is not None:
        cols.append("high_52w")
    if tf.near_low_pct is not None:
        cols.append("low_52w")
    if tf.volume_spike:
        cols.append("vol_avg20")
    return cols


def signals(g: pd.DataFrame, tf: TAFilters) -> np.ndarray:
    """Boolean array: True on days the TA condition turns on.
    Uses ta_mask() — the exact same function as the screener."""
    cond = ta_mask(g, tf).to_numpy()
    valid = g[required_columns(tf)].notna().all(axis=1).to_numpy()
    sig = np.zeros(len(g), dtype=bool)
    sig[1:] = cond[1:] & ~cond[:-1] & valid[:-1]
    return sig


# -------------------------------------------------------------- trades

def _trade(ticker, dates, opens, closes, t, e, x, holding, cost, last):
    entry, exit_ = opens[e], closes[x]
    if not (np.isfinite(entry) and entry > 0 and np.isfinite(exit_)):
        return None
    return {
        "ticker": ticker,
        "signal_date": dates[t] if t is not None else pd.NaT,
        "entry_date": dates[e],
        "entry_price": float(entry),
        "exit_date": dates[x],
        "exit_price": float(exit_),
        "days_held": int(x - e + 1),
        "return_pct": float((exit_ * (1 - cost)) / (entry * (1 + cost)) - 1) * 100,
        "closed_at_window_end": bool(holding is not None and e + holding - 1 > last),
        "_e": e, "_x": x,
    }


def simulate_trades(ticker, g, sig, start, end, holding, cost) -> list[dict]:
    """Walk through one stock's signals and open/close positions by the rules."""
    dates = g["date"].to_numpy()
    opens, closes = g["open"].to_numpy(float), g["close"].to_numpy(float)
    upto_end = np.flatnonzero(dates <= np.datetime64(end))
    if not len(upto_end):
        return []
    last = upto_end[-1]

    trades, next_free = [], 0
    in_window = (dates >= np.datetime64(start)) & (dates <= np.datetime64(end))
    for t in np.flatnonzero(sig & in_window):
        if t < next_free:          # a position is still open
            continue
        e = t + 1                  # enter at the NEXT day's open
        if e > last:
            break
        x = min(e + holding - 1, last)
        tr = _trade(ticker, dates, opens, closes, t, e, x, holding, cost, last)
        if tr:
            trades.append(tr)
            next_free = x          # a signal on the exit day may open the next trade
    return trades


def basket_trade(ticker, g, start, end, cost) -> list[dict]:
    """FA basket mode: buy at the window's first open, hold to the end."""
    dates = g["date"].to_numpy()
    in_window = np.flatnonzero((dates >= np.datetime64(start)) & (dates <= np.datetime64(end)))
    if not len(in_window):
        return []
    e, x = in_window[0], in_window[-1]
    tr = _trade(ticker, dates, g["open"].to_numpy(float), g["close"].to_numpy(float),
                None, e, x, None, cost, x)
    return [tr] if tr else []


def position_returns(g, trades, cost) -> pd.Series:
    """Daily returns of one stock while it is held (NaN when not held).
    Entry day: close / (open x (1+cost)) - 1. Later days: close / previous close - 1.
    Exit day also pays the exit cost. Compounding these daily returns over a
    trade gives exactly the trade's return_pct."""
    opens, closes = g["open"].to_numpy(float), g["close"].to_numpy(float)
    out = np.full(len(g), np.nan)
    for tr in trades:
        e, x = tr["_e"], tr["_x"]
        r = np.empty(x - e + 1)
        r[0] = closes[e] / (opens[e] * (1 + cost)) - 1
        if x > e:
            r[1:] = closes[e + 1:x + 1] / closes[e:x] - 1
        r[-1] = (1 + r[-1]) * (1 - cost) - 1
        out[e:x + 1] = r
    return pd.Series(out, index=g["date"].to_numpy())


def portfolio_returns(per_stock: dict[str, pd.Series], calendar: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """Equal-weight portfolio: average of the held positions each day, 0 if none.
    Also returns the number of positions held each day."""
    if not per_stock:
        zeros = pd.Series(0.0, index=calendar)
        return zeros, zeros.astype(int)
    held = pd.DataFrame(per_stock).reindex(calendar)
    return held.mean(axis=1, skipna=True).fillna(0.0), held.notna().sum(axis=1)


def benchmark_returns(bench: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.Series | None:
    """Nifty 50 bought at the window's first open, held to the end — the same
    entry convention as a strategy position."""
    if bench is None or bench.empty:
        return None
    b = bench.set_index("date").reindex(calendar).ffill()
    if b["close"].isna().all():
        return None
    r = b["close"].pct_change()
    r.iloc[0] = b["close"].iloc[0] / b["open"].iloc[0] - 1
    return r.fillna(0.0)


# -------------------------------------------------------------- result

@dataclass
class BacktestResult:
    metrics: dict
    benchmark_metrics: dict
    equity: pd.DataFrame                  # date, strategy, benchmark (start = 100)
    trades: pd.DataFrame
    settings: dict
    random_total_returns: np.ndarray | None = field(default=None, repr=False)

    @property
    def random_percentile(self) -> float | None:
        """% of random-entry runs this strategy beat on total return."""
        if self.random_total_returns is None or not len(self.random_total_returns):
            return None
        return float((self.random_total_returns < self.metrics["Total return %"] / 100).mean() * 100)

    def summary(self) -> pd.DataFrame:
        rows = []
        for k, v in self.metrics.items():
            rows.append((k, v, self.benchmark_metrics.get(k, np.nan)))
        return pd.DataFrame(rows, columns=["Metric", "Strategy", "Nifty 50"]).set_index("Metric")

    def save_csv(self, folder="backtest_output") -> list[Path]:
        folder = Path(folder)
        folder.mkdir(exist_ok=True)
        paths = [folder / "trades.csv", folder / "equity.csv", folder / "summary.csv"]
        self.trades.to_csv(paths[0], index=False)
        self.equity.to_csv(paths[1], index=False)
        self.summary().to_csv(paths[2])
        return paths


# ---------------------------------------------------------------- run

def default_window(ind: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    dates = np.sort(ind["date"].unique())
    start = dates[WARMUP_DAYS] if len(dates) > WARMUP_DAYS else dates[0]
    return pd.Timestamp(start), pd.Timestamp(dates[-1])


def _run_trades(groups, tf, start, end, holding, cost, basket, sig_override=None):
    per_stock, all_trades = {}, []
    for t, g in groups.items():
        if basket:
            trades = basket_trade(t, g, start, end, cost)
        else:
            sig = sig_override[t] if sig_override is not None else signals(g, tf)
            trades = simulate_trades(t, g, sig, start, end, holding, cost)
        if trades:
            per_stock[t] = position_returns(g, trades, cost)
            all_trades += trades
    return per_stock, all_trades


def random_baseline(groups, trades_df, start, end, holding, cost, calendar, runs, seed):
    """Same stocks, same number of trades per stock, but entry dates picked at
    random. If the strategy can't beat most random runs, its TA timing adds nothing."""
    rng = np.random.default_rng(seed)
    counts = trades_df["ticker"].value_counts().to_dict()
    results = []
    for _ in range(runs):
        sig_override = {}
        for t, g in groups.items():
            sig = np.zeros(len(g), dtype=bool)
            k = counts.get(t, 0)
            if k:
                dates = g["date"].to_numpy()
                cand = np.flatnonzero((dates >= np.datetime64(start)) & (dates <= np.datetime64(end)))[:-1]
                if len(cand):
                    sig[rng.choice(cand, size=min(k, len(cand)), replace=False)] = True
            sig_override[t] = sig
        per_stock, _ = _run_trades(groups, None, start, end, holding, cost, False, sig_override)
        daily, _ = portfolio_returns(per_stock, calendar)
        results.append((1 + daily).prod() - 1)
    return np.array(results)


def run_backtest(ind: pd.DataFrame, bench: pd.DataFrame, tickers, tf: TAFilters,
                 holding_period: int = DEFAULT_HOLDING, cost_pct: float = DEFAULT_COST_PCT,
                 start=None, end=None, random_runs: int = RANDOM_RUNS, seed: int = 42) -> BacktestResult:
    """Core backtest on in-memory data (used directly by the tests)."""
    if holding_period < 1:
        raise ValueError("holding_period must be at least 1 day")
    d_start, d_end = default_window(ind)
    start = pd.Timestamp(start) if start is not None else d_start
    end = pd.Timestamp(end) if end is not None else d_end
    cost = cost_pct / 100
    basket = tf.is_empty()

    sub = ind[ind["ticker"].isin(set(tickers))]
    groups = {t: g.sort_values("date").reset_index(drop=True) for t, g in sub.groupby("ticker")}
    all_dates = pd.DatetimeIndex(np.sort(ind["date"].unique()))
    calendar = all_dates[(all_dates >= start) & (all_dates <= end)]

    per_stock, trades = _run_trades(groups, tf, start, end, holding_period, cost, basket)
    daily, n_held = portfolio_returns(per_stock, calendar)
    trades_df = (pd.DataFrame(trades).drop(columns=["_e", "_x"]) if trades
                 else pd.DataFrame(columns=TRADE_COLUMNS))
    trades_df = trades_df.sort_values(["entry_date", "ticker"]).reset_index(drop=True)

    metrics = summarize(daily, trades_df["return_pct"].to_numpy() / 100)
    metrics["Exposure %"] = float((n_held > 0).mean() * 100) if len(n_held) else 0.0
    metrics["Avg positions held"] = float(n_held[n_held > 0].mean()) if (n_held > 0).any() else 0.0

    b_daily = benchmark_returns(bench, calendar)
    bench_metrics = summarize(b_daily) if b_daily is not None else {}

    equity = pd.DataFrame({
        "date": calendar,
        "strategy": equity_curve(daily).to_numpy(),
        "benchmark": equity_curve(b_daily).to_numpy() if b_daily is not None else np.nan,
    })

    rand = None
    if not basket and len(trades_df) and random_runs > 0:
        rand = random_baseline(groups, trades_df, start, end, holding_period, cost, calendar, random_runs, seed)

    settings = {
        "mode": "FA basket buy-and-hold" if basket else "TA entry signals",
        "tickers": len(groups), "ta_filters": tf, "holding_period": holding_period,
        "cost_pct_per_side": cost_pct, "start": start.date(), "end": end.date(),
        "trading_days": len(calendar),
    }
    return BacktestResult(metrics, bench_metrics, equity, trades_df, settings, rand)


def backtest(tickers, ta_filters=None, holding_period: int = DEFAULT_HOLDING,
             cost_pct: float = DEFAULT_COST_PCT, start=None, end=None,
             random_runs: int = RANDOM_RUNS, seed: int = 42, db_path=DB_PATH) -> BacktestResult:
    """Backtest TA entry signals on a given list of tickers, using stocks.db."""
    tf = ta_filters if isinstance(ta_filters, TAFilters) else TAFilters(**(ta_filters or {}))
    ind = get_indicator_table(str(db_path))
    return run_backtest(ind, load_benchmark(db_path), tickers, tf, holding_period, cost_pct,
                        start, end, random_runs, seed)


def backtest_strategy(holding_period: int = DEFAULT_HOLDING, cost_pct: float = DEFAULT_COST_PCT,
                      start=None, end=None, random_runs: int = RANDOM_RUNS, seed: int = 42,
                      db_path=DB_PATH, **filters) -> BacktestResult:
    """Same keyword filters as screen(): FA filters pick the stocks (today's
    snapshot), TA filters generate the historical entry signals."""
    unknown = set(filters) - set(FA_KEYS) - set(TA_KEYS)
    if unknown:
        raise TypeError(f"Unknown filter(s): {sorted(unknown)}")
    fa = fa_filter(**{k: filters[k] for k in FA_KEYS if k in filters}, db_path=db_path)
    tf = TAFilters(**{k: filters[k] for k in TA_KEYS if k in filters})
    res = backtest(list(fa["ticker"]), tf, holding_period, cost_pct, start, end, random_runs, seed, db_path)
    res.settings["fa_filters"] = {k: filters[k] for k in FA_KEYS if k in filters}
    return res


# ---------------------------------------------------------------- demo

def _print(res: BacktestResult, title: str):
    s = res.settings
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
    print(f"Mode: {s['mode']} | stocks: {s['tickers']} | window: {s['start']} -> {s['end']} "
          f"({s['trading_days']} days) | hold: "
          f"{'to end' if s['mode'].startswith('FA basket') else str(s['holding_period']) + 'd'} | "
          f"cost: {s['cost_pct_per_side']}%/side")
    print(res.summary().round(2).to_string())
    if res.random_percentile is not None:
        r = res.random_total_returns * 100
        print(f"\nRandom-entry baseline ({len(r)} runs): median {np.median(r):.1f}%, "
              f"range {r.min():.1f}% to {r.max():.1f}%")
        print(f"Strategy beat {res.random_percentile:.0f}% of random runs")


if __name__ == "__main__":
    pd.set_option("display.width", 160)

    res = backtest_strategy(pe_max=20, roe_min=15, rsi_max=40)
    _print(res, "1) Oversold quality: P/E <= 20, ROE >= 15, RSI <= 40, hold 30 days")
    print("\nFirst 5 trades (check these by hand against stocks.db):")
    print(res.trades.head().to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    paths = res.save_csv()
    print(f"\nSaved {', '.join(str(p) for p in paths)}")

    res2 = backtest_strategy(roe_min=15, ma_condition="above", rsi_max=45)
    _print(res2, "2) Pullback in uptrend: ROE >= 15, 50MA > 200MA, RSI <= 45, hold 30 days")

    res3 = backtest_strategy(pe_max=20, roe_min=15)
    _print(res3, "3) FA basket buy-and-hold: P/E <= 20, ROE >= 15 (no TA filters)")
