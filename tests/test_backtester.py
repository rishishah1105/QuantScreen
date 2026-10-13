"""
Backtester, metrics and corporate-action tests. Run from the project root:  pytest -v

Strategy: build small price series where the right answer is known in advance
(steady rise, flat price, a volume spike on a chosen day...) and check the
backtester gets exactly that answer.
"""

import math

import numpy as np
import pandas as pd
import pytest

from data.corporate_actions import apply_corporate_actions, find_jumps
from engine import metrics
from engine.backtester import run_backtest, signals
from engine.indicators import compute_indicators
from engine.screener import TAFilters

N = 500
DATES = pd.bdate_range("2023-01-02", periods=N)
START = DATES[252]                  # default window start (after warm-up)


def make_prices(ticker, close, open_=None, volume=None, dates=DATES):
    close = np.asarray(close, dtype=float)
    if open_ is None:                               # open = previous close (no overnight gaps)
        open_ = np.concatenate([[close[0]], close[:-1]])
    vol = np.full(len(close), 1_000_000.0) if volume is None else np.asarray(volume, float)
    return pd.DataFrame({"date": dates[:len(close)], "ticker": ticker, "open": open_,
                         "high": np.maximum(open_, close) * 1.001, "low": np.minimum(open_, close) * 0.999,
                         "close": close, "volume": vol})


def bench_from(prices):
    return prices[["date", "open", "close"]].copy()


def run(prices_list, tf, bench=None, **kw):
    prices = pd.concat(prices_list, ignore_index=True)
    ind = compute_indicators(prices)
    kw.setdefault("random_runs", 0)
    kw.setdefault("cost_pct", 0.0)
    return run_backtest(ind, bench if bench is not None else bench_from(prices_list[0]),
                        prices["ticker"].unique(), tf, **kw)


def wavy(n=N, seed=1):
    rng = np.random.default_rng(seed)
    return 100 * np.cumprod(1 + rng.normal(0.0003, 0.02, n))


def spikes_at(days, n=N):
    v = np.full(n, 1_000_000.0)
    v[list(days)] = 5_000_000.0
    return v


SPIKE = TAFilters(volume_spike=True)


# -------------------------------------------------------- trade rules

def test_entry_exit_follow_the_rules():
    """Volume spikes on days 300, 305 and 330. Hold 30 days.
    300 -> enter 301 open, exit 330 close.  305 is ignored (position open).
    330 is the exit day, so a new trade may start: enter 331, exit 360."""
    p = make_prices("A", wavy(), volume=spikes_at([300, 305, 330]))
    res = run([p], SPIKE, holding_period=30)
    t = res.trades
    assert len(t) == 2
    assert t.loc[0, "signal_date"] == DATES[300]
    assert t.loc[0, "entry_date"] == DATES[301] and t.loc[0, "entry_price"] == p["open"][301]
    assert t.loc[0, "exit_date"] == DATES[330] and t.loc[0, "exit_price"] == p["close"][330]
    assert t.loc[0, "days_held"] == 30
    assert t.loc[1, "entry_date"] == DATES[331] and t.loc[1, "exit_date"] == DATES[360]


def test_trade_return_includes_costs_both_sides():
    p = make_prices("A", wavy(), volume=spikes_at([300]))
    res = run([p], SPIKE, holding_period=30, cost_pct=0.1)
    entry, exit_ = p["open"][301], p["close"][330]
    expected = (exit_ * 0.999) / (entry * 1.001) - 1
    assert res.trades.loc[0, "return_pct"] == pytest.approx(expected * 100)


def test_daily_returns_compound_to_trade_return():
    """The equity curve and the trade log must tell the same story."""
    p = make_prices("A", wavy(), volume=spikes_at([300, 400]))
    res = run([p], SPIKE, holding_period=30, cost_pct=0.1)
    eq = res.equity.set_index("date")["strategy"]
    for _, tr in res.trades.iterrows():
        before = eq[eq.index < tr["entry_date"]].iloc[-1]
        after = eq[tr["exit_date"]]
        assert after / before - 1 == pytest.approx(tr["return_pct"] / 100)


def test_trade_closed_at_window_end():
    p = make_prices("A", wavy(), volume=spikes_at([490]))
    res = run([p], SPIKE, holding_period=30)
    t = res.trades.iloc[0]
    assert t["exit_date"] == DATES[-1] and t["closed_at_window_end"] and t["days_held"] == 9


def test_no_overlapping_positions():
    p = make_prices("A", wavy(seed=3))
    res = run([p], TAFilters(rsi_max=45), holding_period=20)
    t = res.trades
    assert len(t) > 3
    assert (t["entry_date"].iloc[1:].to_numpy() > t["exit_date"].iloc[:-1].to_numpy()).all()


# ---------------------------------------------------- known answers

def test_steady_rise_known_return():
    """Price rises exactly 1% every day: a 30-day hold must return 1.01^30 - 1 = 34.78%."""
    close = 100 * 1.01 ** np.arange(N)
    p = make_prices("A", close, volume=spikes_at([300]))
    res = run([p], SPIKE, holding_period=30)
    assert res.trades.loc[0, "return_pct"] == pytest.approx((1.01 ** 30 - 1) * 100)


def test_flat_price_zero_return():
    p = make_prices("A", np.full(N, 100.0), volume=spikes_at([300]))
    res = run([p], SPIKE, holding_period=30)
    assert res.trades.loc[0, "return_pct"] == pytest.approx(0)
    assert res.metrics["Total return %"] == pytest.approx(0)
    assert res.metrics["Max drawdown %"] == pytest.approx(0)


def test_buy_and_hold_matches_benchmark():
    """No TA filters + a 'stock' identical to the index = the benchmark's return exactly."""
    p = make_prices("NIFTYCLONE", wavy(seed=7))
    res = run([p], TAFilters(), bench=bench_from(p))
    assert res.settings["mode"] == "FA basket buy-and-hold"
    assert res.metrics["Total return %"] == pytest.approx(res.benchmark_metrics["Total return %"])
    np.testing.assert_allclose(res.equity["strategy"], res.equity["benchmark"])


def test_equal_weight_average_of_positions():
    """Two stocks held on the same days: portfolio daily return = their average."""
    a = make_prices("A", 100 * 1.01 ** np.arange(N), volume=spikes_at([300]))    # +1%/day
    b = make_prices("B", 100 * 0.995 ** np.arange(N), volume=spikes_at([300]))   # -0.5%/day
    res = run([a, b], SPIKE, holding_period=10)
    eq = res.equity.set_index("date")["strategy"]
    held_days = eq.loc[DATES[302]:DATES[310]].pct_change().dropna()   # days 2..10 of the hold
    np.testing.assert_allclose(held_days, (0.01 + -0.005) / 2, atol=1e-12)   # 0.25%, not the 0.5% sum


def test_no_trades_flat_equity():
    p = make_prices("A", wavy())
    res = run([p], TAFilters(rsi_min=99.99, rsi_max=99.999))
    assert res.trades.empty and (res.equity["strategy"] == 100).all()
    assert res.metrics["Total return %"] == 0


# --------------------------------------------------------- signals

def test_signal_only_when_condition_turns_true():
    """Price rises (RSI high) then falls for good (RSI stays low).
    RSI <= 30 becomes true once and stays true -> exactly ONE signal."""
    close = np.concatenate([100 * 1.01 ** np.arange(300), 100 * 1.01 ** 299 * 0.99 ** np.arange(1, 201)])
    g = compute_indicators(make_prices("A", close))
    sig = signals(g, TAFilters(rsi_max=30))
    assert sig.sum() == 1


def test_warmup_end_is_not_a_signal():
    """Falling from day 1: RSI is low as soon as it exists. That is not a
    'turning true' event, so no signal."""
    g = compute_indicators(make_prices("A", 100 * 0.99 ** np.arange(100)))
    assert signals(g, TAFilters(rsi_max=30)).sum() == 0


def test_no_lookahead_in_backtest():
    """Multiply every price after a cut-off by 10. Every trade that was already
    closed before the cut-off must be identical."""
    close = wavy(seed=11)
    base = make_prices("A", close)
    changed = base.copy()
    cut = 420
    changed.loc[cut:, ["open", "high", "low", "close"]] *= 10
    tf = TAFilters(rsi_max=45, ma_condition="above")
    a = run([base], tf, holding_period=15).trades
    b = run([changed], tf, holding_period=15).trades
    a, b = a[a["exit_date"] < DATES[cut]], b[b["exit_date"] < DATES[cut]]
    assert len(a) > 0
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_random_baseline_runs():
    p = make_prices("A", wavy(seed=5))
    res = run([p], TAFilters(rsi_max=45), holding_period=20, random_runs=25)
    assert len(res.random_total_returns) == 25
    assert 0 <= res.random_percentile <= 100


def test_invalid_holding_period():
    with pytest.raises(ValueError):
        run([make_prices("A", wavy())], SPIKE, holding_period=0)


# --------------------------------------------------------- metrics

def test_max_drawdown_known_value():
    # 100 -> 120 -> 90 -> 130 : worst fall is 120 -> 90 = 25%
    r = pd.Series([0.20, 90 / 120 - 1, 130 / 90 - 1])
    assert metrics.max_drawdown(r) == pytest.approx(0.25)


def test_sharpe_matches_formula():
    r = pd.Series([0.01, -0.005, 0.002, 0.007, -0.003])
    expected = (r.mean() - 0.065 / 252) / r.std(ddof=1) * math.sqrt(252)
    assert metrics.sharpe_ratio(r) == pytest.approx(expected)
    assert math.isnan(metrics.sharpe_ratio(pd.Series([0.0] * 10)))


def test_annualised_return():
    daily = pd.Series([(1.21) ** (1 / 504) - 1] * 504)      # +21% over two years
    assert metrics.annualised_return(daily) == pytest.approx(0.10)


def test_win_rate():
    assert metrics.win_rate([0.05, -0.02, 0.0, 0.01]) == pytest.approx(0.5)   # 0 is not a win


# ------------------------------------------------ corporate actions

def test_restart_drops_history_before_demerger():
    p = make_prices("X.NS", wavy())
    out = apply_corporate_actions(p, [{"ticker": "X.NS", "date": str(DATES[300].date()), "method": "restart"}])
    assert out["date"].min() == DATES[300] and len(out) == N - 300


def test_adjust_removes_fake_jump_and_keeps_history():
    close = np.full(N, 100.0)
    close[300:] = 50.0                                   # fake -50% level break
    p = make_prices("X.NS", close)
    out = apply_corporate_actions(p, [{"ticker": "X.NS", "date": str(DATES[300].date()), "method": "adjust"}])
    assert len(out) == N                                 # nothing dropped
    assert out["close"].pct_change().abs().max() == pytest.approx(0)
    assert (out["close"].iloc[300:] == 50).all()         # recent prices unchanged


def test_find_jumps_labels_known_and_unexplained():
    close = np.full(30, 100.0)
    close[20:] = 70.0
    j = find_jumps(make_prices("MYSTERY.NS", close))
    assert len(j) == 1 and j.loc[0, "status"] == "UNEXPLAINED"
