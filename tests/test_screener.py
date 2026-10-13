"""
Screener + indicator tests. Run from the project root:  pytest -v

Each test checks a result that is known in advance — either by an independent
method (raw SQL, a hand-written loop) or by construction of the test data.
"""

import numpy as np
import pandas as pd
import pytest

from data.database import get_connection
from engine.indicators import add_indicators, rsi
from engine.screener import TAFilters, explain, fa_filter, screen, ta_mask
from tests.conftest import DATES, STOCKS, _path


# ------------------------------------------------------------------ FA

@pytest.mark.parametrize("params, where", [
    ({}, "1=1"),
    ({"pe_max": 20}, "pe_ratio <= 20"),
    ({"roe_min": 15}, "roe >= 15"),
    ({"pe_max": 20, "roe_min": 15}, "pe_ratio <= 20 AND roe >= 15"),
    ({"de_max": 1}, "(debt_to_equity IS NULL OR debt_to_equity <= 1)"),
    ({"mcap_min": 300_000}, "market_cap >= 300000"),
])
def test_fa_filter_matches_raw_sql(db_path, params, where):
    """Independent check: the same rule written as plain SQL must give the same stocks."""
    with get_connection(db_path) as conn:
        expected = {r[0] for r in conn.execute(
            f"SELECT ticker FROM fundamentals WHERE pe_ratio IS NOT NULL AND roe IS NOT NULL AND {where}")}
    got = set(fa_filter(**params, db_path=db_path)["ticker"])
    assert got == expected


def test_null_rules(db_path):
    all_fa = set(fa_filter(db_path=db_path)["ticker"])
    assert "NOPE.NS" not in all_fa                      # missing P/E -> excluded
    assert "BANK.NS" in all_fa                          # missing D/E -> still included
    low_debt = set(fa_filter(de_max=1, db_path=db_path)["ticker"])
    assert "BANK.NS" in low_debt                        # NULL D/E passes the D/E filter
    assert "LEV.NS" not in low_debt                     # D/E 3.0 fails it


def test_boundary_is_inclusive(db_path):
    assert "UP.NS" in set(fa_filter(pe_max=18.0, db_path=db_path)["ticker"])   # P/E exactly 18
    assert "UP.NS" not in set(fa_filter(pe_max=17.99, db_path=db_path)["ticker"])


def test_impossible_filter_returns_empty_not_crash(db_path):
    r = screen(pe_max=1, db_path=db_path)
    assert r.empty and "ticker" in r.columns
    r = screen(rsi_min=99.9, rsi_max=99.95, db_path=db_path)
    assert r.empty


# ----------------------------------------------------------- indicators

def _rsi_reference(close, window=14):
    """Wilder's RSI written as a plain loop — the textbook definition,
    independent of the pandas implementation being tested."""
    d = np.diff(close)
    gains, losses = np.clip(d, 0, None), np.clip(-d, 0, None)
    out = [np.nan] * len(close)
    ag = al = None
    for i in range(len(d)):
        if i < window - 1:
            continue
        if ag is None:   # seed: EMA with alpha=1/window and adjust=False starts from the first value
            ag, al = gains[0], losses[0]
            for j in range(1, window):
                ag = ag + (gains[j] - ag) / window
                al = al + (losses[j] - al) / window
        else:
            ag = ag + (gains[i] - ag) / window
            al = al + (losses[i] - al) / window
        out[i + 1] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    return np.array(out)


def test_rsi_matches_reference_loop():
    close = pd.Series(_path("wavy"))
    ours = rsi(close).to_numpy()
    ref = _rsi_reference(close.to_numpy())
    mask = ~np.isnan(ref)
    assert np.isnan(ours[:14]).all()                     # not enough history yet
    np.testing.assert_allclose(ours[mask], ref[mask], rtol=1e-9)


def test_rsi_extremes():
    assert rsi(pd.Series(_path("up"))).iloc[-1] == pytest.approx(100)    # only gains
    assert rsi(pd.Series(_path("down"))).iloc[-1] == pytest.approx(0)    # only losses
    assert rsi(pd.Series([100.0] * 50)).iloc[-1] == 50                  # flat -> neutral


def test_sma_matches_manual_mean():
    close = _path("wavy")
    g = add_indicators(pd.DataFrame({"date": DATES, "open": close, "high": close, "low": close,
                                     "close": close, "volume": 1e6}))
    assert g["sma50"].iloc[48] != g["sma50"].iloc[48]           # NaN before 50 days
    assert g["sma50"].iloc[-1] == pytest.approx(close[-50:].mean())
    assert g["sma200"].iloc[-1] == pytest.approx(close[-200:].mean())
    assert g["high_52w"].iloc[-1] == pytest.approx(close[-252:].max())


def test_no_lookahead_in_indicators():
    """Wildly change every price AFTER a cut-off date. Indicators on or before
    the cut-off must not change at all — they may only use past data."""
    close = _path("wavy")
    base = pd.DataFrame({"date": DATES, "open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": np.arange(len(close)) * 1000 + 1e6})
    cut = 260
    future_changed = base.copy()
    for col in ("open", "high", "low", "close"):
        future_changed.loc[cut + 1:, col] *= 10
    future_changed.loc[cut + 1:, "volume"] *= 50

    cols = ["rsi", "sma50", "sma200", "high_52w", "low_52w", "vol_avg20", "golden_cross_recent"]
    a = add_indicators(base).iloc[: cut + 1][cols]
    b = add_indicators(future_changed).iloc[: cut + 1][cols]
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_golden_cross_detected_and_expires():
    i = np.arange(400)
    close = np.where(i < 250, 200 - i * 0.4, 100 + (i - 250) * 1.5)   # fall, then sharp rise
    g = add_indicators(pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=400), "open": close,
                                     "high": close, "low": close, "close": close, "volume": 1e6}))
    above = g["sma50"] > g["sma200"]
    cross_idx = int(np.argmax(above.to_numpy()))
    assert cross_idx > 0 and not above.iloc[cross_idx - 1]
    assert g["golden_cross_recent"].iloc[cross_idx]
    assert g["golden_cross_recent"].iloc[cross_idx + 9]          # still "recent" 9 days later
    assert not g["golden_cross_recent"].iloc[cross_idx + 10]     # expired after 10


def test_volume_spike_uses_previous_days_only():
    vol = np.full(60, 1_000_000.0)
    vol[-1] = 1_600_000                                  # 1.6x the previous 20-day average
    close = np.linspace(100, 110, 60)
    g = add_indicators(pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=60), "open": close,
                                     "high": close, "low": close, "close": close, "volume": vol}))
    assert g["vol_avg20"].iloc[-1] == 1_000_000          # today's spike not in its own average
    assert ta_mask(g.tail(1), TAFilters(volume_spike=True)).iloc[0]


def test_nan_indicators_never_pass():
    close = _path("up")[:100]                            # < 200 days -> sma200 all NaN
    g = add_indicators(pd.DataFrame({"date": DATES[:100], "open": close, "high": close,
                                     "low": close, "close": close, "volume": 1e6}))
    assert not ta_mask(g, TAFilters(ma_condition="above")).any()
    assert not ta_mask(g, TAFilters(ma_condition="below")).any()


# --------------------------------------------------------------- screen

def test_screen_ta_conditions(db_path):
    uptrend = set(screen(ma_condition="above", db_path=db_path)["ticker"])
    assert {"UP.NS", "BANK.NS"} <= uptrend and "DOWN.NS" not in uptrend

    oversold = set(screen(rsi_max=30, db_path=db_path)["ticker"])
    assert oversold == {"DOWN.NS"}

    near_high = set(screen(near_high_pct=2, db_path=db_path)["ticker"])
    assert "UP.NS" in near_high and "DOWN.NS" not in near_high


def test_screen_combines_fa_and_ta(db_path):
    # BANK passes TA (uptrend) but fails ROE >= 15; UP passes both
    assert set(screen(roe_min=15, ma_condition="above", db_path=db_path)["ticker"]) == {"UP.NS"}


@pytest.mark.parametrize("ticker", list(STOCKS))
@pytest.mark.parametrize("filters", [
    {"pe_max": 20, "roe_min": 15},
    {"rsi_max": 40},
    {"de_max": 1, "ma_condition": "above"},
    {"roe_min": 12, "near_high_pct": 5, "rsi_min": 50},
])
def test_explain_agrees_with_screen(db_path, ticker, filters):
    """A stock is in screen() results exactly when every explain() row passes."""
    in_screen = ticker in set(screen(**filters, db_path=db_path)["ticker"])
    assert in_screen == bool(explain(ticker, **filters, db_path=db_path)["passed"].all())


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        TAFilters(ma_condition="sideways")
    with pytest.raises(ValueError):
        TAFilters(rsi_min=70, rsi_max=30)
