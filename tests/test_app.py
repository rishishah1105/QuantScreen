"""
Smoke tests for the Streamlit UI, using Streamlit's AppTest (no browser needed).
They run app/main.py against a synthetic database and click through the main flows,
checking that nothing crashes and the key outputs appear.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from data.database import get_connection, init_db, upsert_fundamentals, upsert_prices, upsert_stocks

APP = str(Path(__file__).resolve().parents[1] / "app" / "main.py")


@pytest.fixture(scope="module")
def app_db(tmp_path_factory):
    """30 random-walk stocks + benchmark over ~2.5 years: enough for warm-up + a backtest window."""
    path = str(tmp_path_factory.mktemp("appdb") / "app.db")
    init_db(path)
    dates = pd.bdate_range("2024-01-01", periods=650)
    rng = np.random.default_rng(3)
    tickers = [f"S{i:02d}.NS" for i in range(30)]
    rows = []
    for t in tickers + ["^NSEI"]:
        c = 500 * np.cumprod(1 + rng.normal(0.0004, 0.018, len(dates)))
        o = np.concatenate([[c[0]], c[:-1]])
        v = rng.integers(500_000, 1_500_000, len(dates)).astype(float)
        rows += [(d.strftime("%Y-%m-%d"), t, oo, max(oo, cc) * 1.005, min(oo, cc) * 0.995, cc, int(vv))
                 for d, oo, cc, vv in zip(dates, o, c, v)]
    with get_connection(path) as conn:
        upsert_stocks(conn, [{"ticker": t, "symbol": t[:-3], "name": f"Company {t[:-3]}",
                              "industry": "Test", "index_name": "NIFTY 50"} for t in tickers])
        upsert_prices(conn, rows)
        upsert_fundamentals(conn, [{
            "ticker": t, "pe_ratio": float(rng.uniform(8, 40)), "roe": float(rng.uniform(8, 35)),
            "debt_to_equity": None if i % 5 == 0 else float(rng.uniform(0, 2)),
            "market_cap": float(rng.uniform(5e4, 1e6)), "eps": float(rng.uniform(5, 100)),
            "beta": 1.0, "last_updated": "2026-10-13"} for i, t in enumerate(tickers)])
    return path


@pytest.fixture
def app(app_db, monkeypatch):
    monkeypatch.setenv("QUANTSCREEN_DB", app_db)
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def _text(at):
    return " ".join(m.value for m in at.markdown) + " ".join(s.value for s in at.subheader)


def test_loads_with_default_preset(app):
    assert app.title[0].value == "QuantScreen"
    assert app.selectbox(key="preset").value == "Oversold quality"
    assert "matching stock" in _text(app)


def test_backtest_runs_and_shows_verdict(app):
    app.button[0].click().run()
    assert not app.exception, app.exception
    assert 'class="verdict"' in _text(app)
    assert len(app.metric) == 5
    assert len(app.get("plotly_chart")) >= 2


def test_basket_preset(app):
    app.selectbox(key="preset").set_value("Value + quality basket (buy and hold)").run()
    app.button[0].click().run()
    assert not app.exception, app.exception
    assert "Holding these" in _text(app)


def test_changing_a_filter_switches_to_custom_and_marks_results_stale(app):
    app.button[0].click().run()
    app.toggle(key="vol").set_value(True).run()
    assert not app.exception, app.exception
    assert app.selectbox(key="preset").value == "Custom"
    assert any("Filters changed" in i.value for i in app.info)


def test_impossible_filter_shows_empty_state(app):
    app.slider(key="pe").set_value(2.0).run()
    assert not app.exception, app.exception
    assert any("No stock passes" in i.value for i in app.info)


def test_missing_database_shows_error(monkeypatch, tmp_path):
    monkeypatch.setenv("QUANTSCREEN_DB", str(tmp_path / "missing.db"))
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert at.error and "python -m data.fetch" in at.error[0].value
