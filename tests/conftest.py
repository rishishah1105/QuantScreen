"""
Shared test fixture: a small synthetic stocks.db with price paths and
fundamentals chosen so the correct screener answer is known in advance.
Tests never touch the real data/stocks.db.
"""

import numpy as np
import pandas as pd
import pytest

from data.database import get_connection, init_db, upsert_fundamentals, upsert_prices, upsert_stocks

N_DAYS = 300
DATES = pd.bdate_range("2025-01-01", periods=N_DAYS)


def _path(kind: str) -> np.ndarray:
    i = np.arange(N_DAYS)
    if kind == "up":
        return 100 * 1.003 ** i                         # steady rise
    if kind == "down":
        return 100 * 0.997 ** i                         # steady fall
    if kind == "wavy":
        return 100 + 10 * np.sin(i / 8) + i * 0.02      # oscillating, mild drift
    raise ValueError(kind)


# ticker: (price path, pe, roe, d/e, market cap Rs cr, eps)
STOCKS = {
    "UP.NS":   ("up",   18.0, 22.0, 0.40, 500_000, 50.0),
    "DOWN.NS": ("down", 12.0, 16.0, 0.20, 200_000, 30.0),
    "BANK.NS": ("up",   15.0, 14.0, None, 900_000, 80.0),   # bank: no D/E
    "LEV.NS":  ("wavy", 25.0, 12.0, 3.00, 100_000, 10.0),   # high debt
    "NOPE.NS": ("wavy", None, 18.0, 0.50, 150_000, 20.0),   # missing P/E -> always excluded
}


def price_rows(ticker: str, close: np.ndarray, volume=None) -> list[tuple]:
    vol = volume if volume is not None else np.full(len(close), 1_000_000)
    return [
        (d.strftime("%Y-%m-%d"), ticker, c * 0.995, c * 1.01, c * 0.99, c, int(v))
        for d, c, v in zip(DATES, close, vol)
    ]


@pytest.fixture(scope="session")
def db_path(tmp_path_factory) -> str:
    path = str(tmp_path_factory.mktemp("db") / "test_stocks.db")
    init_db(path)
    with get_connection(path) as conn:
        upsert_stocks(conn, [
            {"ticker": t, "symbol": t[:-3], "name": t[:-3].title(), "industry": "Test", "index_name": "NIFTY 50"}
            for t in STOCKS
        ])
        rows = []
        for t, (kind, *_rest) in STOCKS.items():
            rows += price_rows(t, _path(kind))
        rows += price_rows("^NSEI", _path("up"))
        upsert_prices(conn, rows)
        upsert_fundamentals(conn, [
            {"ticker": t, "pe_ratio": pe, "roe": roe, "debt_to_equity": de, "market_cap": mc,
             "eps": eps, "beta": 1.0, "last_updated": "2026-01-01"}
            for t, (_k, pe, roe, de, mc, eps) in STOCKS.items()
        ])
    return path
