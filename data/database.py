"""
SQLite layer: connection, schema, and idempotent upserts.

Every write is INSERT OR REPLACE keyed on the primary key, so running
fetch.py again updates rows instead of duplicating them.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "stocks.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS stocks (
    ticker      TEXT PRIMARY KEY,   -- yfinance format, e.g. RELIANCE.NS
    symbol      TEXT,               -- NSE symbol, e.g. RELIANCE
    name        TEXT,
    industry    TEXT,
    index_name  TEXT                -- 'NIFTY 50' or 'NIFTY NEXT 50'
);

CREATE TABLE IF NOT EXISTS prices (
    date        TEXT,               -- 'YYYY-MM-DD'
    ticker      TEXT,
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL,
    volume      INTEGER,
    PRIMARY KEY (date, ticker)
);

CREATE INDEX IF NOT EXISTS idx_prices_ticker_date ON prices (ticker, date);

CREATE TABLE IF NOT EXISTS fundamentals (
    ticker          TEXT PRIMARY KEY,
    pe_ratio        REAL,
    roe             REAL,           -- percent (15.0 = 15%)
    debt_to_equity  REAL,           -- ratio (0.45)
    market_cap      REAL,           -- Rs crore
    eps             REAL,
    beta            REAL,
    last_updated    TEXT
);
"""


def get_connection(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path | str = DB_PATH) -> None:
    """Create tables if they don't exist. Safe to call repeatedly."""
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA)


def upsert_stocks(conn: sqlite3.Connection, rows: list[dict]) -> int:
    conn.executemany(
        """INSERT OR REPLACE INTO stocks (ticker, symbol, name, industry, index_name)
           VALUES (:ticker, :symbol, :name, :industry, :index_name)""",
        rows,
    )
    return len(rows)


def upsert_prices(conn: sqlite3.Connection, rows: list[tuple]) -> int:
    """rows: (date, ticker, open, high, low, close, volume)"""
    conn.executemany(
        """INSERT OR REPLACE INTO prices (date, ticker, open, high, low, close, volume)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    return len(rows)


def upsert_fundamentals(conn: sqlite3.Connection, rows: list[dict]) -> int:
    conn.executemany(
        """INSERT OR REPLACE INTO fundamentals
           (ticker, pe_ratio, roe, debt_to_equity, market_cap, eps, beta, last_updated)
           VALUES (:ticker, :pe_ratio, :roe, :debt_to_equity, :market_cap, :eps, :beta, :last_updated)""",
        rows,
    )
    return len(rows)


if __name__ == "__main__":
    init_db()
    print(f"Database ready at {DB_PATH}")
