"""
Fetch prices + fundamentals for the universe and store them in stocks.db.

Run from the project root:
    python -m data.fetch              # everything
    python -m data.fetch --skip-fa    # prices only (fast)
    python -m data.fetch --skip-prices

Run locally, then commit data/stocks.db. The deployed app never calls this.
"""

import argparse
import math
import time
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

from data.database import (
    DB_PATH, get_connection, init_db,
    upsert_fundamentals, upsert_prices, upsert_stocks,
)
from data.tickers import BENCHMARK, get_universe

PRICE_YEARS = 4          # ~1y warm-up for the 200-day MA + ~3y backtest
FA_DELAY_SECONDS = 0.3   # pause between .info calls
FA_RETRIES = 3

# Done-when thresholds from context.md
MIN_PRICE_STOCKS = 90
MIN_FA_STOCKS = 70


# ---------------------------------------------------------------- prices

def reshape_download(df: pd.DataFrame, tickers: list[str]) -> list[tuple]:
    """Turn yf.download's wide MultiIndex frame into DB rows:
    (date, ticker, open, high, low, close, volume).
    Handles either column order: (Ticker, Price) or (Price, Ticker)."""
    if df is None or df.empty:
        return []

    if not isinstance(df.columns, pd.MultiIndex):
        # Single ticker download: plain OHLCV columns
        df = pd.concat({tickers[0]: df}, axis=1)

    level0 = set(df.columns.get_level_values(0))
    if "Close" in level0:            # (Price, Ticker) -> swap to (Ticker, Price)
        df = df.swaplevel(0, 1, axis=1)

    available = set(df.columns.get_level_values(0))
    rows = []
    for t in tickers:
        if t not in available:
            continue
        sub = df[t].dropna(subset=["Close"])
        for date, r in sub.iterrows():
            vol = r.get("Volume")
            rows.append((
                pd.Timestamp(date).strftime("%Y-%m-%d"),
                t,
                _num(r.get("Open")),
                _num(r.get("High")),
                _num(r.get("Low")),
                _num(r.get("Close")),
                int(vol) if _num(vol) is not None else None,
            ))
    return rows


def fetch_prices(tickers: list[str], years: int = PRICE_YEARS) -> list[tuple]:
    print(f"\n[prices] Downloading {years}y of daily OHLCV for {len(tickers)} tickers...")
    df = yf.download(
        tickers,
        period=f"{years}y",
        interval="1d",
        group_by="ticker",
        auto_adjust=True,      # split/bonus-adjusted prices
        threads=True,
        progress=False,
    )
    rows = reshape_download(df, tickers)
    got = {r[1] for r in rows}
    missing = sorted(set(tickers) - got)
    print(f"[prices] {len(rows):,} rows for {len(got)} tickers")
    if missing:
        print(f"[prices] No data for: {', '.join(missing)}")
    return rows


# ---------------------------------------------------------- fundamentals

def _num(v):
    """Return a finite float, or None for missing / NaN / non-numeric."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def normalise_info(ticker: str, info: dict, fallback: dict | None = None) -> dict:
    """Map a yfinance .info dict to the fundamentals schema with clean units.

    returnOnEquity  fraction (0.15) -> percent (15.0)
    debtToEquity    percent  (45.3) -> ratio   (0.453)
    marketCap       Rs              -> Rs crore
    P/E <= 0 (loss-making)          -> NULL, so it can't pass 'P/E < X'

    Yahoo's .info often omits returnOnEquity / trailingPE for NSE stocks.
    When it does, `fallback` (from the annual statements) fills the gap:
        P/E = price / EPS
        ROE = net income / shareholders' equity x 100
    Adds 'pe_source' / 'roe_source' ('yahoo' | 'computed' | None) for reporting.
    """
    fallback = fallback or {}

    eps = _num(info.get("trailingEps"))
    if eps is None:
        eps = fallback.get("eps")

    price = next((p for p in (_num(info.get(k)) for k in
                  ("currentPrice", "regularMarketPrice", "previousClose")) if p), None)

    pe, pe_src = _num(info.get("trailingPE")), "yahoo"
    if pe is None and price and eps and eps > 0:
        pe, pe_src = price / eps, "computed"
    if pe is None or pe <= 0:
        pe, pe_src = None, None

    roe, roe_src = _num(info.get("returnOnEquity")), "yahoo"
    roe = roe * 100 if roe is not None else None
    if roe is None:
        ni, eq = fallback.get("net_income"), fallback.get("equity")
        if ni is not None and eq and eq > 0:
            roe, roe_src = ni / eq * 100, "computed"
    if roe is None:
        roe_src = None

    de = _num(info.get("debtToEquity"))
    mcap = _num(info.get("marketCap"))
    return {
        "ticker": ticker,
        "pe_ratio": round(pe, 2) if pe is not None else None,
        "roe": round(roe, 2) if roe is not None else None,
        "debt_to_equity": round(de / 100, 3) if de is not None else None,
        "market_cap": round(mcap / 1e7, 2) if mcap is not None else None,
        "eps": round(eps, 2) if eps is not None else None,
        "beta": _num(info.get("beta")),
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "pe_source": pe_src,
        "roe_source": roe_src,
    }


def _latest(df: pd.DataFrame | None, labels: list[str]):
    """Most recent non-empty value of the first matching row label.
    yfinance statements: rows = line items, columns = dates (newest first)."""
    if df is None or df.empty:
        return None
    for label in labels:
        if label in df.index:
            s = df.loc[label]
            if isinstance(s, pd.DataFrame):      # duplicate labels
                s = s.iloc[0]
            s = s.dropna()
            if len(s):
                return _num(s.iloc[0])
    return None


def _statement_fallback(tk: "yf.Ticker") -> dict:
    """Net income, equity and EPS from the annual statements."""
    try:
        income = tk.income_stmt
        balance = tk.balance_sheet
    except Exception:
        return {}
    return {
        "net_income": _latest(income, ["Net Income Common Stockholders", "Net Income",
                                       "Net Income From Continuing Operation Net Minority Interest"]),
        "equity": _latest(balance, ["Stockholders Equity", "Common Stock Equity",
                                    "Total Equity Gross Minority Interest"]),
        "eps": _latest(income, ["Diluted EPS", "Basic EPS"]),
    }


def _get_info(ticker: str) -> tuple["yf.Ticker", dict]:
    """.info with retry + backoff (Yahoo rate-limits bursts)."""
    last_err = None
    for attempt in range(FA_RETRIES):
        try:
            tk = yf.Ticker(ticker)
            info = tk.info
            if info:
                return tk, info
        except Exception as e:
            last_err = e
        time.sleep(2 ** attempt)  # 1s, 2s, 4s
    raise RuntimeError(f"no .info data ({last_err.__class__.__name__ if last_err else 'empty'})")


def fetch_fundamentals(tickers: list[str]) -> list[dict]:
    print(f"\n[fa] Fetching fundamentals for {len(tickers)} tickers (1-3 min)...")
    rows, failed = [], []
    for i, t in enumerate(tickers, 1):
        try:
            tk, info = _get_info(t)
            fallback = None
            if _num(info.get("trailingPE")) is None or _num(info.get("returnOnEquity")) is None:
                fallback = _statement_fallback(tk)   # only when Yahoo left a gap
            rows.append(normalise_info(t, info, fallback))
        except Exception as e:
            failed.append(t)
            print(f"  ! {t}: {e}")
        if i % 20 == 0:
            print(f"  ...{i}/{len(tickers)}")
        time.sleep(FA_DELAY_SECONDS)

    print(f"[fa] Got {len(rows)} / {len(tickers)}" + (f"; failed: {', '.join(failed)}" if failed else ""))
    for field in ("pe", "roe"):
        src = [r.pop(f"{field}_source") for r in rows]
        print(f"[fa] {field.upper():>3}: {src.count('yahoo')} from Yahoo, "
              f"{src.count('computed')} computed from statements, {src.count(None)} missing")
    return rows


# ------------------------------------------------------------ verification

def verify(db_path=DB_PATH) -> bool:
    with get_connection(db_path) as conn:
        q = lambda sql: conn.execute(sql).fetchone()[0]
        n_universe = q("SELECT COUNT(*) FROM stocks")
        n_price = q(f"SELECT COUNT(DISTINCT ticker) FROM prices WHERE ticker != '{BENCHMARK}'")
        n_rows = q("SELECT COUNT(*) FROM prices")
        has_bench = q(f"SELECT COUNT(*) FROM prices WHERE ticker = '{BENCHMARK}'")
        n_fa = q("SELECT COUNT(*) FROM fundamentals WHERE pe_ratio IS NOT NULL AND roe IS NOT NULL")
        n_de = q("SELECT COUNT(*) FROM fundamentals WHERE debt_to_equity IS NOT NULL")
        span = conn.execute("SELECT MIN(date), MAX(date) FROM prices").fetchone()
        dupes = q("SELECT COUNT(*) FROM (SELECT date, ticker FROM prices GROUP BY date, ticker HAVING COUNT(*) > 1)")

    checks = [
        (f"Stocks with price data: {n_price} (need {MIN_PRICE_STOCKS}+)", n_price >= MIN_PRICE_STOCKS),
        (f"Benchmark {BENCHMARK} rows: {has_bench}", has_bench > 0),
        (f"Stocks with P/E and ROE: {n_fa} (need {MIN_FA_STOCKS}+)", n_fa >= MIN_FA_STOCKS),
        (f"Duplicate (date, ticker) rows: {dupes}", dupes == 0),
    ]
    print("\n=========== Verification ===========")
    print(f"Universe: {n_universe} stocks | price rows: {n_rows:,} | range: {span[0]} -> {span[1]}")
    print(f"Stocks with D/E (optional): {n_de}")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    all_ok = all(ok for _, ok in checks)
    print("Chunk 1 done-when criteria:", "ALL PASS ✓" if all_ok else "NOT MET — see FAIL lines")
    return all_ok


# -------------------------------------------------------------------- main

def main(skip_prices: bool = False, skip_fa: bool = False, years: int = PRICE_YEARS) -> None:
    print("[universe] Resolving Nifty 50 + Nifty Next 50...")
    universe = get_universe()
    tickers = [r["ticker"] for r in universe]
    print(f"[universe] {len(tickers)} stocks")

    init_db()
    with get_connection() as conn:
        upsert_stocks(conn, universe)

    if not skip_prices:
        rows = fetch_prices(tickers + [BENCHMARK], years)
        with get_connection() as conn:
            upsert_prices(conn, rows)

    if not skip_fa:
        rows = fetch_fundamentals(tickers)
        with get_connection() as conn:
            upsert_fundamentals(conn, rows)

    verify()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fetch QuantScreen data into stocks.db")
    p.add_argument("--skip-prices", action="store_true")
    p.add_argument("--skip-fa", action="store_true")
    p.add_argument("--years", type=int, default=PRICE_YEARS)
    a = p.parse_args()
    main(a.skip_prices, a.skip_fa, a.years)
