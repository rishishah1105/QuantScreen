"""
Stock universe: Nifty 50 + Nifty Next 50.

get_universe() tries the official constituent CSVs from niftyindices.com
first, so the list is always current. If that download fails (no internet,
site change, block), it falls back to the hardcoded lists below.

The fallback lists reflect the indices around the Sep 2026 rebalance and
may drift over time — the live CSV is the source of truth.
"""

import csv
import io
import urllib.request

BENCHMARK = "^NSEI"  # Nifty 50 index on Yahoo Finance

CSV_SOURCES = {
    "NIFTY 50": "https://niftyindices.com/IndexConstituent/ind_nifty50list.csv",
    "NIFTY NEXT 50": "https://niftyindices.com/IndexConstituent/ind_niftynext50list.csv",
}

# Browser-like header: niftyindices.com rejects requests without one
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

FALLBACK = {
    "NIFTY 50": [
        "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
        "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BHARTIARTL",
        "BSE", "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT",
        "ETERNAL", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE",
        "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDIGO", "INFY",
        "ITC", "JIOFIN", "JSWSTEEL", "KOTAKBANK", "LT",
        "M&M", "MARUTI", "MAXHEALTH", "NESTLEIND", "NTPC",
        "ONGC", "POWERGRID", "RELIANCE", "SBILIFE", "SBIN",
        "SHRIRAMFIN", "SUNPHARMA", "TATACONSUM", "TATASTEEL", "TCS",
        "TECHM", "TITAN", "TMPV", "TRENT", "ULTRACEMCO",
    ],
    "NIFTY NEXT 50": [
        "ABB", "ADANIENSOL", "ADANIGREEN", "ADANIPOWER", "AMBUJACEM",
        "BAJAJHFL", "BAJAJHLDNG", "BANKBARODA", "BOSCHLTD", "BPCL",
        "BRITANNIA", "CANBK", "CGPOWER", "CHOLAFIN", "DABUR",
        "DIVISLAB", "DLF", "DMART", "GAIL", "GODREJCP",
        "HAL", "HAVELLS", "HEROMOTOCO", "HINDZINC", "HYUNDAI",
        "ICICIGI", "INDHOTEL", "INDUSINDBK", "IOC", "IRFC",
        "JINDALSTEL", "JSWENERGY", "LICI", "LODHA", "LTIM",
        "MOTHERSON", "NAUKRI", "PFC", "PIDILITIND", "PNB",
        "RECLTD", "SHREECEM", "SIEMENS", "SOLARINDS", "TATAPOWER",
        "TORNTPHARM", "TVSMOTOR", "UNITDSPR", "VBL", "WIPRO",
    ],
}


def to_yf(symbol: str) -> str:
    """NSE symbol -> yfinance ticker. 'RELIANCE' -> 'RELIANCE.NS'."""
    return f"{symbol.strip().upper()}.NS"


def parse_constituents_csv(text: str, index_name: str) -> list[dict]:
    """Parse a niftyindices constituent CSV.
    Expected columns: Company Name, Industry, Symbol, Series, ISIN Code."""
    rows = []
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    for r in reader:
        r = {(k or "").strip(): (v or "").strip() for k, v in r.items()}
        symbol = r.get("Symbol")
        if not symbol:
            continue
        rows.append({
            "ticker": to_yf(symbol),
            "symbol": symbol,
            "name": r.get("Company Name") or symbol,
            "industry": r.get("Industry") or None,
            "index_name": index_name,
        })
    return rows


def _download(url: str, timeout: int = 15) -> str:
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _fallback(index_name: str) -> list[dict]:
    return [
        {"ticker": to_yf(s), "symbol": s, "name": s,
         "industry": None, "index_name": index_name}
        for s in FALLBACK[index_name]
    ]


def get_universe(verbose: bool = True) -> list[dict]:
    """Return the ~100-stock universe as a list of dicts:
    {ticker, symbol, name, industry, index_name}. Duplicates removed."""
    universe: dict[str, dict] = {}
    for index_name, url in CSV_SOURCES.items():
        rows = []
        try:
            rows = parse_constituents_csv(_download(url), index_name)
        except Exception as e:  # network error, block, bad CSV
            if verbose:
                print(f"  ! Could not download {index_name} list ({e.__class__.__name__}); using fallback")
        if len(rows) < 40:  # sanity check — a real list has 50 rows
            if rows and verbose:
                print(f"  ! {index_name} CSV looked incomplete ({len(rows)} rows); using fallback")
            rows = _fallback(index_name)
        elif verbose:
            print(f"  ✓ {index_name}: {len(rows)} stocks from niftyindices.com")
        for row in rows:
            universe.setdefault(row["ticker"], row)
    return list(universe.values())


if __name__ == "__main__":
    u = get_universe()
    print(f"{len(u)} tickers, e.g. {[r['ticker'] for r in u[:5]]}")
