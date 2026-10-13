"""
Price-data corrections for corporate actions Yahoo did not adjust for,
plus a check that flags any new suspicious jump.

Corrections are applied when prices are READ (engine/indicators.load_prices),
never written into stocks.db — so re-running fetch.py can't undo them.

Two kinds of fix:
  restart  Demergers. The company left after the demerger is a different
           business, so its earlier prices are dropped: the stock is treated
           as if it listed on `date`.
  adjust   A price-scale break where the real move was ~0% (verified on
           TradingView). All prices before `date` are multiplied by
           close[date] / close[day before], so that day's change becomes 0%
           and the full history is kept. This is how data vendors
           back-adjust for splits.

Run from the project root to list unexplained >20% one-day moves:
    python -m data.corporate_actions
"""

import pandas as pd

JUMP_THRESHOLD_PCT = 20.0

# date = first trading day AFTER the action (the day of the jump)
CORPORATE_ACTIONS = [
    {"ticker": "TMPV.NS",  "date": "2025-10-14", "method": "restart",
     "note": "Tata Motors demerged its commercial-vehicle business"},
    {"ticker": "VEDL.NS",  "date": "2026-04-30", "method": "restart",
     "note": "Vedanta demerger into separate listed companies"},
    {"ticker": "TRENT.NS", "date": "2026-01-01", "method": "adjust",
     "note": "Price-scale break in Yahoo data; TradingView shows a small gain that day"},
]

# Large one-day moves that really happened. Kept in the data on purpose:
# real crashes are risk a strategy must survive.
KNOWN_REAL_MOVES = {
    ("ADANIENSOL.NS", "2023-01-27"): "Hindenburg report",
    ("ADANIENT.NS", "2023-02-01"): "Hindenburg report / FPO withdrawn",
    ("ADANIENT.NS", "2023-02-02"): "Hindenburg report",
    ("ADANIENT.NS", "2023-02-08"): "Rebound",
    ("ADANIENT.NS", "2024-11-21"): "US indictment news",
    ("ADANIGREEN.NS", "2024-11-29"): "Rebound after indictment clarification",
    ("ADANIPORTS.NS", "2024-06-04"): "Election results day",
    ("PFC.NS", "2024-06-04"): "Election results day",
    ("HINDZINC.NS", "2024-05-21"): "Rally",
    ("IDEA.NS", "2023-02-06"): "Volatile low-priced stock",
    ("IDEA.NS", "2023-12-29"): "Volatile low-priced stock",
    ("IRFC.NS", "2024-01-16"): "PSU rally",
    ("POLYCAB.NS", "2024-01-11"): "Income-tax investigation news",
    ("POWERINDIA.NS", "2025-01-30"): "Results-day upper circuit",
}

PRICE_COLS = ["open", "high", "low", "close"]


def apply_corporate_actions(prices: pd.DataFrame, actions=None) -> pd.DataFrame:
    """Return a corrected copy of a long price table (date, ticker, open, high, low, close, volume)."""
    actions = CORPORATE_ACTIONS if actions is None else actions
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"])
    for a in actions:
        t, d = a["ticker"], pd.Timestamp(a["date"])
        is_t = df["ticker"] == t
        if not is_t.any():
            continue
        if a["method"] == "restart":
            df = df[~(is_t & (df["date"] < d))]
        elif a["method"] == "adjust":
            g = df[is_t].sort_values("date")
            before, on = g[g["date"] < d], g[g["date"] >= d]
            if before.empty or on.empty:
                continue
            factor = on["close"].iloc[0] / before["close"].iloc[-1]
            df.loc[is_t & (df["date"] < d), PRICE_COLS] *= factor
        else:
            raise ValueError(f"Unknown method {a['method']!r} for {t}")
    return df.reset_index(drop=True)


def find_jumps(prices: pd.DataFrame, threshold_pct: float = JUMP_THRESHOLD_PCT) -> pd.DataFrame:
    """Every one-day close-to-close move bigger than threshold, labelled:
    'known real move', or 'UNEXPLAINED' (needs checking on TradingView)."""
    df = prices.sort_values(["ticker", "date"]).copy()
    df["chg_pct"] = df.groupby("ticker")["close"].pct_change() * 100
    jumps = df[df["chg_pct"].abs() > threshold_pct][["date", "ticker", "close", "chg_pct"]].copy()
    jumps["date_str"] = pd.to_datetime(jumps["date"]).dt.strftime("%Y-%m-%d")
    jumps["status"] = [
        "known real move: " + KNOWN_REAL_MOVES[(t, d)] if (t, d) in KNOWN_REAL_MOVES else "UNEXPLAINED"
        for t, d in zip(jumps["ticker"], jumps["date_str"])
    ]
    return jumps.drop(columns="date_str").reset_index(drop=True)


def unexplained_jumps(prices_corrected: pd.DataFrame) -> pd.DataFrame:
    j = find_jumps(prices_corrected)
    return j[j["status"] == "UNEXPLAINED"]


if __name__ == "__main__":
    from engine.indicators import load_prices
    pd.set_option("display.width", 160)
    raw = load_prices(apply_corrections=False)
    fixed = load_prices()
    print(f"Corporate-action corrections applied: {len(CORPORATE_ACTIONS)}")
    for a in CORPORATE_ACTIONS:
        print(f"  {a['ticker']:<12} {a['date']}  {a['method']:<8} {a['note']}")
    print(f"\nOne-day moves > {JUMP_THRESHOLD_PCT:.0f}% AFTER corrections:")
    j = find_jumps(fixed)
    print(j.round(1).to_string(index=False) if len(j) else "  none")
    n = (j["status"] == "UNEXPLAINED").sum()
    print(f"\n{n} unexplained jump(s)." + (" Check each on TradingView, then add it to "
          "CORPORATE_ACTIONS (if fake) or KNOWN_REAL_MOVES (if real)." if n else " Data looks clean."))
