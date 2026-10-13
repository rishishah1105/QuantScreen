"""
Screener: FA filters (SQL on the fundamentals snapshot) + TA filters
(conditions on the indicator table).

    screen(pe_max=20, roe_min=15, rsi_max=40)   -> DataFrame of matching stocks
    explain("TCS.NS", pe_max=20, rsi_max=40)    -> why a stock passed / failed each filter

ta_mask() is shared with the backtester: the screener applies it to the latest
day, the backtester applies the SAME function to every historical day.
One definition of each condition means screener and backtest can't disagree.

Run from the project root for a demo:
    python -m engine.screener
"""

from dataclasses import dataclass

import pandas as pd

from data.database import DB_PATH, get_connection
from engine.indicators import get_indicator_table

MA_CONDITIONS = {
    None: "Any",
    "above": "50-day MA above 200-day MA (uptrend)",
    "below": "50-day MA below 200-day MA (downtrend)",
    "golden_cross": "50-day MA crossed above 200-day MA in the last 10 days",
}
VOLUME_SPIKE_MULT = 1.5      # today's volume > 1.5x the previous 20-day average
STALE_DAYS = 7               # ignore stocks whose latest price is older than this

RESULT_COLUMNS = [
    "ticker", "name", "industry",
    "pe_ratio", "roe", "debt_to_equity", "market_cap", "eps",
    "date", "close", "rsi", "sma50", "sma200", "pct_from_52w_high",
]


# ------------------------------------------------------------------ FA

def fa_filter(pe_max=None, roe_min=None, de_max=None, mcap_min=None, eps_min=None,
              db_path=DB_PATH) -> pd.DataFrame:
    """Stocks passing the FA filters, from today's fundamentals snapshot.

    NULL rules:
      - P/E and ROE are required: stocks missing either are always excluded.
      - D/E is optional: a NULL D/E passes the D/E filter (banks/NBFCs).
    A filter left as None is not applied.
    """
    sql = """
        SELECT s.ticker, s.symbol, s.name, s.industry, s.index_name,
               f.pe_ratio, f.roe, f.debt_to_equity, f.market_cap, f.eps, f.beta
        FROM fundamentals f
        JOIN stocks s ON s.ticker = f.ticker
        WHERE f.pe_ratio IS NOT NULL
          AND f.roe IS NOT NULL
    """
    params = {}
    if pe_max is not None:
        sql += " AND f.pe_ratio <= :pe_max"
        params["pe_max"] = pe_max
    if roe_min is not None:
        sql += " AND f.roe >= :roe_min"
        params["roe_min"] = roe_min
    if de_max is not None:
        sql += " AND (f.debt_to_equity IS NULL OR f.debt_to_equity <= :de_max)"
        params["de_max"] = de_max
    if mcap_min is not None:
        sql += " AND f.market_cap >= :mcap_min"
        params["mcap_min"] = mcap_min
    if eps_min is not None:
        sql += " AND f.eps >= :eps_min"
        params["eps_min"] = eps_min
    sql += " ORDER BY f.market_cap DESC"

    with get_connection(db_path) as conn:
        return pd.read_sql(sql, conn, params=params)


# ------------------------------------------------------------------ TA

@dataclass
class TAFilters:
    rsi_min: float | None = None
    rsi_max: float | None = None
    ma_condition: str | None = None       # None | 'above' | 'below' | 'golden_cross'
    near_high_pct: float | None = None    # close within X% below the 52-week high
    near_low_pct: float | None = None     # close within X% above the 52-week low
    volume_spike: bool = False

    def __post_init__(self):
        if self.ma_condition == "any":
            self.ma_condition = None
        if self.ma_condition not in MA_CONDITIONS:
            raise ValueError(f"ma_condition must be one of {list(MA_CONDITIONS)}, got {self.ma_condition!r}")
        if self.rsi_min is not None and self.rsi_max is not None and self.rsi_min > self.rsi_max:
            raise ValueError("rsi_min cannot be greater than rsi_max")

    def is_empty(self) -> bool:
        return (self.rsi_min is None and self.rsi_max is None and self.ma_condition is None
                and self.near_high_pct is None and self.near_low_pct is None and not self.volume_spike)


def ta_mask(df: pd.DataFrame, f: TAFilters) -> pd.Series:
    """True for each row (stock-day) that meets ALL the TA conditions.
    Works on one row per stock (screener) or every day (backtester).
    Missing indicator values (NaN) never pass: comparisons with NaN are False."""
    m = pd.Series(True, index=df.index)
    if f.rsi_min is not None:
        m &= df["rsi"] >= f.rsi_min
    if f.rsi_max is not None:
        m &= df["rsi"] <= f.rsi_max
    if f.ma_condition == "above":
        m &= df["sma50"] > df["sma200"]
    elif f.ma_condition == "below":
        m &= df["sma50"] < df["sma200"]
    elif f.ma_condition == "golden_cross":
        m &= df["golden_cross_recent"].fillna(False).astype(bool)
    if f.near_high_pct is not None:
        m &= df["close"] >= df["high_52w"] * (1 - f.near_high_pct / 100)
    if f.near_low_pct is not None:
        m &= df["close"] <= df["low_52w"] * (1 + f.near_low_pct / 100)
    if f.volume_spike:
        m &= df["volume"] > VOLUME_SPIKE_MULT * df["vol_avg20"]
    return m.fillna(False).astype(bool)


def latest_snapshot(ind: pd.DataFrame, as_of=None) -> pd.DataFrame:
    """Each stock's most recent row on or before `as_of` (default: latest date in the DB).
    Stocks whose last price is more than STALE_DAYS older are dropped (e.g. suspended)."""
    as_of = pd.Timestamp(as_of) if as_of is not None else ind["date"].max()
    snap = ind[ind["date"] <= as_of].groupby("ticker", sort=False).tail(1)
    return snap[snap["date"] >= as_of - pd.Timedelta(days=STALE_DAYS)]


# --------------------------------------------------------------- screen

def screen(pe_max=None, roe_min=None, de_max=None, mcap_min=None, eps_min=None,
           rsi_min=None, rsi_max=None, ma_condition=None,
           near_high_pct=None, near_low_pct=None, volume_spike=False,
           as_of=None, db_path=DB_PATH) -> pd.DataFrame:
    """FA filters on the fundamentals snapshot, then TA filters on each stock's
    latest trading day. Returns one row per matching stock (possibly zero rows)."""
    tf = TAFilters(rsi_min, rsi_max, ma_condition, near_high_pct, near_low_pct, volume_spike)

    fa = fa_filter(pe_max, roe_min, de_max, mcap_min, eps_min, db_path=db_path)
    if fa.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    snap = latest_snapshot(get_indicator_table(str(db_path)), as_of)
    df = fa.merge(snap, on="ticker", how="inner")
    df = df[ta_mask(df, tf)].copy()
    df["pct_from_52w_high"] = (df["close"] / df["high_52w"] - 1) * 100
    return df[RESULT_COLUMNS].reset_index(drop=True)


def explain(ticker: str, pe_max=None, roe_min=None, de_max=None, mcap_min=None, eps_min=None,
            rsi_min=None, rsi_max=None, ma_condition=None,
            near_high_pct=None, near_low_pct=None, volume_spike=False,
            as_of=None, db_path=DB_PATH) -> pd.DataFrame:
    """Pass/fail for each filter on one stock. Each check calls the same
    fa_filter / ta_mask code that screen() uses, so it can't disagree with it."""
    with get_connection(db_path) as conn:
        fund = pd.read_sql("SELECT * FROM fundamentals WHERE ticker = ?", conn, params=[ticker])
    snap = latest_snapshot(get_indicator_table(str(db_path)), as_of)
    snap = snap[snap["ticker"] == ticker]
    f = fund.iloc[0] if len(fund) else pd.Series(dtype=object)
    s = snap.iloc[0] if len(snap) else pd.Series(dtype=object)

    rows = []

    def fa_row(label, value, rule, **kw):
        passed = ticker in set(fa_filter(**kw, db_path=db_path)["ticker"])
        rows.append((label, value, rule, passed))

    def ta_row(label, value, rule, **kw):
        passed = bool(len(snap)) and bool(ta_mask(snap, TAFilters(**kw)).iloc[0])
        rows.append((label, value, rule, passed))

    fa_row("Has P/E and ROE", f"P/E={_r(f.get('pe_ratio'))}, ROE={_r(f.get('roe'))}", "both not NULL")
    if pe_max is not None:
        fa_row("P/E", _r(f.get("pe_ratio")), f"<= {pe_max}", pe_max=pe_max)
    if roe_min is not None:
        fa_row("ROE %", _r(f.get("roe")), f">= {roe_min}", roe_min=roe_min)
    if de_max is not None:
        fa_row("Debt/Equity", _r(f.get("debt_to_equity"), 3), f"<= {de_max} (NULL passes)", de_max=de_max)
    if mcap_min is not None:
        fa_row("Market cap (Rs cr)", _r(f.get("market_cap")), f">= {mcap_min}", mcap_min=mcap_min)
    if eps_min is not None:
        fa_row("EPS", _r(f.get("eps")), f">= {eps_min}", eps_min=eps_min)

    if rsi_min is not None or rsi_max is not None:
        ta_row("RSI(14)", _r(s.get("rsi")), f"{rsi_min if rsi_min is not None else '-'} to "
               f"{rsi_max if rsi_max is not None else '-'}", rsi_min=rsi_min, rsi_max=rsi_max)
    if ma_condition not in (None, "any"):
        ta_row("MA condition", f"50MA={_r(s.get('sma50'))}, 200MA={_r(s.get('sma200'))}",
               MA_CONDITIONS[ma_condition], ma_condition=ma_condition)
    if near_high_pct is not None:
        ta_row("Near 52w high", f"close={_r(s.get('close'))}, 52wH={_r(s.get('high_52w'))}",
               f"within {near_high_pct}%", near_high_pct=near_high_pct)
    if near_low_pct is not None:
        ta_row("Near 52w low", f"close={_r(s.get('close'))}, 52wL={_r(s.get('low_52w'))}",
               f"within {near_low_pct}%", near_low_pct=near_low_pct)
    if volume_spike:
        ta_row("Volume spike", f"vol={s.get('volume')}, avg20={_r(s.get('vol_avg20'))}",
               f"> {VOLUME_SPIKE_MULT}x avg", volume_spike=True)

    return pd.DataFrame(rows, columns=["filter", "value", "rule", "passed"])


def _r(v, nd=2):
    return round(float(v), nd) if v is not None and pd.notna(v) else None


# ---------------------------------------------------------------- demo

if __name__ == "__main__":
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    show = ["ticker", "pe_ratio", "roe", "debt_to_equity", "close", "rsi", "pct_from_52w_high"]

    print("\n1) No filters (everything with P/E and ROE + recent prices)")
    print(f"   -> {len(screen())} stocks")

    print("\n2) Value + quality: P/E <= 20, ROE >= 15")
    r = screen(pe_max=20, roe_min=15)
    print(f"   -> {len(r)} stocks\n{r[show].round(2).to_string(index=False)}")

    print("\n3) Context.md example: P/E <= 20, ROE >= 15, RSI <= 40 (oversold quality)")
    r = screen(pe_max=20, roe_min=15, rsi_max=40)
    print(f"   -> {len(r)} stocks\n{r[show].round(2).to_string(index=False) if len(r) else '   (none)'}")

    print("\n4) Uptrend near highs: 50MA > 200MA, within 5% of 52-week high, ROE >= 15")
    r = screen(roe_min=15, ma_condition="above", near_high_pct=5)
    print(f"   -> {len(r)} stocks\n{r[show].round(2).to_string(index=False) if len(r) else '   (none)'}")

    print("\n5) Edge case — impossible filter: P/E <= 1")
    r = screen(pe_max=1)
    print(f"   -> {len(r)} stocks (expected 0, no crash)")

    print("\n6) explain(): why TCS.NS did or didn't match screen #3")
    print(explain("TCS.NS", pe_max=20, roe_min=15, rsi_max=40).to_string(index=False))
