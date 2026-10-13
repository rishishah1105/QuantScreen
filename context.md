# QuantScreen — Project Context

## What This Project Is

A hybrid stock screener + backtesting engine for NSE-listed Indian equities.
Users set Fundamental Analysis (FA) and Technical Analysis (TA) filters,
the screener finds matching stocks from a universe of 100 NSE stocks,
and the integrated backtester tests how the TA entry signals would have
performed historically on the FA-selected stocks.

**Core idea:** Screen → Backtest → Validate.

---

## Why It Exists

Most screeners (Screener.in) do FA only.
Most TA tools (Chartink) do TA only.
QuantScreen combines both — that's the differentiator.

---

## Stock Universe

- Nifty 50 + Nifty Next 50 = 100 stocks (current constituents)
- `tickers.py` downloads the official constituent CSVs from niftyindices.com
  at fetch time; falls back to a hardcoded list if the download fails
- Large cap only — best FA data availability on free APIs
- yfinance ticker format: "RELIANCE.NS" (.NS suffix for NSE)

**Why not all 1800 NSE stocks:**
Small/mid caps have very patchy FA data on free APIs.
100 stocks with 90% clean data beats 1800 with 70% patchy data.

**Known limitation — survivorship bias:** the universe is *today's*
Nifty 100, replayed over past years. Stocks that dropped out of the index
aren't included, which flatters results slightly. Mention this in the README.

---

## Data Sources

| Data Type        | Source             | Notes                                          |
|------------------|--------------------|------------------------------------------------|
| OHLCV (prices)   | yfinance           | One batched `yf.download` call, split-adjusted |
| Benchmark        | yfinance (^NSEI)   | Stored in `prices` table with ticker "^NSEI"   |
| FA data          | yfinance `.info`   | One call per ticker, current snapshot only     |
| Constituents     | niftyindices.com   | Official CSVs, hardcoded fallback              |

**Data is fetched once, locally.** `stocks.db` is committed to GitHub.
The deployed Streamlit app only *reads* the database — it never calls an API.
(This avoids Yahoo/NSE rate limits and blocks on Streamlit Cloud servers.)

**Price history:** 4 years fetched. The first ~200 trading days are warm-up
for the 200-day MA, leaving ~3 clean years for backtesting.

**Key FA fields from yfinance .info — and unit normalisation:**

| yfinance field   | DB column        | Raw unit           | Stored as                  |
|------------------|------------------|--------------------|----------------------------|
| trailingPE       | pe_ratio         | ratio              | ratio (unchanged)          |
| returnOnEquity   | roe              | fraction (0.15)    | percent (15.0) — ×100      |
| debtToEquity     | debt_to_equity   | percent (45.3)     | ratio (0.453) — ÷100       |
| marketCap        | market_cap       | ₹                  | ₹ crore — ÷1e7             |
| trailingEps      | eps              | ₹                  | ₹ (unchanged)              |
| beta             | beta             | ratio              | ratio (unchanged)          |

Normalisation happens once in `fetch.py`, so screener code uses intuitive units
(ROE > 15 means 15%, D/E < 1 means debt less than equity).

---

## Tech Stack

```
Data Layer    → yfinance + pandas + SQLite (stdlib)
Logic Layer   → Python (screener filters + backtest engine)
Metrics       → numpy (Sharpe, drawdown, win rate)
TA indicators → pandas (RSI with Wilder smoothing, SMAs) — own implementation, tested
Frontend      → Streamlit
Charts        → plotly
Deployment    → Streamlit Community Cloud (reads committed stocks.db)
```

---

## Folder Structure

```
QuantScreen/
│
├── data/
│   ├── __init__.py
│   ├── tickers.py        → Nifty 50 + Next 50 constituents (live CSV + fallback)
│   ├── database.py       → SQLite connection, table creation, upserts
│   ├── fetch.py          → Pull + normalise + store data (run locally)
│   └── stocks.db         → SQLite database (committed to repo)
│
├── engine/
│   ├── __init__.py
│   ├── indicators.py     → RSI, 50/200 SMA, 52w high/low, volume avg (backward-looking only)
│   ├── screener.py       → fa_filter, ta_mask (shared with backtester), screen(), explain()
│   ├── backtester.py     → Backtest loop + trade simulation
│   └── metrics.py        → Sharpe, drawdown, win rate calc
│
├── app/
│   └── main.py           → Streamlit UI
│
├── tests/
│   ├── conftest.py       → synthetic test database with known answers
│   └── test_screener.py  → screener + indicator tests (run: pytest -v)
│
├── pytest.ini
├── context.md            → This file
└── requirements.txt
```

Run all scripts from the project root as modules, e.g. `python -m data.fetch`.

---

## Database Schema

### Table 1 — `stocks` (universe)
```sql
CREATE TABLE stocks (
    ticker      TEXT PRIMARY KEY,   -- yfinance format, e.g. RELIANCE.NS
    symbol      TEXT,               -- NSE symbol, e.g. RELIANCE
    name        TEXT,
    industry    TEXT,
    index_name  TEXT                -- 'NIFTY 50' or 'NIFTY NEXT 50'
);
```

### Table 2 — `prices` (OHLCV, split-adjusted)
```sql
CREATE TABLE prices (
    date        TEXT,               -- 'YYYY-MM-DD'
    ticker      TEXT,
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL,
    volume      INTEGER,
    PRIMARY KEY (date, ticker)
);
```

### Table 3 — `fundamentals` (FA snapshot, normalised units)
```sql
CREATE TABLE fundamentals (
    ticker          TEXT PRIMARY KEY,
    pe_ratio        REAL,
    roe             REAL,           -- percent
    debt_to_equity  REAL,           -- ratio
    market_cap      REAL,           -- ₹ crore
    eps             REAL,
    beta            REAL,
    last_updated    TEXT
);
```

All writes use `INSERT OR REPLACE`, so re-running fetch never creates duplicates.

---

## Screener Filters

### FA Filters (applied to today's snapshot — they pick the universe)
- P/E ratio → max slider (e.g. P/E < 20)
- ROE → min input (e.g. ROE > 15%)
- Debt/Equity → max input (e.g. D/E < 1)
- Market Cap → min ₹ crore
- EPS → min input

### TA Filters (computed per day — these generate backtest signals)
- RSI(14) → min/max range (Wilder's RSI, same definition as TradingView)
- 50 MA vs 200 MA → any / above (uptrend) / below (downtrend) /
  golden_cross (50 MA crossed above 200 MA within the last 10 trading days)
- Near 52-week high → close within X% below the highest high of the last 252 days
- Near 52-week low → close within X% above the lowest low of the last 252 days
- Volume spike → today's volume > 1.5 × average of the PREVIOUS 20 days

Indicators are NaN until enough history exists (e.g. 200 days for the 200 MA,
252 days for 52-week levels). NaN never passes a filter.

### One definition, two uses
`ta_mask()` in screener.py evaluates the TA conditions. The screener applies it
to each stock's latest day; the backtester applies the same function to every
historical day. Screener and backtest cannot disagree about what a condition means.

### NULL Handling Rule
```sql
-- Required for every stock
WHERE pe_ratio IS NOT NULL
AND roe IS NOT NULL
-- D/E is optional: banks/NBFCs often have no D/E in yfinance
AND (debt_to_equity IS NULL OR debt_to_equity <= :de_max)
```
Stocks with a NULL D/E pass the D/E filter and show "—" in the results table.
This keeps banks and financials in the universe.

---

## Backtester Logic

**FA vs TA in the backtest (look-ahead fix):**
FA data is a current snapshot, so it cannot be replayed historically.
FA filters choose *which* stocks are eligible; the backtest replays only
the TA signals on those stocks. The UI states this under the results:
> "FA filters use current fundamentals; the backtest tests TA entry timing
> on the FA-selected universe."

**Signal rule:** a signal fires on day *t* only when the full TA condition
is true on day *t* AND was false on day *t−1* (entry on the condition turning on).

**Position rules:**
```
Entry   → Buy at next day's open after the signal (t+1 open)
Exit    → Sell at close after holding period (default 30 trading days)
Limit   → Max one open position per stock; signals during an open
          position are ignored
Sizing  → Equal weight. Portfolio daily return = mean daily return
          of all stocks currently held. 0% on days with no positions.
```

**Output:**
```
Daily portfolio value (equity curve, starts at 100)
vs Nifty 50 buy-and-hold (^NSEI) over the same dates
Sharpe Ratio, Max Drawdown, Win Rate, Total Return % vs Benchmark %
Trade log (ticker, entry date, exit date, return %)
```

### Look-Ahead Bias Rule — Critical
TA indicators on day *t* may only use prices up to and including day *t*.
Trades enter at *t+1* open, never at *t* close.

---

## Metrics Definitions

| Metric            | Formula / Definition                                                 |
|-------------------|----------------------------------------------------------------------|
| Total Return %    | (Final Value − Initial) / Initial × 100                              |
| Annualised Return | (1 + total_return)^(252 / trading_days) − 1                          |
| Sharpe Ratio      | (mean(daily_ret) − 0.065/252) / std(daily_ret) × √252                |
| Max Drawdown      | max over t of (peak_to_date − value_t) / peak_to_date                |
| Win Rate          | Trades with return > 0 / Total closed trades × 100                   |
| Benchmark         | ^NSEI buy-and-hold over the same date range                          |

Risk-free rate: 6.5% annualised (approximate Indian 10-year bond yield).

---

## Streamlit UI Layout

```
┌─────────────────────────────────────────────────┐
│  Sidebar              │  Main Panel              │
│                       │                          │
│  FA Filters           │  [Matching Stocks Table] │
│  → P/E slider         │  Ticker | P/E | ROE | …  │
│  → ROE input          │                          │
│  → D/E input          │  [Backtest This Strategy]│
│                       │                          │
│  TA Filters           │  Results:                │
│  → RSI range          │  Return: 31% vs 18%      │
│  → MA condition       │  Sharpe: 1.4             │
│  → Volume toggle      │  Drawdown: 11%           │
│                       │  Win Rate: 58%           │
│  Holding Period       │                          │
│  → 30 days default    │  [Equity Curve Chart]    │
│                       │  Portfolio vs Nifty      │
│  [Run Screener]       │  FA-snapshot disclaimer  │
└─────────────────────────────────────────────────┘
```

---

## Build Chunks + Timeline

### Chunk 1 — Data Foundation (2-3 hrs) — DONE
**Goal:** Fetch and store clean data for 100 stocks + Nifty benchmark in SQLite

Steps:
1. tickers.py — download official constituent CSVs, fallback list, convert to .NS
2. database.py — create stocks.db with stocks, prices, fundamentals tables
3. fetch.py — one batched yf.download for prices (incl. ^NSEI),
   per-ticker .info for FA with a short delay, normalise units
4. Handle errors gracefully (missing tickers, NULL fields)
5. Print a verification summary

Done when:
- stocks.db exists with 3 populated tables
- 90+ stocks have price data, ^NSEI present
- 70+ stocks have P/E and ROE
- Re-running fetch doesn't create duplicates

---

### Chunk 2 — Screener Logic (3-4 hrs) — DONE
**Goal:** Filter stocks by FA + TA criteria via Python functions

Steps:
1. FA filter — query fundamentals table with user params (NULL rule above)
2. Compute TA indicators per stock with the ta library on the prices table
3. TA filter — evaluate conditions on the latest day (for the screener table)
   and on every day (for backtest signals)
4. Combine FA + TA into a single screen() function
5. Test with hardcoded values, no UI yet

Done when:
- screen(pe_max=20, roe_min=15, rsi_max=40) returns correct stocks
- NULL P/E / ROE stocks excluded; NULL D/E stocks kept
- Edge case handled — no stocks match strict filters

---

### Chunk 3 — Backtesting Engine (3-4 hrs)
**Goal:** Simulate strategy on historical data, output metrics

Steps:
1. Build signal series per stock (condition turning true)
2. Simulate trades with the position rules above
3. Build equal-weight daily portfolio return series → equity curve
4. Load ^NSEI from the prices table for the benchmark
5. Compute metrics
6. Verify 2-3 trades by hand

Done when:
- backtest(tickers, ta_filters, holding_period=30) runs without errors
- Returns metrics dict + equity curve + trade log
- Hand-checked trades match

---

### Chunk 4 — Streamlit UI (3-4 hrs)
**Goal:** Connect all logic into a working interface

Steps:
1. Sidebar with all filter controls
2. Run Screener button → results table
3. Backtest This Strategy button → metrics + chart + trade log
4. Equity curve with plotly (portfolio vs Nifty)
5. @st.cache_data on DB reads and indicator computation
6. Edge cases — no results, no trades, loading states

Done when:
- Full flow works end to end without crashes
- Charts render correctly

---

### Chunk 5 — Polish + Deploy (2 hrs)
**Goal:** Make it presentable and live online

Steps:
1. Clean UI — spacing, labels, descriptions, disclaimer
2. README.md (incl. FA-snapshot and survivorship-bias limitations)
3. Push to GitHub with stocks.db committed
4. Deploy on Streamlit Community Cloud
5. Test the live link; take a screenshot/GIF

Done when:
- Live Streamlit link works for anyone
- GitHub repo is public and clean

---

## Resume Line (Target)

> "Built QuantScreen — a hybrid FA+TA stock screener with integrated
> backtesting engine. Screened 100 NSE stocks across 8 filters,
> backtested strategies on 3 years of data, benchmarked against
> Nifty 50 reporting Sharpe ratio, max drawdown and annualised returns."

---

## Key Decisions Made

| Decision | Reason |
|---|---|
| Nifty 100 not all NSE | Clean data > large universe |
| yfinance for prices AND FA | One batched call, no ticker conversion, works anywhere |
| Dropped nsepython | NSE throttles/blocks non-browser and overseas traffic |
| Fetch locally, commit stocks.db | Deployed app never calls an API |
| FA = universe filter, TA = backtest signals | FA is a snapshot; replaying it would be look-ahead bias |
| Signal = condition turning true | Avoids re-buying the same stock every day |
| One position per stock, equal weight | No cash accounting needed, easy to explain |
| Normalise yfinance units in fetch.py | ROE is a fraction, D/E is a percent in raw data |
| NULL D/E allowed | Keeps banks/financials in the universe |
| 4y of prices | ~1y warm-up for 200-day MA + 3y backtest |
| ROE computed from statements when Yahoo omits it | Yahoo .info had ROE for only 24/100 NSE stocks |
| Own RSI instead of `ta` library | `ta` fails to install on newer Python; RSI is 6 lines and tested |
| Shared ta_mask() for screener + backtest | One definition of each condition |
| SQLite not PostgreSQL | Simple, local, no setup overhead |
| Streamlit not React | Python only, fast to build, deployable |
| Fixed holding period | Simpler backtester, explainable in interviews |
| 6.5% risk-free rate | India 10-year bond approximation |

---

## Libraries to Install

```bash
pip install -r requirements.txt
```
(`sqlite3` is part of Python's standard library — do not pip install it.)

---

## Important Notes for Development

1. Always use .NS suffix for yfinance tickers (RELIANCE.NS not RELIANCE)
2. Run `python -m data.fetch` from the project root, roughly once a week
3. Streamlit reruns the entire script on every interaction — use @st.cache_data
4. Verify 2-3 backtest trades by hand before trusting the full run
5. Never let future data influence past signals (look-ahead bias)
6. Run `pytest -v` before every commit — all tests must pass
