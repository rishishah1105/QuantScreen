# QuantScreen

**Screen Nifty 100 stocks on fundamentals and technicals, then test whether the entry timing would have worked.**

[![Tests](https://github.com/rishishah1105/QuantScreen/actions/workflows/tests.yml/badge.svg)](https://github.com/rishishah1105/QuantScreen/actions/workflows/tests.yml)

**Live demo:** https://YOUR-APP.streamlit.app

![QuantScreen backtest view]("C:\Users\Rishi\Downloads\screenshot-dark.png")

---

## Why this exists

Most screeners stop at a list of stocks that match today. Screener.in filters on fundamentals; Chartink filters on technicals. QuantScreen does both, then asks the question a list can't answer: **would buying on these signals actually have worked?**

It answers that two ways:

1. **Against the market:** the strategy's equity curve is compared with buying and holding the Nifty 50 over the same three years.
2. **Against luck:** the same stocks are traded the same number of times on *random* dates, 100 times over. Beating Nifty can come from picking good stocks; beating random entry dates is what shows the *timing* signal adds value.

## What I found

With data to 13 October 2026, holding each trade for 30 trading days with 0.1% cost per side:

| Strategy | Return (3 yrs) | Nifty 50 | Beat random entries |
|---|---|---|---|
| Oversold quality: P/E ≤ 20, ROE ≥ 15%, RSI ≤ 40 | 21.9% | 16.3% | 15% of runs |
| Pullback in an uptrend: ROE ≥ 15%, 50-day MA > 200-day MA, RSI ≤ 45 | 63.3% | 16.3% | 17% of runs |
| Same stocks as row 1, bought on day one and held | 59.4% | 16.3% | — |

Both timing strategies beat the index, but **both lost to random entry dates on the same stocks**. Buying when RSI looked "oversold" mostly bought stocks that kept falling. The returns came from *which* stocks were chosen, not *when* they were bought. The random-entry baseline exists to catch exactly this.

## How it works

```
NSE index lists ─┐
Yahoo Finance ───┼─► data/fetch.py ──► stocks.db ──► engine/indicators.py ──► engine/screener.py ──► app/main.py
                 │   (clean, fix,       (SQLite)      (RSI, moving averages,   (FA filters + TA          (Streamlit)
                 │    normalise)                       52-week levels)          conditions)
                 │                                                                   │
                 └───────────────────────────────────────────────────────────► engine/backtester.py
                                                                                (signals → trades → equity,
                                                                                 Nifty + random baseline)
```

**Data.** The current Nifty 50 and Nifty Next 50 constituents come from niftyindices.com. Four years of daily, split-adjusted prices and a fundamentals snapshot come from Yahoo Finance. Yahoo's summary data had ROE for only 24 of 100 NSE stocks, so ROE and P/E are calculated from the annual income statement and balance sheet where missing, bringing coverage to 96. Yahoo's units are normalised (ROE arrives as a fraction, debt/equity as a percentage).

**Data quality.** An automated check flags any one-day move above 20%. Each was checked against TradingView: real events (the Adani–Hindenburg sell-off, election-results day) stay in the data; unadjusted corporate actions are corrected when prices are read:
- Tata Motors and Vedanta demergers: history restarts at the demerger, since the remaining company is a different business.
- A price-scale break in Trent's data (real move ≈ 0%): earlier prices are back-adjusted, keeping the full history.

**Screener.** Fundamental filters (P/E, ROE, debt/equity, market cap, EPS) choose *which* stocks are eligible. Technical filters (RSI, 50- vs 200-day trend, golden cross, near 52-week high or low, volume spike) choose *when* to buy. Banks report no debt/equity, so a missing value passes that filter rather than silently dropping every bank.

**Backtest rules.**
- **Signal:** the technical conditions are true today *and* were false yesterday.
- **Entry:** next day's open, never the signal day's close.
- **Exit:** close of the last day of the holding period (default 30 trading days).
- **One position per stock;** signals while a position is open are ignored.
- **Costs:** 0.1% on entry and on exit (brokerage, STT, slippage).
- **Portfolio:** equal weight across the positions held each day.
- **Warm-up:** the first year of data only builds indicators; the backtest covers the three years after.

**Metrics.** Total and annualised return, Sharpe ratio (6.5% risk-free rate), maximum drawdown, win rate, and exposure, each shown next to Nifty 50.

## Testing

```bash
pytest -v      # 67 tests, run automatically on every push
```

The tests check results that are known in advance, not just "it runs":

- **Look-ahead bias:** multiply every price after a cut-off date by 10; every indicator and every trade closed before the cut-off must be unchanged.
- **Known answers:** a price rising exactly 1% a day must return 1.01³⁰ − 1 = 34.78% over a 30-day hold; a flat price must return 0%; a stock identical to the index must match the benchmark exactly.
- **Independent cross-checks:** fundamental filters against the same rules written as plain SQL; RSI against a textbook loop implementation.
- **Trade rules:** entry at the next open, exit after N days, overlapping signals ignored, costs on both sides; the equity curve must agree with the trade log.
- **UI:** the Streamlit app is driven through its main flows with Streamlit's `AppTest`.

The backtester tests were themselves checked by **mutation testing**: five deliberately broken versions of the engine (entering on the signal day, dropping the exit cost, allowing overlapping positions, and so on) each had to make at least one test fail.

## Run it locally

Requires Python 3.10 or newer.

```bash
git clone https://github.com/rishishah1105/QuantScreen.git
cd QuantScreen
python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app/main.py
```

The repository includes `data/stocks.db`, so the app works without downloading anything. To refresh the data:

```bash
python -m data.fetch                 # prices + fundamentals, 2–3 minutes
python -m data.corporate_actions     # lists any new unexplained price jumps
```

## Project structure

```
data/
  tickers.py             Nifty 50 + Next 50 constituents (live CSV, built-in fallback)
  database.py            SQLite schema and idempotent upserts
  fetch.py               download, normalise, verify
  corporate_actions.py   demerger / data-break corrections and jump check
  stocks.db              the database the app reads
engine/
  indicators.py          RSI (Wilder), moving averages, 52-week levels, volume average
  screener.py            FA filters, shared TA conditions, screen(), explain()
  backtester.py          signals, trades, portfolio, benchmark, random baseline
  metrics.py             return, Sharpe, drawdown, win rate
app/main.py              Streamlit dashboard (light and dark themes)
tests/                   67 tests
```

## Limitations

- **Fundamentals are today's snapshot.** Free sources don't offer historical fundamentals, so FA filters pick the stocks and only the technical timing is replayed historically. Results for fundamental selection therefore carry hindsight.
- **Survivorship bias.** Today's index members are replayed over the past; companies that left the Nifty 100 are missing.
- **Time-based exits only:** no stop-loss or take-profit.
- **Free data.** Yahoo Finance data can have gaps and errors; the jump check catches large ones, not small ones.

## What I'd add next

- Stop-loss and take-profit exit rules.
- Walk-forward testing: choose parameters on one period, evaluate on the next.
- Historical index membership to remove survivorship bias.
- A significance test on the random-entry comparison.

---

*For learning and research. Not investment advice.*
