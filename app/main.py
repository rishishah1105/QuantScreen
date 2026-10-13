"""
QuantScreen — Streamlit UI.

Run from the project root:
    streamlit run app/main.py

The app only READS data/stocks.db; it never calls an API.
Set QUANTSCREEN_DB=<path> to point it at a different database (used by tests).
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # streamlit puts app/ on the path, not the project root
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data.database import DB_PATH, get_connection
from engine.backtester import backtest_strategy
from engine.indicators import get_indicator_table
from engine.screener import MA_CONDITIONS, explain, screen

DB = os.environ.get("QUANTSCREEN_DB", str(DB_PATH))

# Chart colours: validated categorical pair (CVD-safe), fixed per entity
STRATEGY = "#2a78d6"
NIFTY = "#eb6834"
INK, MUTED, GRID = "#1c2333", "#5b6475", "#e4e7ec"
RANDOM_BARS = "#9ec5f4"
CHART_CONFIG = {"displayModeBar": False}

st.set_page_config(page_title="QuantScreen", page_icon="📈", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="st-"], .stMarkdown, button, input, select, textarea {
  font-family: 'Public Sans', system-ui, sans-serif;
  font-variant-numeric: tabular-nums;
}
/* keep Streamlit's icon font for expander arrows, help icons etc. */
[data-testid="stIconMaterial"], .material-symbols-rounded, [class*="material-symbols"] {
  font-family: 'Material Symbols Rounded' !important;
}
h1 { font-weight: 700; letter-spacing: -0.02em; }
.verdict { font-size: 1.55rem; line-height: 1.35; font-weight: 600; color: #1c2333;
           max-width: 52rem; margin: 0.25rem 0 1.25rem; }
.verdict-note { color: #5b6475; font-size: 0.95rem; max-width: 52rem; margin-top: -0.75rem; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------- presets

PRESETS = {
    "Oversold quality": dict(pe=20.0, roe=15.0, rsi_on=True, rsi=(0, 40)),
    "Pullback in an uptrend": dict(roe=15.0, rsi_on=True, rsi=(0, 45), ma="above"),
    "Breakout near 52-week high": dict(roe=12.0, ma="above", high=3.0, vol=True),
    "Value + quality basket (buy and hold)": dict(pe=20.0, roe=15.0),
}

DEFAULTS = {
    "pe_on": False, "pe": 20.0, "roe_on": False, "roe": 15.0, "de_on": False, "de": 1.0,
    "mcap_on": False, "mcap": 100_000.0, "eps_on": False, "eps": 10.0,
    "rsi_on": False, "rsi": (30, 70), "ma": "any", "high_on": False, "high": 5.0,
    "low_on": False, "low": 10.0, "vol": False, "hold": 30, "cost": 0.1,
}


def apply_preset():
    name = st.session_state["preset"]
    if name not in PRESETS:
        return
    p = PRESETS[name]
    for k, v in DEFAULTS.items():
        if k in ("hold", "cost"):
            continue
        st.session_state[k] = v
    for k, v in p.items():
        st.session_state[k] = v
        if k in ("pe", "roe", "de", "mcap", "eps", "high", "low"):
            st.session_state[k + "_on"] = True


def mark_custom():
    st.session_state["preset"] = "Custom"


if "preset" not in st.session_state:
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.session_state["preset"] = "Oversold quality"
    apply_preset()
# Streamlit forgets a widget's value while the widget is hidden (e.g. a slider
# whose toggle is off); restore the default so it reappears at a sensible value.
for _k, _v in DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ------------------------------------------------------- cached calls

@st.cache_data(show_spinner=False)
def data_info(db):
    with get_connection(db) as conn:
        prices = conn.execute("SELECT MIN(date), MAX(date) FROM prices").fetchone()
        fa = conn.execute("SELECT MAX(last_updated), COUNT(*) FROM fundamentals").fetchone()
    return {"from": prices[0], "to": prices[1], "fa_date": fa[0], "fa_count": fa[1]}


@st.cache_data(show_spinner=False)
def all_tickers(db):
    return sorted(get_indicator_table(db)["ticker"].unique())


@st.cache_data(show_spinner=False)
def run_screen(db, **f):
    return screen(db_path=db, **f)


@st.cache_data(show_spinner=False)
def run_backtest(db, hold, cost, **f):
    return backtest_strategy(holding_period=hold, cost_pct=cost, db_path=db, **f)


# ------------------------------------------------------------ sidebar

with st.sidebar:
    st.selectbox("Strategy preset", ["Custom", *PRESETS], key="preset", on_change=apply_preset,
                 help="Fills in the filters below. Changing any filter switches to Custom.")

    st.subheader("Fundamentals")
    st.caption("Picks which stocks are eligible, using today's numbers.")

    def fa_control(key, label, lo, hi, step, fmt, help_=None):
        on = st.toggle(label, key=key + "_on", on_change=mark_custom, help=help_)
        if on:
            st.slider(label, lo, hi, step=step, format=fmt, key=key, on_change=mark_custom,
                      label_visibility="collapsed")

    fa_control("pe", "Max P/E", 2.0, 100.0, 1.0, "%.0f")
    fa_control("roe", "Min ROE %", 0.0, 60.0, 1.0, "%.0f%%")
    fa_control("de", "Max debt/equity", 0.0, 5.0, 0.1, "%.1f",
               "Banks and NBFCs report no D/E, so they always pass this filter.")
    fa_control("mcap", "Min market cap (₹ crore)", 0.0, 2_000_000.0, 10_000.0, "%.0f")
    fa_control("eps", "Min EPS (₹)", 0.0, 500.0, 1.0, "%.0f")

    st.subheader("Technicals")
    st.caption("Decides when to buy in the backtest.")
    if st.toggle("RSI(14) range", key="rsi_on", on_change=mark_custom,
                 help="Below 30 is usually called oversold, above 70 overbought."):
        st.slider("RSI range", 0, 100, key="rsi", on_change=mark_custom, label_visibility="collapsed")
    st.selectbox("Trend (50-day vs 200-day average)", ["any", "above", "below", "golden_cross"],
                 key="ma", format_func=lambda k: MA_CONDITIONS[None if k == "any" else k],
                 on_change=mark_custom)
    if st.toggle("Near 52-week high", key="high_on", on_change=mark_custom):
        st.slider("Within % of 52-week high", 1.0, 30.0, step=0.5, key="high", format="%.1f%%",
                  on_change=mark_custom, label_visibility="collapsed")
    if st.toggle("Near 52-week low", key="low_on", on_change=mark_custom):
        st.slider("Within % of 52-week low", 1.0, 30.0, step=0.5, key="low", format="%.1f%%",
                  on_change=mark_custom, label_visibility="collapsed")
    st.toggle("Volume spike (1.5× the 20-day average)", key="vol", on_change=mark_custom)

    st.subheader("Backtest")
    st.slider("Holding period (trading days)", 5, 90, key="hold")
    st.slider("Trading cost per side (%)", 0.0, 0.5, step=0.05, key="cost", format="%.2f%%",
              help="Brokerage, STT and slippage, charged on both buy and sell.")


ss = st.session_state
filters = {}
if ss.pe_on: filters["pe_max"] = ss.pe
if ss.roe_on: filters["roe_min"] = ss.roe
if ss.de_on: filters["de_max"] = ss.de
if ss.mcap_on: filters["mcap_min"] = ss.mcap
if ss.eps_on: filters["eps_min"] = ss.eps
if ss.rsi_on: filters["rsi_min"], filters["rsi_max"] = ss.rsi
if ss.ma != "any": filters["ma_condition"] = ss.ma
if ss.high_on: filters["near_high_pct"] = ss.high
if ss.low_on: filters["near_low_pct"] = ss.low
if ss.vol: filters["volume_spike"] = True
fa_keys = {"pe_max", "roe_min", "de_max", "mcap_min", "eps_min"}
has_ta = any(k not in fa_keys for k in filters)


# --------------------------------------------------------------- header

if not Path(DB).exists():
    st.error(f"No database found at {DB}. Run `python -m data.fetch` from the project root first.")
    st.stop()

info = data_info(DB)
st.title("QuantScreen")
st.markdown("Screen Nifty 100 stocks on fundamentals and technicals, then test the entry "
            "timing on three years of price history.")
st.caption(f"Prices from {info['from']} to {info['to']}. Fundamentals as of {info['fa_date']}. "
           "Data from Yahoo Finance.")


# -------------------------------------------------------------- screener

matches = run_screen(DB, **filters)
st.subheader(f"{len(matches)} matching stock{'s' if len(matches) != 1 else ''} today")

if matches.empty:
    st.info("No stock passes every filter right now. Loosen a filter in the sidebar — "
            "the backtest can still run, since signals may have fired in the past.")
else:
    table = matches.copy()
    table["date"] = pd.to_datetime(table["date"]).dt.date
    table["industry"] = table["industry"].fillna("")
    st.dataframe(
        table, hide_index=True,
        column_order=["ticker", "name", "industry", "close", "pe_ratio", "roe", "debt_to_equity",
                      "market_cap", "rsi", "pct_from_52w_high"],
        column_config={
            "ticker": "Ticker", "name": "Company", "industry": "Industry",
            "close": st.column_config.NumberColumn("Price ₹", format="%.2f"),
            "pe_ratio": st.column_config.NumberColumn("P/E", format="%.1f"),
            "roe": st.column_config.NumberColumn("ROE %", format="%.1f"),
            "debt_to_equity": st.column_config.NumberColumn("D/E", format="%.2f",
                                                            help="Blank for banks and NBFCs"),
            "market_cap": st.column_config.NumberColumn("Mkt cap ₹ cr", format="%.0f"),
            "rsi": st.column_config.NumberColumn("RSI", format="%.1f"),
            "pct_from_52w_high": st.column_config.NumberColumn("From 52w high %", format="%.1f"),
        },
    )

with st.expander("Why is a stock in or out?"):
    pick = st.selectbox("Stock", all_tickers(DB),
                        index=all_tickers(DB).index(matches["ticker"].iloc[0]) if len(matches) else 0)
    ex = explain(pick, db_path=DB, **filters)
    ex["passed"] = ex["passed"].map({True: "✓ pass", False: "✗ fail"})
    ex["value"] = ex["value"].astype(str)
    st.dataframe(ex, hide_index=True, column_config={
        "filter": "Filter", "value": "Value", "rule": "Rule", "passed": "Result"})


# -------------------------------------------------------------- backtest

st.divider()
st.subheader("Backtest")
mode_note = ("Buys every stock that passes the fundamentals on the first day and holds to the end."
             if not has_ta else
             f"Buys a stock the day after its technical conditions turn on, sells after "
             f"{ss.hold} trading days. Equal weight, {ss.cost:.2f}% cost per side.")
st.caption(mode_note)

params = (tuple(sorted(filters.items())), ss.hold, ss.cost)
if st.button("Run backtest", type="primary"):
    with st.spinner("Replaying three years of signals and 100 random-entry runs…"):
        ss["bt"] = (params, run_backtest(DB, ss.hold, ss.cost, **filters))

if "bt" in ss and ss["bt"][0] != params:
    st.info("Filters changed since the last run. Run the backtest again to update the results.")

if "bt" in ss and ss["bt"][0] == params:
    res = ss["bt"][1]
    m, b = res.metrics, res.benchmark_metrics
    n_trades = int(m["Trades"])
    diff = m["Total return %"] - b.get("Total return %", np.nan)

    # ---- verdict: the one sentence that answers "did it work?"
    if n_trades == 0:
        verdict = "No entry signals fired in these three years. Loosen the technical filters."
        note = ""
    elif res.settings["mode"].startswith("FA basket"):
        verdict = (f"Holding these {n_trades} stocks returned {m['Total return %']:.1f}%, "
                   f"{'beating' if diff >= 0 else 'trailing'} Nifty 50 by {abs(diff):.1f} points.")
        note = ("Treat this as an upper bound: the stocks were chosen with today's fundamentals, "
                "which nobody had three years ago.")
    else:
        pct = res.random_percentile
        timing = ("the entry timing added value" if pct >= 75 else
                  "the timing did no better than chance" if pct > 25 else
                  "random entry dates did better — the gains came from stock selection, not timing")
        verdict = (f"{m['Total return %']:.1f}% over three years, "
                   f"{'beating' if diff >= 0 else 'trailing'} Nifty 50 by {abs(diff):.1f} points — "
                   f"but it beat only {pct:.0f}% of random-entry runs, so {timing}."
                   if pct < 75 else
                   f"{m['Total return %']:.1f}% over three years, "
                   f"{'beating' if diff >= 0 else 'trailing'} Nifty 50 by {abs(diff):.1f} points, "
                   f"and it beat {pct:.0f}% of random-entry runs: {timing}.")
        note = ("Random-entry runs use the same stocks and the same number of trades on random "
                "dates. They separate the value of picking stocks from the value of timing them.")
    st.markdown(f'<p class="verdict">{verdict}</p>', unsafe_allow_html=True)
    if note:
        st.markdown(f'<p class="verdict-note">{note}</p>', unsafe_allow_html=True)

    # ---- headline numbers, each compared with Nifty
    cols = st.columns(5)
    def vs(key, unit=" pts", nd=1):
        if key not in b or pd.isna(b[key]) or pd.isna(m[key]):
            return None
        return f"{m[key] - b[key]:+.{nd}f}{unit} vs Nifty"
    cols[0].metric("Total return", f"{m['Total return %']:.1f}%", vs("Total return %"))
    cols[1].metric("Annualised", f"{m['Annualised return %']:.1f}%", vs("Annualised return %"))
    cols[2].metric("Sharpe ratio", "—" if pd.isna(m["Sharpe ratio"]) else f"{m['Sharpe ratio']:.2f}",
                   vs("Sharpe ratio", unit="", nd=2),
                   help="Return per unit of risk, after the 6.5% risk-free rate. Above 1 is good.")
    cols[3].metric("Max drawdown", f"{m['Max drawdown %']:.1f}%", vs("Max drawdown %"),
                   delta_color="inverse", help="Worst fall from a peak. Lower is better.")
    cols[4].metric("Win rate", "—" if pd.isna(m["Win rate %"]) else f"{m['Win rate %']:.0f}%",
                   f"{n_trades} trades", delta_color="off")

    # ---- equity curve
    eq = res.equity

    def style(fig, title, ytitle, height=380):
        fig.update_layout(
            title=dict(text=title, x=0, font=dict(size=16, color=INK)),
            height=height, margin=dict(l=8, r=110, t=48, b=8),
            paper_bgcolor="white", plot_bgcolor="white",
            font=dict(family="Public Sans, system-ui, sans-serif", color=MUTED, size=13),
            hovermode="x unified", showlegend=True,
            legend=dict(orientation="h", x=1, xanchor="right", y=1.02, yanchor="bottom",
                        font=dict(color=INK)),
        )
        fig.update_xaxes(showgrid=False, linecolor=GRID, ticks="outside", tickcolor=GRID)
        fig.update_yaxes(title_text=ytitle, gridcolor=GRID, gridwidth=1, zeroline=False)
        return fig

    def end_label(fig, x, y, text, color):
        fig.add_annotation(x=x, y=y, text=text, showarrow=False, xanchor="left", xshift=8,
                           font=dict(color=INK, size=12), bgcolor="white")
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers", marker=dict(size=8, color=color,
                      line=dict(color="white", width=2)), showlegend=False, hoverinfo="skip"))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq["date"], y=eq["strategy"], name="Strategy",
                             line=dict(color=STRATEGY, width=2),
                             hovertemplate="₹%{y:.1f}<extra>Strategy</extra>"))
    if eq["benchmark"].notna().any():
        fig.add_trace(go.Scatter(x=eq["date"], y=eq["benchmark"], name="Nifty 50",
                                 line=dict(color=NIFTY, width=2),
                                 hovertemplate="₹%{y:.1f}<extra>Nifty 50</extra>"))
        end_label(fig, eq["date"].iloc[-1], eq["benchmark"].iloc[-1],
                  f"Nifty ₹{eq['benchmark'].iloc[-1]:.0f}", NIFTY)
    end_label(fig, eq["date"].iloc[-1], eq["strategy"].iloc[-1],
              f"Strategy ₹{eq['strategy'].iloc[-1]:.0f}", STRATEGY)
    st.plotly_chart(style(fig, "Growth of ₹100", "Value (₹)"), config=CHART_CONFIG)

    left, right = st.columns(2)

    # ---- drawdown
    def drawdown(series):
        values = np.concatenate([[100.0], series.to_numpy()])
        peak = np.maximum.accumulate(values)
        return -((peak - values) / peak * 100)[1:]
    dd = go.Figure()
    dd.add_trace(go.Scatter(x=eq["date"], y=drawdown(eq["strategy"]), name="Strategy",
                            line=dict(color=STRATEGY, width=2),
                            hovertemplate="%{y:.1f}%<extra>Strategy</extra>"))
    if eq["benchmark"].notna().any():
        dd.add_trace(go.Scatter(x=eq["date"], y=drawdown(eq["benchmark"]), name="Nifty 50",
                                line=dict(color=NIFTY, width=2),
                                hovertemplate="%{y:.1f}%<extra>Nifty 50</extra>"))
    style(dd, "Fall from previous peak", "Drawdown (%)", height=320)
    dd.update_layout(margin=dict(r=16))
    left.plotly_chart(dd, config=CHART_CONFIG)

    # ---- random baseline
    if res.random_total_returns is not None:
        rnd = res.random_total_returns * 100
        hist = go.Figure()
        hist.add_trace(go.Histogram(x=rnd, nbinsx=20, name="Random-entry runs",
                                    marker=dict(color=RANDOM_BARS, line=dict(color="white", width=2)),
                                    hovertemplate="%{x} return: %{y} runs<extra></extra>"))
        y_top = np.histogram(rnd, bins=20)[0].max()
        for x, name, color in [(m["Total return %"], "Strategy", STRATEGY),
                               (b.get("Total return %"), "Nifty 50", NIFTY)]:
            if x is None or pd.isna(x):
                continue
            hist.add_shape(type="line", x0=x, x1=x, y0=0, y1=y_top * 1.08,
                           line=dict(color=color, width=2))
            hist.add_annotation(x=x, y=y_top * 1.08, text=f"{name} {x:.1f}%", showarrow=False,
                                yanchor="bottom", font=dict(color=INK, size=12), bgcolor="white")
        style(hist, f"Strategy vs {len(rnd)} random-entry runs", "Runs", height=320)
        hist.update_layout(hovermode="closest", showlegend=False, margin=dict(r=16), bargap=0)
        hist.update_xaxes(title_text="Total return (%)")
        right.plotly_chart(hist, config=CHART_CONFIG)
    else:
        right.info("The random-entry comparison runs when technical filters are set and "
                   "at least one trade happens.")

    # ---- trade log
    with st.expander(f"Trade log ({n_trades} trades)"):
        trades = res.trades.copy()
        for c in ("signal_date", "entry_date", "exit_date"):
            trades[c] = pd.to_datetime(trades[c]).dt.date
        st.dataframe(trades, hide_index=True, column_config={
            "ticker": "Ticker", "signal_date": "Signal", "entry_date": "Bought", "exit_date": "Sold",
            "entry_price": st.column_config.NumberColumn("Buy ₹", format="%.2f"),
            "exit_price": st.column_config.NumberColumn("Sell ₹", format="%.2f"),
            "days_held": "Days", "return_pct": st.column_config.NumberColumn("Return %", format="%.2f"),
            "closed_at_window_end": st.column_config.CheckboxColumn("Closed at end"),
        })
        st.download_button("Download trades as CSV", res.trades.to_csv(index=False),
                           file_name="quantscreen_trades.csv", mime="text/csv")

    st.caption(
        f"Window {res.settings['start']} to {res.settings['end']} after a one-year indicator warm-up. "
        "Fundamentals are today's snapshot, and the universe is today's Nifty 100, so results "
        "carry some hindsight. Past performance does not predict future returns. Not investment advice."
    )

with st.expander("How the backtest works"):
    st.markdown("""
- **Signal:** the technical conditions are true today and were false yesterday.
- **Buy:** at the next day's opening price, so no trade uses information from after the signal.
- **Sell:** at the close of the last day of the holding period. One position per stock at a time.
- **Costs:** charged on both the buy and the sell.
- **Portfolio:** equal weight across the stocks held each day; cash earns nothing.
- **Benchmark:** Nifty 50 bought on the first day of the window and held.
- **Random-entry runs:** the same stocks and number of trades on random dates, repeated 100 times.
- **Data fixes:** Tata Motors and Vedanta are tracked from their demerger dates; a price break in
  Trent's data is adjusted. Real crashes, like the Adani sell-off, stay in.
""")
