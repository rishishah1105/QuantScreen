"""
QuantScreen — Streamlit UI.

Run from the project root:
    streamlit run app/main.py

The app only READS data/stocks.db; it never calls an API.
Set QUANTSCREEN_DB=<path> to point it at a different database (used by tests).

Light and dark themes are defined in .streamlit/config.toml; the viewer switches
between them from the ⋮ menu. Custom styling is theme-neutral, so it never needs
to know which one is active.
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
from data.tickers import BENCHMARK
from engine.backtester import backtest_strategy
from engine.indicators import get_indicator_table
from engine.screener import MA_CONDITIONS, explain, screen

DB = os.environ.get("QUANTSCREEN_DB", str(DB_PATH))
CHART_CONFIG = {"displayModeBar": False}

st.set_page_config(page_title="QuantScreen", page_icon="📈", layout="wide")

# ----------------------------------------------------------- colours
# Strategy = blue, Nifty = orange in every chart. This pair passes the
# colour-blind separation checks on BOTH the light and the dark background.
#
# Nothing below depends on which theme is active: panels use translucent
# fills and inherit text colour, and Plotly's axes/text come from Streamlit's
# own chart theme. So switching light/dark (⋮ menu) updates everything at once.
C = dict(strategy="#3283e0", nifty="#e2622e", bars="rgba(128, 140, 160, 0.45)")

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
.block-container { padding-top: 4rem; }

/* Header band: deep navy in both themes */
.qs-band { background: #13213a; color: #ffffff; border: 1px solid rgba(255, 255, 255, 0.08);
           border-radius: 14px; padding: 1.6rem 1.8rem 1.4rem; margin-bottom: 1.2rem; }
.qs-band h1 { color: #ffffff; font-size: 2.1rem; font-weight: 700; letter-spacing: -0.02em;
              margin: 0; padding: 0; }
.qs-band p { color: #a9b4c9; margin: 0.35rem 0 1.2rem; font-size: 1.02rem; max-width: 46rem; }
.qs-stats { display: flex; flex-wrap: wrap; gap: 0.6rem 2.6rem; }
.qs-stat .v { color: #ffffff; font-size: 1.25rem; font-weight: 600; }
.qs-stat .v .up { color: #4cc38a; font-size: 0.95rem; font-weight: 500; margin-left: 0.35rem; }
.qs-stat .v .down { color: #ff7a6b; font-size: 0.95rem; font-weight: 500; margin-left: 0.35rem; }
.qs-stat .l { color: #a9b4c9; font-size: 0.85rem; }

/* Translucent panels: read correctly on light and dark backgrounds */
[data-testid="stMetric"] { background: rgba(128, 140, 160, 0.08);
                           border: 1px solid rgba(128, 140, 160, 0.25);
                           border-radius: 12px; padding: 0.9rem 1rem; }
.verdict-box { border-left: 4px solid #3283e0; background: rgba(50, 131, 224, 0.09);
               border-radius: 0 12px 12px 0; padding: 1rem 1.3rem; margin: 0.6rem 0 1.2rem; }
.verdict { font-size: 1.45rem; line-height: 1.35; font-weight: 600; color: inherit;
           max-width: 54rem; margin: 0; }
.verdict-note { color: inherit; opacity: 0.72; font-size: 0.95rem; max-width: 54rem; margin: 0.5rem 0 0; }
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
    for k, v in DEFAULTS.items():
        if k not in ("hold", "cost"):
            st.session_state[k] = v
    for k, v in PRESETS[name].items():
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
        fa = conn.execute("SELECT MAX(last_updated), "
                          "SUM(pe_ratio IS NOT NULL AND roe IS NOT NULL) FROM fundamentals").fetchone()
        nifty = [r[0] for r in conn.execute(
            "SELECT close FROM prices WHERE ticker = ? ORDER BY date DESC LIMIT 253", [BENCHMARK])]
    return {"from": prices[0], "to": prices[1], "fa_date": fa[0], "screened": int(fa[1] or 0),
            "nifty": nifty[0] if nifty else None,
            "nifty_1y": (nifty[0] / nifty[-1] - 1) * 100 if len(nifty) > 250 else None}


@st.cache_data(show_spinner=False)
def all_tickers(db):
    return sorted(get_indicator_table(db)["ticker"].unique())


@st.cache_data(show_spinner=False)
def sparklines(db, days=63):
    """Last ~3 months of closes per stock, for the table's mini charts."""
    ind = get_indicator_table(db)
    return {t: g["close"].tail(days).round(2).tolist() for t, g in ind.groupby("ticker")}


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
        if st.toggle(label, key=key + "_on", on_change=mark_custom, help=help_):
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

    st.caption("Switch between light and dark from the ⋮ menu at the top right.")


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


def _fmt_date(d):
    if not d:
        return "—"
    ts = pd.Timestamp(d)
    return f"{ts.day} {ts:%b %Y}"          # no %-d: it crashes on Windows


nifty_html = ""
if info["nifty"]:
    chg = info["nifty_1y"]
    chg_html = (f'<span class="{"up" if chg >= 0 else "down"}">{"▲" if chg >= 0 else "▼"} '
                f'{abs(chg):.1f}% in a year</span>' if chg is not None else "")
    nifty_html = (f'<div class="qs-stat"><div class="v">{info["nifty"]:,.0f}{chg_html}</div>'
                  f'<div class="l">Nifty 50, last close</div></div>')

st.markdown(f"""
<div class="qs-band">
  <h1>QuantScreen</h1>
  <p>Screen Nifty 100 stocks on fundamentals and technicals, then test whether the entry
     timing would have worked over three years of prices.</p>
  <div class="qs-stats">
    {nifty_html}
    <div class="qs-stat"><div class="v">{info['screened']}</div>
      <div class="l">stocks with complete fundamentals</div></div>
    <div class="qs-stat"><div class="v">{_fmt_date(info['to'])}</div>
      <div class="l">latest price data</div></div>
    <div class="qs-stat"><div class="v">{_fmt_date(info['from'])}</div>
      <div class="l">history starts</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

tab_screen, tab_bt, tab_how = st.tabs(["Screener", "Backtest", "How it works"])


# -------------------------------------------------------------- screener

with tab_screen:
    matches = run_screen(DB, **filters)
    st.subheader(f"{len(matches)} matching stock{'s' if len(matches) != 1 else ''} today")

    if matches.empty:
        st.info("No stock passes every filter right now. Loosen a filter in the sidebar — "
                "the backtest can still run, since signals may have fired in the past.")
    else:
        table = matches.copy()
        table["industry"] = table["industry"].fillna("")
        spark = sparklines(DB)
        table["trend"] = table["ticker"].map(lambda t: spark.get(t, []))
        st.dataframe(
            table, hide_index=True,
            column_order=["ticker", "name", "industry", "trend", "close", "pe_ratio", "roe",
                          "debt_to_equity", "market_cap", "rsi", "pct_from_52w_high"],
            column_config={
                "ticker": "Ticker", "name": "Company", "industry": "Industry",
                "trend": st.column_config.LineChartColumn("Last 3 months", width="small"),
                "close": st.column_config.NumberColumn("Price ₹", format="%.2f"),
                "pe_ratio": st.column_config.NumberColumn("P/E", format="%.1f"),
                "roe": st.column_config.NumberColumn("ROE %", format="%.1f"),
                "debt_to_equity": st.column_config.NumberColumn("D/E", format="%.2f",
                                                                help="None for banks and NBFCs, which don't report it"),
                "market_cap": st.column_config.NumberColumn("Mkt cap ₹ cr", format="%.0f"),
                "rsi": st.column_config.NumberColumn("RSI", format="%.1f"),
                "pct_from_52w_high": st.column_config.NumberColumn("From 52w high %", format="%.1f"),
            },
        )

    with st.expander("Why is a stock in or out?"):
        tickers = all_tickers(DB)
        default = tickers.index(matches["ticker"].iloc[0]) if len(matches) else 0
        pick = st.selectbox("Stock", tickers, index=default)
        ex = explain(pick, db_path=DB, **filters)
        ex["passed"] = ex["passed"].map({True: "✓ pass", False: "✗ fail"})
        ex["value"] = ex["value"].astype(str)
        st.dataframe(ex, hide_index=True, column_config={
            "filter": "Filter", "value": "Value", "rule": "Rule", "passed": "Result"})


# -------------------------------------------------------------- charts

def style(fig, title, ytitle, height=380, right_margin=110):
    """Layout only. Text, grid and hover colours come from Streamlit's chart
    theme, so they follow the light/dark switch without a rerun."""
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=16)),
        height=height, margin=dict(l=8, r=right_margin, t=48, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Public Sans, system-ui, sans-serif", size=13),
        hovermode="x unified", showlegend=True,
        legend=dict(orientation="h", x=1, xanchor="right", y=1.02, yanchor="bottom"),
    )
    fig.update_xaxes(showgrid=False, ticks="outside")
    fig.update_yaxes(title_text=ytitle, zeroline=False)
    return fig


def end_label(fig, x, y, text, color):
    fig.add_annotation(x=x, y=y, text=text, showarrow=False, xanchor="left", xshift=8,
                       font=dict(size=12))
    fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers", showlegend=False, hoverinfo="skip",
                             marker=dict(size=8, color=color)))


def drawdown(series):
    values = np.concatenate([[100.0], series.to_numpy()])
    peak = np.maximum.accumulate(values)
    return -((peak - values) / peak * 100)[1:]


# -------------------------------------------------------------- backtest

with tab_bt:
    st.caption("Buys every stock that passes the fundamentals on the first day and holds to the end."
               if not has_ta else
               f"Buys a stock the day after its technical conditions turn on and sells after "
               f"{ss.hold} trading days. Equal weight, {ss.cost:.2f}% cost per side.")

    params = (tuple(sorted(filters.items())), ss.hold, ss.cost)
    if st.button("Run backtest", type="primary"):
        with st.spinner("Replaying three years of signals and 100 random-entry runs…"):
            ss["bt"] = (params, run_backtest(DB, ss.hold, ss.cost, **filters))

    if "bt" not in ss:
        st.info("Set your filters in the sidebar, then run the backtest to see how the strategy "
                "would have done over the last three years.")
    elif ss["bt"][0] != params:
        st.info("Filters changed since the last run. Run the backtest again to update the results.")
    else:
        res = ss["bt"][1]
        m, b = res.metrics, res.benchmark_metrics
        n_trades = int(m["Trades"])
        diff = m["Total return %"] - b.get("Total return %", np.nan)
        beat = "beating" if diff >= 0 else "trailing"

        # ---- verdict: the one sentence that answers "did it work?"
        if n_trades == 0:
            verdict, note = "No entry signals fired in these three years. Loosen the technical filters.", ""
        elif res.settings["mode"].startswith("FA basket"):
            verdict = (f"Holding these {n_trades} stocks returned {m['Total return %']:.1f}%, "
                       f"{beat} Nifty 50 by {abs(diff):.1f} points.")
            note = ("Treat this as an upper bound: the stocks were chosen with today's fundamentals, "
                    "which nobody had three years ago.")
        else:
            pct = res.random_percentile
            head = f"{m['Total return %']:.1f}% over three years, {beat} Nifty 50 by {abs(diff):.1f} points"
            if pct >= 75:
                verdict = f"{head}, and it beat {pct:.0f}% of random-entry runs: the entry timing added value."
            elif pct > 25:
                verdict = f"{head}, but it beat only {pct:.0f}% of random-entry runs, so the timing did no better than chance."
            else:
                verdict = (f"{head}, but it beat only {pct:.0f}% of random-entry runs. The gains came "
                           f"from stock selection, not timing.")
            note = ("Random-entry runs use the same stocks and the same number of trades on random "
                    "dates. They separate the value of picking stocks from the value of timing them.")
        note_html = f'<p class="verdict-note">{note}</p>' if note else ""
        st.markdown(f'<div class="verdict-box"><p class="verdict">{verdict}</p>{note_html}</div>',
                    unsafe_allow_html=True)

        # ---- headline numbers, each compared with Nifty
        def vs(key, unit=" pts", nd=1):
            if key not in b or pd.isna(b[key]) or pd.isna(m[key]):
                return None
            return f"{m[key] - b[key]:+.{nd}f}{unit} vs Nifty"

        cols = st.columns(5)
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
        has_bench = eq["benchmark"].notna().any()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eq["date"], y=eq["strategy"], name="Strategy",
                                 line=dict(color=C["strategy"], width=2),
                                 hovertemplate="₹%{y:.1f}<extra>Strategy</extra>"))
        if has_bench:
            fig.add_trace(go.Scatter(x=eq["date"], y=eq["benchmark"], name="Nifty 50",
                                     line=dict(color=C["nifty"], width=2),
                                     hovertemplate="₹%{y:.1f}<extra>Nifty 50</extra>"))
            end_label(fig, eq["date"].iloc[-1], eq["benchmark"].iloc[-1],
                      f"Nifty ₹{eq['benchmark'].iloc[-1]:.0f}", C["nifty"])
        end_label(fig, eq["date"].iloc[-1], eq["strategy"].iloc[-1],
                  f"Strategy ₹{eq['strategy'].iloc[-1]:.0f}", C["strategy"])
        st.plotly_chart(style(fig, "Growth of ₹100", "Value (₹)"), config=CHART_CONFIG)

        left, right = st.columns(2)

        # ---- drawdown
        dd = go.Figure()
        dd.add_trace(go.Scatter(x=eq["date"], y=drawdown(eq["strategy"]), name="Strategy",
                                line=dict(color=C["strategy"], width=2),
                                hovertemplate="%{y:.1f}%<extra>Strategy</extra>"))
        if has_bench:
            dd.add_trace(go.Scatter(x=eq["date"], y=drawdown(eq["benchmark"]), name="Nifty 50",
                                    line=dict(color=C["nifty"], width=2),
                                    hovertemplate="%{y:.1f}%<extra>Nifty 50</extra>"))
        left.plotly_chart(style(dd, "Fall from previous peak", "Drawdown (%)", 320, 16),
                          config=CHART_CONFIG)

        # ---- random baseline
        if res.random_total_returns is not None:
            rnd = res.random_total_returns * 100
            # Same bin edges for the drawing and for the label height, so labels clear the bars
            edges = np.histogram_bin_edges(rnd, bins=20)
            y_top = np.histogram(rnd, bins=edges)[0].max()
            hist = go.Figure()
            hist.add_trace(go.Histogram(
                x=rnd, name="Random-entry runs", marker=dict(color=C["bars"]),
                xbins=dict(start=edges[0], end=edges[-1] + 1e-9, size=edges[1] - edges[0]),
                hovertemplate="%{x} return: %{y} runs<extra></extra>"))
            # Two marker lines; the Nifty label sits higher so the two never overlap
            for x, name, color, lift in [(m["Total return %"], "Strategy", C["strategy"], 1.10),
                                         (b.get("Total return %"), "Nifty 50", C["nifty"], 1.28)]:
                if x is None or pd.isna(x):
                    continue
                hist.add_shape(type="line", x0=x, x1=x, y0=0, y1=y_top * lift,
                               line=dict(color=color, width=2))
                hist.add_annotation(x=x, y=y_top * lift, text=f"{name} {x:.1f}%", showarrow=False,
                                    yanchor="bottom", font=dict(size=12))
            style(hist, f"Strategy vs {len(rnd)} random-entry runs", "Runs", 320, 16)
            hist.update_layout(hovermode="closest", showlegend=False, bargap=0.08)
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


# ---------------------------------------------------------- how it works

with tab_how:
    st.markdown("""
#### The idea
Most screeners stop at a list of stocks. QuantScreen also asks whether buying them on the
screener's signals would have worked, and whether that beat simply buying at random times.

#### Backtest rules
- **Signal:** the technical conditions are true today and were false yesterday.
- **Buy:** at the next day's opening price, so no trade uses information from after the signal.
- **Sell:** at the close of the last day of the holding period. One position per stock at a time.
- **Costs:** charged on both the buy and the sell.
- **Portfolio:** equal weight across the stocks held each day; cash earns nothing.
- **Benchmark:** Nifty 50 bought on the first day of the window and held.
- **Random-entry runs:** the same stocks and number of trades on random dates, repeated 100 times.

#### Data
- Daily prices for the current Nifty 50 and Nifty Next 50 from Yahoo Finance, adjusted for splits.
- Fundamentals are today's snapshot. Where Yahoo leaves out ROE or P/E, they are calculated from
  the latest annual financial statements.
- Tata Motors and Vedanta are tracked from their demerger dates, and a price break in Trent's
  data is adjusted. Real crashes, like the Adani sell-off, stay in.

#### Known limits
- Fundamentals are not historical, so the stock list carries hindsight.
- Today's index members are replayed over the past (survivorship bias).
- Exits are time-based only; there is no stop-loss or take-profit.
""")
