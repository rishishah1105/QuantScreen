"""
Performance metrics. All functions take plain pandas/numpy inputs so each
one can be checked by hand (or in Excel) with a few numbers.

Conventions:
  daily_returns  fractional daily returns (0.01 = +1%), one per trading day
  equity         portfolio value series that starts from START_VALUE before day 1
"""

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252
RISK_FREE = 0.065          # 6.5% a year, approx. Indian 10-year government bond yield
START_VALUE = 100.0


def equity_curve(daily_returns: pd.Series, start_value: float = START_VALUE) -> pd.Series:
    return start_value * (1 + daily_returns).cumprod()


def total_return(daily_returns: pd.Series) -> float:
    """(Final value - Initial) / Initial."""
    return float((1 + daily_returns).prod() - 1)


def annualised_return(daily_returns: pd.Series) -> float:
    """(1 + total)^(252 / trading_days) - 1."""
    n = len(daily_returns)
    if n == 0:
        return 0.0
    return float((1 + total_return(daily_returns)) ** (TRADING_DAYS / n) - 1)


def sharpe_ratio(daily_returns: pd.Series, risk_free: float = RISK_FREE) -> float:
    """(mean daily return - daily risk-free) / std of daily returns x sqrt(252).
    NaN if returns never vary (std = 0)."""
    r = pd.Series(daily_returns, dtype=float)
    sd = r.std(ddof=1)
    if len(r) < 2 or not sd or math.isnan(sd):
        return float("nan")
    return float((r.mean() - risk_free / TRADING_DAYS) / sd * math.sqrt(TRADING_DAYS))


def max_drawdown(daily_returns: pd.Series) -> float:
    """Largest peak-to-trough fall in portfolio value, as a positive fraction.
    The starting value counts as a peak."""
    values = np.concatenate([[START_VALUE], equity_curve(daily_returns).to_numpy()])
    peaks = np.maximum.accumulate(values)
    return float(((peaks - values) / peaks).max())


def win_rate(trade_returns) -> float:
    """Share of trades with return > 0. NaN when there are no trades."""
    r = np.asarray(trade_returns, dtype=float)
    return float((r > 0).mean()) if len(r) else float("nan")


def summarize(daily_returns: pd.Series, trade_returns=None) -> dict:
    """All headline metrics, in percent where it reads naturally."""
    out = {
        "Total return %": total_return(daily_returns) * 100,
        "Annualised return %": annualised_return(daily_returns) * 100,
        "Sharpe ratio": sharpe_ratio(daily_returns),
        "Max drawdown %": max_drawdown(daily_returns) * 100,
    }
    if trade_returns is not None:
        tr = np.asarray(trade_returns, dtype=float)
        out.update({
            "Trades": int(len(tr)),
            "Win rate %": win_rate(tr) * 100,
            "Avg trade return %": float(tr.mean() * 100) if len(tr) else float("nan"),
        })
    return out
