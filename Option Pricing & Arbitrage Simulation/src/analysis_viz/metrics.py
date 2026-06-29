# src/analysis_viz/metrics.py
# Strike Squad - FIN F311
# Performance Metrics for Option Backtest
# Student ID: 2024A1PS0271P

"""
Performance Metrics Module

This module provides standard performance metrics for the option
backtest results. It works with the ledger produced by the
BacktestEngine and returns a dictionary summarizing key statistics.

Metrics included:
- Cumulative return
- Annualized return
- Annualized volatility
- Sharpe ratio
- Maximum drawdown
- Hit ratio (percentage of profitable days)
- Mean absolute pricing error (synthetic vs actual)
"""

import pandas as pd
import numpy as np


def compute_cumulative_return(ledger: pd.DataFrame) -> float:
    """
    Compute cumulative return from final cumulative P&L.
    """
    if ledger.empty:
        return np.nan

    final_pnl = ledger["cumulative_pnl"].iloc[-1]
    return final_pnl


def compute_annualized_return(ledger: pd.DataFrame) -> float:
    """
    Annualized return based on average daily P&L.
    """
    if ledger.empty:
        return np.nan

    total_days = len(ledger)
    daily_mean = ledger["daily_pnl_total"].mean()
    annualized = daily_mean * 252
    return annualized


def compute_annualized_volatility(ledger: pd.DataFrame) -> float:
    """
    Annualized volatility of daily P&L.
    """
    if ledger.empty:
        return np.nan

    daily_std = ledger["daily_pnl_total"].std()
    annualized_vol = daily_std * np.sqrt(252)
    return annualized_vol


def compute_sharpe_ratio(ledger: pd.DataFrame, risk_free_rate: float = 0.0) -> float:
    """
    Compute Sharpe ratio using daily P&L.

    Sharpe = (mean_daily_return - daily_rfr) / std_daily_return
    """
    if ledger.empty:
        return np.nan

    daily_rfr = risk_free_rate / 252
    daily_mean = ledger["daily_pnl_total"].mean()
    daily_std = ledger["daily_pnl_total"].std()

    if daily_std == 0:
        return np.nan

    sharpe = (daily_mean - daily_rfr) / daily_std * np.sqrt(252)
    return sharpe


def compute_max_drawdown(ledger: pd.DataFrame) -> float:
    """
    Compute maximum drawdown from cumulative P&L path.
    """
    if ledger.empty:
        return np.nan

    cum_pnl = ledger["cumulative_pnl"]
    running_max = cum_pnl.cummax()
    drawdowns = cum_pnl - running_max
    max_dd = drawdowns.min()

    return max_dd


def compute_hit_ratio(ledger: pd.DataFrame) -> float:
    """
    Hit ratio: percentage of days with positive daily P&L.
    """
    if ledger.empty:
        return np.nan

    positive_days = (ledger["daily_pnl_total"] > 0).sum()
    return positive_days / len(ledger)


def compute_mean_abs_error(ledger: pd.DataFrame) -> float:
    """
    Mean absolute pricing error: |synthetic_price - actual_price|.
    """
    if ledger.empty:
        return np.nan

    mae = np.abs(ledger["synthetic_price"] - ledger["actual_price"]).mean()
    return mae


def compute_all_metrics(ledger: pd.DataFrame) -> dict:
    """
    Compute all performance metrics and return as a dictionary.
    """
    metrics = {
        "cumulative_return": compute_cumulative_return(ledger),
        "annualized_return": compute_annualized_return(ledger),
        "annualized_volatility": compute_annualized_volatility(ledger),
        "sharpe_ratio": compute_sharpe_ratio(ledger),
        "max_drawdown": compute_max_drawdown(ledger),
        "hit_ratio": compute_hit_ratio(ledger),
        "mean_abs_pricing_error": compute_mean_abs_error(ledger)
    }

    return metrics
