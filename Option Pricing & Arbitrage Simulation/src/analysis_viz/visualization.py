# src/analysis_viz/visualization.py
# Strike Squad - FIN F311
# Visualization utilities for reporting
# Student ID: 2024A1PS0271P

"""
Visualization utilities used by the final analysis notebook.

This module provides functions to create and save the standard set of
figures used in the project:
- Synthetic vs Actual price comparison
- Distribution of daily P&L
- Cumulative P&L curve
- Drawdown chart

Design principles
- Defensive: check for required columns and provide informative errors
- Robust: accept multiple plausible column names for daily P&L
- Reproducible: always save high-resolution PNGs to an output directory
- Minimal external dependencies: uses matplotlib only for plotting
"""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def _ensure_trade_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the dataframe has a datetime 'trade_date' column and is sorted.
    """
    if "trade_date" not in df.columns:
        raise ValueError("Ledger must contain a 'trade_date' column.")
    if not pd.api.types.is_datetime64_any_dtype(df["trade_date"]):
        df = df.copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df = df.sort_values("trade_date").reset_index(drop=True)
    return df


def _pick_daily_pnl_column(df: pd.DataFrame) -> str:
    """
    Determine which column to use for daily P&L.

    Acceptable column names (in order of preference):
    - daily_pnl_total
    - daily_pnl
    - daily_pnl_per_contract
    - daily_pnl_per (fallback)
    """
    candidates = ["daily_pnl_total", "daily_pnl", "daily_pnl_per_contract", "daily_pnl_per"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        "Ledger must contain a daily P&L column. "
        "Expected one of: " + ", ".join(candidates)
    )


def _safe_save(fig, path: Path) -> str:
    """
    Save a matplotlib figure to path and return the string path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_price_comparison(
    ledger: pd.DataFrame,
    output_dir: str,
    filename: str = "price_comparison_271P.png",
    figsize: tuple = (12, 5),
) -> str:
    """
    Plot synthetic vs actual ATM call prices over time and save the figure.

    Returns the saved file path.
    """
    df = _ensure_trade_date(ledger)
    required = ["actual_price", "synthetic_price"]
    if not any(col in df.columns for col in required):
        # Some versions use call_mid instead of actual_price
        if "call_mid" in df.columns:
            df = df.copy()
            df["actual_price"] = df["call_mid"]
        else:
            raise ValueError("Ledger must contain 'synthetic_price' and 'call_mid' or 'actual_price'.")

    if "synthetic_price" not in df.columns:
        raise ValueError("Ledger must contain 'synthetic_price' column.")

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df["trade_date"], df["actual_price"], label="Actual (market mid)", linewidth=1.25)
    ax.plot(df["trade_date"], df["synthetic_price"], label="Synthetic (model)", linewidth=1.25, linestyle="--")
    ax.set_title("Synthetic vs Actual ATM Call Prices")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (₹)")
    ax.legend()
    ax.grid(alpha=0.25)

    out = Path(output_dir) / filename
    return _safe_save(fig, out)


def plot_daily_pnl_distribution(
    ledger: pd.DataFrame,
    output_dir: str,
    filename: str = "daily_pnl_distribution_271P.png",
    figsize: tuple = (10, 5),
    bins: int = 40,
) -> str:
    """
    Plot histogram of daily P&L and save the figure.
    """
    df = _ensure_trade_date(ledger)
    pnl_col = _pick_daily_pnl_column(df)

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(df[pnl_col].dropna(), bins=bins, edgecolor="black")
    ax.set_title("Distribution of Daily P&L")
    ax.set_xlabel("Daily P&L (₹)")
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.25)

    out = Path(output_dir) / filename
    return _safe_save(fig, out)


def plot_cumulative_pnl(
    ledger: pd.DataFrame,
    output_dir: str,
    filename: str = "cumulative_pnl_271P.png",
    figsize: tuple = (12, 5),
) -> str:
    """
    Plot cumulative P&L over time and save the figure.
    """
    df = _ensure_trade_date(ledger)
    if "cumulative_pnl" not in df.columns:
        # compute cumulative if missing and daily exists
        if any(c in df.columns for c in ["daily_pnl_total", "daily_pnl", "daily_pnl_per_contract"]):
            pnl_col = _pick_daily_pnl_column(df)
            df = df.copy()
            df["cumulative_pnl"] = df[pnl_col].cumsum()
        else:
            raise ValueError("Ledger must contain 'cumulative_pnl' or a daily pnl column.")

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df["trade_date"], df["cumulative_pnl"], linewidth=1.5)
    ax.set_title("Cumulative P&L Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative P&L (₹)")
    ax.grid(alpha=0.25)

    out = Path(output_dir) / filename
    return _safe_save(fig, out)


def plot_drawdown(
    ledger: pd.DataFrame,
    output_dir: str,
    filename: str = "drawdown_271P.png",
    figsize: tuple = (12, 5),
) -> str:
    """
    Plot drawdown curve derived from cumulative P&L and save the figure.
    """
    df = _ensure_trade_date(ledger)
    if "cumulative_pnl" not in df.columns:
        # compute cumulative if missing
        pnl_col = _pick_daily_pnl_column(df)
        df = df.copy()
        df["cumulative_pnl"] = df[pnl_col].cumsum()

    curve = df["cumulative_pnl"]
    running_max = curve.cummax()
    drawdown = curve - running_max

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df["trade_date"], drawdown, color="tab:red", linewidth=1.25)
    ax.set_title("Drawdown Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (₹)")
    ax.grid(alpha=0.25)

    out = Path(output_dir) / filename
    return _safe_save(fig, out)


def generate_all_plots(
    ledger: pd.DataFrame,
    output_dir: str = "../slides/figures",
    return_paths: bool = True,
) -> Optional[List[str]]:
    """
    Generate the full set of plots and save them in output_dir.

    Returns a list of saved file paths when return_paths=True.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    saved.append(plot_price_comparison(ledger, output_dir=out_dir))
    saved.append(plot_daily_pnl_distribution(ledger, output_dir=out_dir))
    saved.append(plot_cumulative_pnl(ledger, output_dir=out_dir))
    saved.append(plot_drawdown(ledger, output_dir=out_dir))

    if return_paths:
        return saved
    return None
