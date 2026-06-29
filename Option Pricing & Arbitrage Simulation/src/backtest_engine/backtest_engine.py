# src/backtest_engine/backtest_engine.py
# Strike Squad - FIN F311
# Robust Backtest Engine (final, compatible, defensive)
# Student ID: 2024A1PS0271P

"""
Backtest Engine

Purpose
-------
Produce a clean, well-formed backtest ledger comparing model (synthetic)
ATM call prices against observed market mid prices. This implementation
is defensive (accepts multiple plausible input schemas), non-invasive
(does not change other files), and backward-compatible with previous
notebook usage patterns.

Key design choices
------------------
- The constructor accepts an optional atm_lookup to support both:
    engine = BacktestEngine(atm_lookup)  or
    engine = BacktestEngine(); ledger = engine.run_backtest(atm_lookup)
- The run_backtest method never mutates the provided atm dataframe.
- Column names produced are compatible with downstream modules:
    'trade_date', 'stock_price', 'atm_strike', 'days_to_expiry',
    'implied_vol', 'risk_free_rate', 'actual_price', 'synthetic_price',
    'daily_pnl_total', 'cumulative_pnl'
- Defensive pricing: attempts common BSM method signatures; falls back
  to positional call if required.
- Optional economics: position sizing, contract multiplier, entry cost,
  and slippage parameters (set to zero by default).
- Optional CSV save with explicit output_path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Any
from pathlib import Path
import math
import warnings

import pandas as pd
import numpy as np

from src.pricing_models.black_scholes import BlackScholesModel


@dataclass
class DailyRecord:
    trade_date: pd.Timestamp
    stock_price: float
    atm_strike: float
    days_to_expiry: int
    implied_vol: float
    risk_free_rate: float
    actual_price: float
    synthetic_price: float
    daily_pnl_total: float
    cumulative_pnl: float


class BacktestEngine:
    """
    Robust Backtest Engine

    Parameters
    ----------
    atm_lookup : pandas.DataFrame or None
        Optional preloaded ATM lookup. If provided, run_backtest() may be
        called without arguments.
    position_size : float
        Number of option contracts taken per day (default 1.0).
    contract_multiplier : float
        Multiplier per contract (e.g., lot size). Default 1.0 for per-contract P&L.
    entry_cost_pct : float
        Proportional transaction / commission cost applied to actual price (decimal).
    slippage_pct : float
        Proportional slippage applied to execution price (decimal).
    """

    def __init__(
        self,
        atm_lookup: Optional[pd.DataFrame] = None,
        position_size: float = 1.0,
        contract_multiplier: float = 1.0,
        entry_cost_pct: float = 0.0,
        slippage_pct: float = 0.0,
    ) -> None:
        self._atm_lookup = atm_lookup
        self.position_size = float(position_size)
        self.contract_multiplier = float(contract_multiplier)
        self.entry_cost_pct = float(entry_cost_pct)
        self.slippage_pct = float(slippage_pct)
        self._bsm = BlackScholesModel()

    def _resolve_actual_price_column(self, df: pd.DataFrame) -> str:
        """
        Return column name to use as market 'actual' price.
        Prefer 'actual_price' then 'call_mid' then 'call_mid_price' then 'call_mid_px'.
        """
        candidates = ["actual_price", "call_mid", "call_mid_price", "call_mid_px", "call_mid_m"]
        for c in candidates:
            if c in df.columns:
                return c
        raise ValueError("No market mid column found in atm_lookup. Expected one of: "
                         + ", ".join(candidates))

    def _ensure_required_columns(self, df: pd.DataFrame) -> None:
        """
        Validate minimal ATM lookup schema; raise informative errors.
        """
        required = ["trade_date", "stock_price", "atm_strike", "days_to_expiry",
                    "implied_vol", "risk_free_rate"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"ATM lookup is missing required columns: {missing}")

    def _coerce_trade_date(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "trade_date" in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df["trade_date"]):
                df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
        return df

    def _price_call_bsm(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Wrapper to call the BlackScholesModel price_call with multiple possible signatures.
        Returns a float price.
        """
        # Ensure valid inputs
        S = float(S)
        K = float(K)
        T = max(float(T), 1e-8)
        r = float(r)
        sigma = float(sigma)

        # Try keyword signature first (most likely)
        try:
            return float(self._bsm.price_call(
                stock_price=S,
                strike_price=K,
                time_to_expiry=T,
                risk_free_rate=r,
                volatility=sigma,
            ))
        except TypeError:
            pass
        # Try alternate keyword names if the implementation uses different param names
        try:
            return float(self._bsm.price_call(
                S, K, T, r, sigma
            ))
        except TypeError:
            pass
        # Generic fallback: try positional call with up to 6 args
        try:
            return float(self._bsm.price_call(S, K, T, r, sigma, 0.0))
        except Exception as exc:
            warnings.warn(f"BSM pricing wrapper failed with error: {exc}. Returning NaN.")
            return float("nan")

    def _apply_trade_costs(self, price: float, is_buy: bool = True) -> float:
        """
        Apply entry cost and slippage to an execution price. is_buy indicates
        whether the trade direction is buy; for simple ATM daily P&L this is symmetric.
        """
        price = float(price)
        if self.entry_cost_pct:
            price = price - math.copysign(price * self.entry_cost_pct, price)  # approximate
        if self.slippage_pct:
            price = price - math.copysign(price * self.slippage_pct, price)
        return price

    def run_backtest(
        self,
        atm_lookup: Optional[pd.DataFrame] = None,
        output_path: Optional[str] = "../data/processed/backtest_results_271P.csv",
        save_csv: bool = True,
    ) -> pd.DataFrame:
        """
        Run the backtest.

        Parameters
        ----------
        atm_lookup : DataFrame or None
            ATM lookup table. If None, the instance must have one from constructor.
        output_path : str or None
            CSV path to save ledger. If None, ledger is not saved.
        save_csv : bool
            Whether to save CSV when output_path is provided.

        Returns
        -------
        ledger_df : pandas.DataFrame
            Ledger with required columns for metrics & visualization.
        """
        df = atm_lookup if atm_lookup is not None else self._atm_lookup
        if df is None:
            raise ValueError("No atm_lookup provided to run_backtest (either pass to method or constructor).")

        df = df.copy()
        df = self._coerce_trade_date(df)
        self._ensure_required_columns(df)

        actual_col = None
        try:
            actual_col = self._resolve_actual_price_column(df)
        except ValueError:
            # allow presence of 'call_mid' later, but raise if absolutely missing
            raise

        # Normalize column names for internal use
        df = df.rename(columns={actual_col: "actual_price", "atm_strike": "atm_strike"})
        # Some datasets use 'atm_strike' vs 'strike'; ensure we have atm_strike
        if "atm_strike" not in df.columns and "strike" in df.columns:
            df = df.rename(columns={"strike": "atm_strike"})

        ledger_records = []
        cumulative = 0.0

        # Iteration order: sorted by trade_date to keep deterministic cumulative pnl
        if "trade_date" in df.columns:
            df = df.sort_values("trade_date").reset_index(drop=True)

        for idx, row in df.iterrows():
            try:
                trade_date = row.get("trade_date", pd.NaT)
                S = float(row["stock_price"])
                K = float(row["atm_strike"])
                dte = int(row["days_to_expiry"])
                T = max(float(dte) / 365.0, 1e-8)
                implied_vol = float(row["implied_vol"])
                r = float(row["risk_free_rate"])
                actual = float(row["actual_price"])
            except Exception as e:
                # skip rows that cannot be parsed but keep a warning
                warnings.warn(f"Skipping row {idx} during parsing: {e}")
                continue

            synthetic = self._price_call_bsm(S, K, T, r, implied_vol)

            # Adjust execution prices for costs/slippage if configured
            exec_actual = self._apply_trade_costs(actual, is_buy=True)
            exec_synthetic = self._apply_trade_costs(synthetic, is_buy=False)

            # P&L per contract (synthetic - actual). Multiply by position and contract size.
            daily_pnl_per_contract = (exec_synthetic - exec_actual)
            daily_pnl_total = daily_pnl_per_contract * self.position_size * self.contract_multiplier

            cumulative += daily_pnl_total

            rec = DailyRecord(
                trade_date=pd.to_datetime(trade_date),
                stock_price=S,
                atm_strike=K,
                days_to_expiry=dte,
                implied_vol=implied_vol,
                risk_free_rate=r,
                actual_price=actual,
                synthetic_price=synthetic,
                daily_pnl_total=float(daily_pnl_total),
                cumulative_pnl=float(cumulative),
            )
            ledger_records.append(rec.__dict__)

        ledger_df = pd.DataFrame(ledger_records)

        # Add safety columns if missing (keeps downstream code stable)
        if "daily_pnl" not in ledger_df.columns and "daily_pnl_total" in ledger_df.columns:
            ledger_df["daily_pnl"] = ledger_df["daily_pnl_total"]

        if save_csv and output_path:
            outp = Path(output_path)
            outp.parent.mkdir(parents=True, exist_ok=True)
            ledger_df.to_csv(outp, index=False)

        return ledger_df


# Example usage (commented; for developer quick reference)
# from src.backtest_engine.backtest_engine import BacktestEngine
# atm = pd.read_csv("../data/processed/atm_lookup_271P.csv")
# engine = BacktestEngine(position_size=1.0, contract_multiplier=1.0, entry_cost_pct=0.001, slippage_pct=0.0005)
# ledger = engine.run_backtest(atm, output_path="../data/processed/backtest_results_271P.csv")
# print(ledger.head())
