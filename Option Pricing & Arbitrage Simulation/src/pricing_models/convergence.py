# src/pricing_models/convergence.py
# Strike Squad - FIN F311
# Convergence Analysis Module — ultra-defensive, production-ready
# Student ID: 2024A1PS0271P

"""
Convergence Analysis Module - Robust version

Design goals:
- Import pricing model implementations defensively (many possible APIs)
- Normalize datetimes and avoid timezone merge errors
- Never crash on NaNs; provide informative summary when data is missing
- Provide a single callable for full convergence analysis and plotting
- Save CSV output and plots with safe filesystem handling
- Minimal external dependencies (only numpy, pandas, matplotlib, logging)

Usage:
- Place this file at src/pricing_models/convergence.py
- Call run_full_convergence_analysis(stock_df, atm_df, dividends_df, config, output_path)
"""

from __future__ import annotations

import time
import math
import logging
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path

import numpy as np
import pandas as pd

# matplotlib imported lazily inside plotting function to avoid heavy import when not needed

logger = logging.getLogger("convergence")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


def _normalize_datetime_series(s: pd.Series) -> pd.Series:
    """Return timezone-naive datetime64[ns] series. Coerces non-datetime to NaT."""
    ser = pd.to_datetime(s, errors="coerce")
    try:
        # if tz-aware, convert to UTC then drop tz
        if pd.api.types.is_datetime64tz_dtype(ser):
            return ser.dt.tz_convert("UTC").dt.tz_localize(None)
    except Exception:
        # some pandas versions might have different behavior; fallback below
        pass
    # ensure dtype is datetime64[ns] (naive)
    return ser.dt.tz_localize(None) if getattr(ser.dt, "tz", None) is not None else ser


def _safe_import_binomial() -> Optional[Callable]:
    """
    Attempt to obtain a single callable with signature:
    price_binomial(S, K, T, r, sigma, n, option_type='call', exercise_type='european', q=0.0)
    Returns None if not found; caller must handle this scenario.
    """
    import importlib
    try:
        module = importlib.import_module("src.pricing_models.binomial_tree")
    except Exception as exc:
        logger.warning("Could not import src.pricing_models.binomial_tree: %s", exc)
        return None

    # Try direct function
    price_fn = getattr(module, "price_binomial", None)
    if callable(price_fn):
        def wrapper(S, K, T, r, sigma, n, option_type="call", exercise_type="european", q=0.0):
            try:
                # try to call using named args first
                return float(price_fn(S, K, T, r, sigma, n, option_type=option_type, exercise_type=exercise_type, q=q))
            except TypeError:
                # try positional
                try:
                    return float(price_fn(S, K, T, r, sigma, n))
                except Exception as e:
                    raise
        return wrapper

    # Try class-based BinomialTreeModel
    cls = getattr(module, "BinomialTreeModel", None)
    if cls is not None:
        def wrapper(S, K, T, r, sigma, n, option_type="call", exercise_type="european", q=0.0):
            # try sensible constructor orders and keyword names
            for ctor_try in (
                {"stock_price": S, "strike_price": K, "time_to_expiry": T, "risk_free_rate": r, "volatility": sigma, "n_steps": n, "option_type": option_type, "exercise_type": exercise_type, "dividend_yield": q},
                {"stock_price": S, "strike_price": K, "time_to_expiry": T, "risk_free_rate": r, "volatility": sigma, "n_steps": n, "option_type": option_type},
                {"S": S, "K": K, "T": T, "r": r, "sigma": sigma, "n": n, "option_type": option_type},
            ):
                try:
                    model = cls(**ctor_try)
                    # common method names to extract price
                    for method in ("price", "price_option", "price_binomial", "compute_price"):
                        fn = getattr(model, method, None)
                        if callable(fn):
                            return float(fn())
                    # fallback: try __call__ if implemented
                    if callable(model):
                        return float(model())
                except TypeError:
                    continue
            # Last attempt: positional constructor
            try:
                model = cls(S, K, T, r, sigma, n, option_type)
                if hasattr(model, "price"):
                    return float(model.price())
            except Exception as exc:
                raise RuntimeError(f"Failed to instantiate/price using BinomialTreeModel: {exc}")
        return wrapper

    logger.warning("Neither price_binomial nor BinomialTreeModel found in binomial_tree module.")
    return None


def _safe_import_bsm() -> Callable:
    """
    Attempt to obtain a callable for BSM pricing:
    bsm_price_call(S, K, T, r, sigma, q=0.0)
    If BlackScholesModel is present with price_call/price_put methods, wrap them.
    Otherwise fall back to a local BSM implementation (uses math.erf).
    """
    import importlib
    try:
        module = importlib.import_module("src.pricing_models.black_scholes")
        BSM = getattr(module, "BlackScholesModel", None)
        if BSM is not None:
            instance = BSM()  # assume no-arg constructor
            if hasattr(instance, "price_call") and hasattr(instance, "price_put"):
                def bsm_call(S, K, T, r, sigma, q=0.0):
                    try:
                        return float(instance.price_call(S, K, T, r, sigma, q))
                    except TypeError:
                        # try alternate signature without q
                        try:
                            return float(instance.price_call(S, K, T, r, sigma))
                        except Exception:
                            raise
                def bsm_put(S, K, T, r, sigma, q=0.0):
                    try:
                        return float(instance.price_put(S, K, T, r, sigma, q))
                    except TypeError:
                        try:
                            return float(instance.price_put(S, K, T, r, sigma))
                        except Exception:
                            raise
                return bsm_call, bsm_put
    except Exception as exc:
        logger.info("BlackScholesModel import not available or failed: %s", exc)

    # Fallback lightweight BSM implementation (no scipy)
    def _norm_cdf(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def bsm_call_fallback(S, K, T, r, sigma, q=0.0):
        if T <= 0:
            return max(S - K, 0.0)
        if sigma <= 0:
            # treat as intrinsic discounted
            return max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)

    def bsm_put_fallback(S, K, T, r, sigma, q=0.0):
        if T <= 0:
            return max(K - S, 0.0)
        if sigma <= 0:
            return max(K * math.exp(-r * T) - S * math.exp(-q * T), 0.0)
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * math.exp(-q * T) * _norm_cdf(-d1)

    logger.info("Using internal BSM fallback implementation.")
    return bsm_call_fallback, bsm_put_fallback


def select_convergence_snapshots(
    stock_data: pd.DataFrame,
    atm_lookup: pd.DataFrame,
    dividend_data: Optional[pd.DataFrame] = None,
    n_snapshots: int = 5
) -> List[Dict[str, Any]]:
    """
    Select representative snapshots for convergence analysis.

    This function is defensive:
    - Normalizes dates
    - Tolerates missing columns by using fallbacks
    - Ensures returned snapshots have required numeric fields
    """
    if stock_data is None or atm_lookup is None:
        raise ValueError("stock_data and atm_lookup are required inputs.")

    s = stock_data.copy()
    a = atm_lookup.copy()
    d = pd.DataFrame() if dividend_data is None else dividend_data.copy()

    # Normalize datetimes to timezone-naive
    if "date" in s.columns:
        s["date"] = _normalize_datetime_series(s["date"])
    if "trade_date" in a.columns:
        a["trade_date"] = _normalize_datetime_series(a["trade_date"])
    if "ex_date" in d.columns:
        d["ex_date"] = _normalize_datetime_series(d["ex_date"])

    # Ensure standard column names exist or create fallbacks
    if "stock_price" not in a.columns:
        for alt in ("adj_close", "close", "S", "spot"):
            if alt in a.columns:
                a["stock_price"] = a[alt]
                break
    if "atm_strike" not in a.columns:
        for alt in ("strike", "K"):
            if alt in a.columns:
                a["atm_strike"] = a[alt]
                break

    # Merge realized volatility when available
    merged = a
    if "realized_vol_annual" in s.columns and "date" in s.columns:
        try:
            merged = pd.merge( a, s[["date", "realized_vol_annual"]], left_on="trade_date", right_on="date", how="left" )
        except Exception:
            # final fallback: align by date string
            merged = a.copy()
            merged["realized_vol_annual"] = np.nan
            try:
                s_date_map = s.set_index(s["date"].dt.date)["realized_vol_annual"].to_dict()
                merged["realized_vol_annual"] = merged["trade_date"].dt.date.map(s_date_map)
            except Exception:
                merged["realized_vol_annual"] = np.nan
    else:
        merged = a.copy()
        merged["realized_vol_annual"] = np.nan

    results: List[Dict[str, Any]] = []

    def _safe_row_to_snapshot(row, name):
        # Extract required fields with safe fallbacks
        S = float(row.get("stock_price", np.nan)) if pd.notna(row.get("stock_price", np.nan)) else np.nan
        K = float(row.get("atm_strike", row.get("strike", np.nan))) if pd.notna(row.get("atm_strike", row.get("strike", np.nan))) else np.nan
        dte = int(row.get("days_to_expiry", row.get("DTE", 30))) if pd.notna(row.get("days_to_expiry", row.get("DTE", 30))) else 30
        vol = float(row.get("realized_vol_annual", row.get("implied_vol", 0.25))) if pd.notna(row.get("realized_vol_annual", row.get("implied_vol", np.nan))) else 0.25
        rf = float(row.get("risk_free_rate", 0.06)) if pd.notna(row.get("risk_free_rate", np.nan)) else 0.06
        td = row.get("trade_date", row.get("date", pd.NaT))
        return {
            "scenario": name,
            "trade_date": td,
            "stock_price": S,
            "strike": K,
            "time_to_expiry": float(dte) / 365.0,
            "risk_free_rate": rf,
            "volatility": vol,
            "days_to_expiry": dte
        }

    # Candidate scenarios
    try:
        if merged["realized_vol_annual"].dropna().size > 0:
            low_idx = merged["realized_vol_annual"].idxmin()
            if pd.notna(low_idx):
                results.append(_safe_row_to_snapshot(merged.loc[low_idx], "Low Volatility"))
            high_idx = merged["realized_vol_annual"].idxmax()
            if pd.notna(high_idx):
                results.append(_safe_row_to_snapshot(merged.loc[high_idx], "High Volatility"))
    except Exception:
        logger.info("Skipping vol-based snapshot selection due to data issues.")

    try:
        if "days_to_expiry" in merged.columns:
            near = merged[merged["days_to_expiry"] < 10]
            if len(near) > 0:
                results.append(_safe_row_to_snapshot(near.iloc[len(near) // 2], "Near Expiry"))
            far = merged[merged["days_to_expiry"] > 40]
            if len(far) > 0:
                results.append(_safe_row_to_snapshot(far.iloc[len(far) // 2], "Far Expiry"))
    except Exception:
        logger.info("Skipping DTE-based selection due to data issues.")

    # Ex-dividend snapshot
    if not d.empty and "ex_date" in d.columns:
        try:
            for _, div_row in d.iterrows():
                ex = div_row["ex_date"]
                window = merged[(merged["trade_date"] >= ex - pd.Timedelta(days=5)) & (merged["trade_date"] <= ex + pd.Timedelta(days=5))]
                if len(window) > 0:
                    results.append(_safe_row_to_snapshot(window.iloc[0], "Ex-Dividend"))
                    break
        except Exception:
            logger.info("Skipping ex-dividend selection due to data issues.")

    # If still not enough snapshots, sample evenly across available rows
    if len(results) < n_snapshots and len(merged) > 0:
        needed = n_snapshots - len(results)
        idxs = np.linspace(0, len(merged) - 1, num=needed + 2, dtype=int)[1:-1]
        for i in idxs:
            results.append(_safe_row_to_snapshot(merged.iloc[i], f"Supplementary-{i}"))
            if len(results) >= n_snapshots:
                break

    # Final clean: remove snapshots with NaN stock or strike
    cleaned = [r for r in results if pd.notna(r.get("stock_price")) and pd.notna(r.get("strike"))]
    return cleaned[:n_snapshots]


def analyze_convergence_for_snapshot(
    snapshot: Dict[str, Any],
    steps_list: List[int],
    option_type: str = "call",
    binomial_callable: Optional[Callable] = None,
    bsm_call: Optional[Callable] = None,
    bsm_put: Optional[Callable] = None
) -> pd.DataFrame:
    """
    Compute convergence table for a single snapshot.
    Returns DataFrame with columns:
    ['scenario', 'n_steps', 'binomial_price', 'bsm_price', 'absolute_error', 'relative_error_pct', 'computation_time_sec']
    This function never raises for numeric errors; it records NaN and continues.
    """
    S = float(snapshot.get("stock_price", np.nan))
    K = float(snapshot.get("strike", np.nan))
    T = float(snapshot.get("time_to_expiry", 0.0))
    r = float(snapshot.get("risk_free_rate", 0.0))
    vol = float(snapshot.get("volatility", 0.0))

    # Prepare bsm callables if not provided
    if bsm_call is None or bsm_put is None:
        bsm_call, bsm_put = _safe_import_bsm()

    # Prepare binomial callable if not provided
    if binomial_callable is None:
        binomial_callable = _safe_import_binomial()

    # Compute BSM price safely
    try:
        bsm_price = float(bsm_call(S, K, T, r, vol, 0.0))
    except Exception as e:
        logger.warning("BSM price computation failed for snapshot %s: %s", snapshot.get("scenario"), e)
        bsm_price = float("nan")

    rows = []
    for n in steps_list:
        start = time.time()
        bin_price = float("nan")
        if binomial_callable is not None:
            try:
                bin_price = float(binomial_callable(S, K, T, r, vol, int(n), option_type, "european", 0.0))
                if not np.isfinite(bin_price):
                    bin_price = float("nan")
            except Exception as e:
                logger.debug("Binomial pricing failed at n=%s for snapshot %s: %s", n, snapshot.get("scenario"), e)
                bin_price = float("nan")
        else:
            # no binomial available; continue with NaN
            bin_price = float("nan")
        elapsed = time.time() - start

        abs_err = float("nan")
        rel_err = float("nan")
        try:
            if np.isfinite(bsm_price) and np.isfinite(bin_price):
                abs_err = abs(bin_price - bsm_price)
                rel_err = (abs_err / bsm_price * 100.0) if bsm_price != 0 else float("nan")
        except Exception:
            abs_err = float("nan")
            rel_err = float("nan")

        rows.append({
            "scenario": snapshot.get("scenario", "snapshot"),
            "n_steps": int(n),
            "binomial_price": bin_price,
            "bsm_price": bsm_price,
            "absolute_error": abs_err,
            "relative_error_pct": rel_err,
            "computation_time_sec": elapsed,
            "trade_date": snapshot.get("trade_date"),
            "stock_price": S,
            "strike": K,
            "volatility": vol,
            "days_to_expiry": snapshot.get("days_to_expiry", int(T * 365))
        })

    return pd.DataFrame(rows)


def run_full_convergence_analysis(
    stock_data: pd.DataFrame,
    atm_lookup: pd.DataFrame,
    dividend_data: Optional[pd.DataFrame],
    config: Dict[str, Any],
    output_path: str = "../data/processed/convergence_results_271P.csv"
) -> pd.DataFrame:
    """
    Orchestrate full convergence analysis across snapshots.

    Writes CSV to output_path and returns concatenated results DataFrame.
    This function logs and continues; only raises if no snapshots or no data.
    """
    if config is None or "pricing" not in config:
        raise ValueError("Config must contain 'pricing' section with 'binomial_steps_list' and 'convergence_tolerance'.")

    steps_list = config["pricing"].get("binomial_steps_list", [5, 10, 20, 40, 80, 160])
    tolerance = config["pricing"].get("convergence_tolerance", 0.5)

    snaps = select_convergence_snapshots(stock_data, atm_lookup, dividend_data, n_snapshots=5)
    if not snaps:
        raise RuntimeError("No valid convergence snapshots were found. Check ATM lookup and stock data.")

    logger.info("Selected snapshots: %s", [s["scenario"] for s in snaps])

    # Obtain callables once
    binom_callable = _safe_import_binomial()
    bsm_call, bsm_put = _safe_import_bsm()

    results_list = []
    for snap in snaps:
        df_snap = analyze_convergence_for_snapshot(snap, steps_list, option_type="call", binomial_callable=binom_callable, bsm_call=bsm_call, bsm_put=bsm_put)
        results_list.append(df_snap)

    if not results_list:
        raise RuntimeError("Convergence run produced no results.")

    full = pd.concat(results_list, ignore_index=True)

    # Ensure output directory exists and save CSV
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        full.to_csv(out, index=False)
        logger.info("Saved convergence CSV to: %s", out)
    except Exception as e:
        logger.error("Failed to save convergence CSV: %s", e)

    # Summarize by scenario (defensive: handle NaNs)
    summary_lines = []
    for scenario in full["scenario"].unique():
        sub = full[full["scenario"] == scenario]
        if sub["absolute_error"].dropna().empty:
            summary_lines.append(f"{scenario}: no numeric absolute_error values (binomial probably missing).")
            continue
        min_err = sub["absolute_error"].min()
        best_rows = sub[sub["absolute_error"] == min_err]
        # pick smallest n among best_rows
        best_n = int(best_rows["n_steps"].min())
        summary_lines.append(f"{scenario}: min abs error = {min_err:.6f} at n = {best_n}")
        # tolerance check
        if (sub["absolute_error"] <= tolerance).any():
            conv_n = int(sub[sub["absolute_error"] <= tolerance]["n_steps"].min())
            summary_lines.append(f"  converged within tolerance {tolerance:.2f} at n = {conv_n}")
        else:
            summary_lines.append(f"  did not converge within tolerance {tolerance:.2f}")

    for line in summary_lines:
        logger.info(line)

    return full


def generate_convergence_plots(convergence_results: pd.DataFrame, output_dir: str = "../slides/figures"):
    """
    Save two plots:
    - convergence_all_scenarios_271P.png : binomial_price vs n (log-x) with BSM horiz line
    - convergence_error_271P.png : absolute_error vs n (log-log)
    Function is defensive and continues on plotting errors.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.error("matplotlib is required for plotting: %s", e)
        return

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    scenarios = list(convergence_results["scenario"].unique())

    # Price convergence
    try:
        plt.figure(figsize=(10, 6))
        for scenario in scenarios:
            df = convergence_results[convergence_results["scenario"] == scenario].sort_values("n_steps")
            if df.empty:
                continue
            plt.plot(df["n_steps"], df["binomial_price"], marker="o", linewidth=1.5, label=scenario)
            if "bsm_price" in df.columns:
                bsm_val = df["bsm_price"].dropna().iloc[0] if not df["bsm_price"].dropna().empty else None
                if bsm_val is not None:
                    plt.hlines(bsm_val, xmin=df["n_steps"].min(), xmax=df["n_steps"].max(), linestyles="--", alpha=0.5)
        plt.xscale("log")
        plt.xlabel("Number of Steps (n)")
        plt.ylabel("Option Price (₹)")
        plt.title("Binomial Price Convergence vs BSM")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / "convergence_all_scenarios_271P.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Saved convergence price plot.")
    except Exception as e:
        logger.error("Failed to create price convergence plot: %s", e)

    # Error convergence
    try:
        plt.figure(figsize=(10, 6))
        for scenario in scenarios:
            df = convergence_results[convergence_results["scenario"] == scenario].sort_values("n_steps")
            if df.empty:
                continue
            plt.plot(df["n_steps"], df["absolute_error"], marker="s", linewidth=1.5, label=scenario)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Number of Steps (n)")
        plt.ylabel("Absolute Error (₹)")
        plt.title("Convergence Error vs Number of Steps (log-log)")
        plt.legend()
        plt.grid(which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / "convergence_error_271P.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Saved convergence error plot.")
    except Exception as e:
        logger.error("Failed to create error convergence plot: %s", e)


def recommended_step_count(results_df: pd.DataFrame, threshold: float = 0.50) -> Optional[int]:
    """
    Return the smallest n_steps where absolute_error <= threshold.
    If none found or insufficient numeric data, returns None.
    """
    if results_df is None or "absolute_error" not in results_df.columns or "n_steps" not in results_df.columns:
        return None
    df = results_df.dropna(subset=["absolute_error"])
    if df.empty:
        return None
    filtered = df[df["absolute_error"] <= float(threshold)]
    if filtered.empty:
        return None
    return int(filtered.sort_values("n_steps").iloc[0]["n_steps"])
