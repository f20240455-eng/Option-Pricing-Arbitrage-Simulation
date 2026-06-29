# src/preprocessing/atm_selector.py
# FIN F311 - Strike Squad
# ATM Option Selection Module
# Student ID: 2024A1PS0271P

"""
ATM Option Selection Module

This module constructs the ATM (At-The-Money) option lookup table.  
For each trading day in the analysis period, the goal is to identify  
the most suitable option contract to be used in the backtest.

Selection is based on:
1. Strike closest to the underlying stock price.
2. Expiry closest to a target maturity (typically 30 days).
3. Liquidity filters (open interest, volume, spread, minimum premium).

The output of this module becomes the core input for the backtest 
engine, since the strategy requires a clean and consistent mapping  
from each trade date to a single option contract (call and put).
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple

def _pick_atm_strike(stock_price: float, strikes: list) -> float:
    """
    Pick the ATM strike given a stock price and list of strikes.
    Used for unit tests.

    Returns the strike with minimum absolute distance to stock_price.
    """
    import numpy as np

    if strikes is None or len(strikes) == 0:
        return np.nan

    strikes = np.array(strikes, dtype=float)
    idx = np.argmin(np.abs(strikes - stock_price))
    return float(strikes[idx])

class ATMOptionSelector:
    """
    Selects ATM options based on pricing, expiry, and liquidity rules.

    Parameters:
    -----------
    stock_data : pd.DataFrame
        Must include [date, adj_close].

    option_chain : pd.DataFrame
        Synthetic option chain created earlier.
        Must include standard columns:
        [date, expiry, strike, days_to_expiry, call_bid, call_ask,
         call_mid, put_bid, put_ask, put_mid, open_interest, volume,
         call_spread_pct, put_spread_pct, implied_vol, risk_free_rate]

    config : dict
        Configuration dictionary loaded from config.yaml.
    """

    def __init__(self, stock_data: pd.DataFrame, option_chain: pd.DataFrame, config: dict):

        self.stock_data = stock_data.copy()
        self.option_chain = option_chain.copy()
        self.config = config

        # Convert date fields
        self.stock_data["date"] = pd.to_datetime(self.stock_data["date"])
        self.option_chain["date"] = pd.to_datetime(self.option_chain["date"])

        # Configuration parameters
        self.target_dte = config["trading"]["target_expiry_days"]
        self.dte_tolerance = config["trading"]["expiry_tolerance_days"]
        self.max_spread_pct = config["liquidity"]["max_bid_ask_spread_pct"]
        self.min_oi = config["liquidity"]["min_open_interest"]
        self.min_volume = config["liquidity"]["min_daily_volume"]
        self.min_premium = config["liquidity"]["min_option_premium"]

    def _select_atm_strike(self, spot_price: float, strikes: list) -> Optional[float]:
        """
        Identifies the strike closest to the underlying stock price.

        Tie-breaking rule:
        - If two strikes are equally close, choose the lower strike.
          This avoids selecting artificially expensive contracts.

        Returns
        -------
        float or None
        """
        if not strikes:
            return None

        distances = [abs(k - spot_price) for k in strikes]
        min_dist = min(distances)

        closest = [k for k, d in zip(strikes, distances) if d == min_dist]

        return min(closest)  # lower strike if tie


    def _select_expiry(self, current_date: pd.Timestamp, expiries: list) -> Optional[pd.Timestamp]:
        """
        Selects an expiry closest to target maturity.

        Rules:
        1. Expiry must be in the future.
        2. Distance to target_dte must be within tolerance.
        3. If multiple expiries qualify, select the later one.
        """

        if not expiries:
            return None

        future_exp = [e for e in expiries if e > current_date]
        if not future_exp:
            return None

        dte = [(e, (e - current_date).days) for e in future_exp]

        # Filter based on tolerance
        valid = [(e, d) for e, d in dte if abs(d - self.target_dte) <= self.dte_tolerance]
        if not valid:
            return None

        # Choose expiry closest to target
        closest = min(valid, key=lambda x: abs(x[1] - self.target_dte))
        diff = abs(closest[1] - self.target_dte)

        candidates = [e for e, d in valid if abs(d - self.target_dte) == diff]

        return max(candidates)  # later expiry among ties


    def _check_liquidity(self, row: pd.Series, option_type: str) -> Tuple[bool, str]:
        """
        Applies liquidity constraints for either call or put.
        """

        # Open interest
        if row["open_interest"] < self.min_oi:
            return False, "Insufficient open interest"

        # Volume
        if row["volume"] < self.min_volume:
            return False, "Insufficient trading volume"

        # Bid-ask spread
        spread_col = f"{option_type}_spread_pct"
        if row[spread_col] > self.max_spread_pct:
            return False, "Wide bid-ask spread"

        # Minimum premium
        mid_col = f"{option_type}_mid"
        if row[mid_col] < self.min_premium:
            return False, "Premium too low"

        return True, ""


    def build_lookup(self) -> pd.DataFrame:
        """
        Constructs a complete ATM lookup table.

        For each stock trading date:
        - Identify ATM strike
        - Identify eligible expiry
        - Validate liquidity for both call and put
        - Record contract information

        Returns
        -------
        pd.DataFrame
            Final ATM table with all daily selections.
        """

        records = []
        skipped_missing_data = 0

        for _, row in self.stock_data.iterrows():
            date = row["date"]
            spot = row["adj_close"]

            daily_options = self.option_chain[self.option_chain["date"] == date]
            if daily_options.empty:
                skipped_missing_data += 1
                continue

            # Step 1: ATM strike
            strikes = sorted(daily_options["strike"].unique().tolist())
            atm_strike = self._select_atm_strike(spot, strikes)
            if atm_strike is None:
                continue

            atm_slice = daily_options[daily_options["strike"] == atm_strike]

            # Step 2: Expiry selection
            expiries = sorted(atm_slice["expiry"].unique().tolist())
            expiry = self._select_expiry(date, expiries)
            if expiry is None:
                continue

            contract = atm_slice[atm_slice["expiry"] == expiry]
            if contract.empty:
                continue

            contract = contract.iloc[0]

            # Step 3: Liquidity checks
            call_ok, call_reason = self._check_liquidity(contract, "call")
            put_ok, put_reason = self._check_liquidity(contract, "put")

            liquid = call_ok and put_ok
            skip_reason = ""
            if not liquid:
                skip_reason = ", ".join([r for r in [call_reason, put_reason] if r])

            # Record
            records.append({
                "trade_date": date,
                "expiry_date": expiry,
                "days_to_expiry": contract["days_to_expiry"],
                "stock_price": spot,
                "atm_strike": atm_strike,
                "call_bid": contract["call_bid"],
                "call_ask": contract["call_ask"],
                "call_mid": contract["call_mid"],
                "put_bid": contract["put_bid"],
                "put_ask": contract["put_ask"],
                "put_mid": contract["put_mid"],
                "moneyness": spot / atm_strike,
                "implied_vol": contract["implied_vol"],
                "open_interest": contract["open_interest"],
                "volume": contract["volume"],
                "risk_free_rate": contract["risk_free_rate"],
                "call_spread_pct": contract["call_spread_pct"],
                "put_spread_pct": contract["put_spread_pct"],
                "liquid": liquid,
                "skip_reason": skip_reason
            })

        return pd.DataFrame(records)


    def summarize_skip_reasons(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Summarises the reasons for skipping trades based on liquidity.
        """

        skipped = df[df["liquid"] == False]
        if skipped.empty:
            return pd.DataFrame()

        reasons = {}

        for r in skipped["skip_reason"]:
            if not r:
                continue
            parts = [x.strip() for x in r.split(",")]
            for p in parts:
                reasons[p] = reasons.get(p, 0) + 1

        out = pd.DataFrame({
            "reason": list(reasons.keys()),
            "count": list(reasons.values())
        }).sort_values("count", ascending=False)

        return out


def build_atm_lookup(stock_df: pd.DataFrame, option_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Convenience function to build the ATM lookup table.
    """

    selector = ATMOptionSelector(stock_df, option_df, config)
    atm = selector.build_lookup()
    selector.summarize_skip_reasons(atm)
    return atm
