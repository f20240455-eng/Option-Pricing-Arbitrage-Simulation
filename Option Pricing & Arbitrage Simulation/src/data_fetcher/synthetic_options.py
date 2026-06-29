"""
Generates a synthetic option chain using the Black–Scholes–Merton model.

This is used because full historical NSE option chain data is not
freely available. The generated data provides consistent call and put
prices required for the backtest and convergence analysis.

Inputs:
- Daily stock prices with realized volatility
- Daily risk-free rate
- Project configuration (strike spacing, expiry rules)

Output:
- DataFrame containing synthetic call/put prices with bid–ask spreads
"""

import numpy as np
import pandas as pd
from datetime import timedelta
from scipy.stats import norm


class SyntheticOptionChainGenerator:
    """Creates synthetic call and put prices for each trading day."""

    def __init__(self, stock_df: pd.DataFrame, rf_df: pd.DataFrame, config: dict):
        self.stock_df = stock_df.copy()
        self.rf_df = rf_df.copy()
        self.config = config

        # Ensure datetime format is consistent and timezone-naive
        self.stock_df["date"] = pd.to_datetime(self.stock_df["date"]).dt.tz_localize(None)
        self.rf_df["date"] = pd.to_datetime(self.rf_df["date"]).dt.tz_localize(None)

        # Align stock and risk-free data by date
        self.data = pd.merge(self.stock_df, self.rf_df, on="date", how="inner")

    def _bsm_price(self, S, K, T, r, sigma, opt_type):
        """Black–Scholes price for European call or put."""
        if T <= 0:
            return max(S - K, 0) if opt_type == "call" else max(K - S, 0)

        if sigma <= 0:
            discounted = K * np.exp(-r * T)
            return max(S - discounted, 0) if opt_type == "call" else max(discounted - S, 0)

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if opt_type == "call":
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def _strike_list(self, spot):
        step = self.config["trading"]["strike_rounding"]
        low = int((spot * 0.90) // step * step)
        high = int((spot * 1.10) // step * step)
        return list(range(low, high + step, step))

    def _expiry_date(self, current_date):
        target_days = self.config["trading"]["target_expiry_days"]
        approx = current_date + timedelta(days=target_days)
        offset = (3 - approx.weekday()) % 7  # Thursday
        return approx + timedelta(days=offset)

    def _add_spread(self, mid):
        if mid < 10:
            pct = 0.03
        elif mid < 50:
            pct = 0.015
        else:
            pct = 0.01

        spread = mid * pct / 2
        bid = max(mid - spread, 0.05)
        ask = mid + spread

        bid = round(bid * 20) / 20
        ask = round(ask * 20) / 20
        return bid, ask

    def _synthetic_oi_vol(self, moneyness, days_to_exp):
        base = 5000
        m_factor = np.exp(-5 * (moneyness - 1)**2)

        if days_to_exp < 7:
            t_factor = 1.5
        elif days_to_exp > 45:
            t_factor = 0.5
        else:
            t_factor = 1.0

        oi = int(base * m_factor * t_factor * np.random.uniform(0.8, 1.2))
        vol = int(oi * 0.05 * np.random.uniform(0.5, 2.0))

        return max(oi, 50), max(vol, 10)

    def generate(self):
        records = []

        for _, row in self.data.iterrows():
            S = row["adj_close"]
            sigma = row["realized_vol_annual"]
            r = row["rate_annual"]

            if pd.isna(sigma):
                continue

            date = row["date"]
            expiry = self._expiry_date(date)
            dte = (expiry - date).days
            T = dte / 365

            strikes = self._strike_list(S)

            for K in strikes:
                m = S / K

                call_mid = self._bsm_price(S, K, T, r, sigma, "call")
                put_mid = self._bsm_price(S, K, T, r, sigma, "put")

                call_bid, call_ask = self._add_spread(call_mid)
                put_bid, put_ask = self._add_spread(put_mid)

                oi, vol = self._synthetic_oi_vol(m, dte)

                records.append({
                    "date": date,
                    "expiry": expiry,
                    "strike": K,
                    "days_to_expiry": dte,
                    "stock_price": S,
                    "call_bid": call_bid,
                    "call_ask": call_ask,
                    "call_mid": call_mid,
                    "put_bid": put_bid,
                    "put_ask": put_ask,
                    "put_mid": put_mid,
                    "implied_vol": sigma,
                    "open_interest": oi,
                    "volume": vol,
                    "risk_free_rate": r,
                    "moneyness": m
                })

        df = pd.DataFrame(records)
        df["call_spread_pct"] = (df["call_ask"] - df["call_bid"]) / df["call_mid"] * 100
        df["put_spread_pct"] = (df["put_ask"] - df["put_bid"]) / df["put_mid"] * 100
        return df


def generate_synthetic_options(stock_df, rf_df, config):
    generator = SyntheticOptionChainGenerator(stock_df, rf_df, config)
    return generator.generate()
