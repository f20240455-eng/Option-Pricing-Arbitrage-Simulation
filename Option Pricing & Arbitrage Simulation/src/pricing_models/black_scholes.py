# Strike Squad - FIN F311
# Black-Scholes-Merton Pricing Model (European Options)
#
# This module implements the standard Black-Scholes-Merton (BSM)
# model for pricing European call and put options. 
import numpy as np
from scipy.stats import norm
from typing import Literal


class BlackScholesModel:
    """
    Black-Scholes-Merton model implementation for European options.
    The assumptions include:
    - constant volatility,
    - lognormal price evolution,
    - continuous compounding,
    - no transaction costs,
    - European-style exercise.
    """

    def __init__(self):
        # No parameters are stored inside the model.
        pass

    # ---------------------------------------------------------
    # Supporting Variables: d1 and d2
    # ---------------------------------------------------------
    @staticmethod
    def calculate_d1(stock_price: float,
                     strike_price: float,
                     time_to_expiry: float,
                     risk_free_rate: float,
                     volatility: float,
                     dividend_yield: float = 0.0) -> float:
        """
        Computes d1 based on standard Black-Scholes formulation.

        d1 = [ ln(S/K) + (r - q + 0.5*σ²)*T ] / ( σ * sqrt(T) )

        If T is zero or negative, return 0. This avoids division issues.
        """

        if time_to_expiry <= 0:
            return 0.0

        numerator = np.log(stock_price / strike_price) \
                    + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_to_expiry
        denominator = volatility * np.sqrt(time_to_expiry)

        return numerator / denominator

    @staticmethod
    def calculate_d2(d1: float,
                     volatility: float,
                     time_to_expiry: float) -> float:
        """
        Computes d2 = d1 - σ√T.
        If T is zero, return 0.
        """

        if time_to_expiry <= 0:
            return 0.0

        return d1 - volatility * np.sqrt(time_to_expiry)

    # ---------------------------------------------------------
    # Pricing Functions
    # ---------------------------------------------------------
    def price_call(self,
                   stock_price: float,
                   strike_price: float,
                   time_to_expiry: float,
                   risk_free_rate: float,
                   volatility: float,
                   dividend_yield: float = 0.0) -> float:
        """
        European call price using:

        C = S * e^(-qT) * N(d1)  –  K * e^(-rT) * N(d2)
        """

        # If option has already expired
        if time_to_expiry <= 0:
            return max(stock_price - strike_price, 0.0)

        # Zero-volatility edge case
        if volatility == 0:
            intrinsic = stock_price - strike_price * np.exp(-risk_free_rate * time_to_expiry)
            return max(intrinsic, 0.0)

        d1 = self.calculate_d1(stock_price, strike_price, time_to_expiry,
                               risk_free_rate, volatility, dividend_yield)
        d2 = self.calculate_d2(d1, volatility, time_to_expiry)

        term_1 = stock_price * np.exp(-dividend_yield * time_to_expiry) * norm.cdf(d1)
        term_2 = strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)

        return term_1 - term_2

    def price_put(self,
                  stock_price: float,
                  strike_price: float,
                  time_to_expiry: float,
                  risk_free_rate: float,
                  volatility: float,
                  dividend_yield: float = 0.0) -> float:
        """
        European put price using:

        P = K * e^(-rT) * N(-d2)  –  S * e^(-qT) * N(-d1)
        """

        if time_to_expiry <= 0:
            return max(strike_price - stock_price, 0.0)

        if volatility == 0:
            intrinsic = strike_price * np.exp(-risk_free_rate * time_to_expiry) - stock_price
            return max(intrinsic, 0.0)

        d1 = self.calculate_d1(stock_price, strike_price, time_to_expiry,
                               risk_free_rate, volatility, dividend_yield)
        d2 = self.calculate_d2(d1, volatility, time_to_expiry)

        term_1 = strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2)
        term_2 = stock_price * np.exp(-dividend_yield * time_to_expiry) * norm.cdf(-d1)

        return term_1 - term_2

    # ---------------------------------------------------------
    # Greeks (Selected)
    # ---------------------------------------------------------
    def calculate_delta(self,
                        stock_price: float,
                        strike_price: float,
                        time_to_expiry: float,
                        risk_free_rate: float,
                        volatility: float,
                        option_type: Literal["call", "put"],
                        dividend_yield: float = 0.0) -> float:
        """
        Computes the option's Delta.

        For calls:
            Δ = e^(-qT) * N(d1)
        For puts:
            Δ = -e^(-qT) * N(-d1)
        """

        if time_to_expiry <= 0:
            if option_type == "call":
                return 1.0 if stock_price > strike_price else 0.0
            else:
                return -1.0 if stock_price < strike_price else 0.0

        d1 = self.calculate_d1(stock_price, strike_price, time_to_expiry,
                               risk_free_rate, volatility, dividend_yield)

        if option_type == "call":
            return np.exp(-dividend_yield * time_to_expiry) * norm.cdf(d1)
        else:
            return -np.exp(-dividend_yield * time_to_expiry) * norm.cdf(-d1)

    def calculate_gamma(self,
                        stock_price: float,
                        strike_price: float,
                        time_to_expiry: float,
                        risk_free_rate: float,
                        volatility: float,
                        dividend_yield: float = 0.0) -> float:
        """
        Computes the option's Gamma:

        Γ = [ e^(-qT) * φ(d1) ] / [ S * σ * sqrt(T) ]
        """

        if time_to_expiry <= 0:
            return 0.0

        d1 = self.calculate_d1(stock_price, strike_price, time_to_expiry,
                               risk_free_rate, volatility, dividend_yield)

        numerator = np.exp(-dividend_yield * time_to_expiry) * norm.pdf(d1)
        denominator = stock_price * volatility * np.sqrt(time_to_expiry)

        return numerator / denominator

    def calculate_vega(self,
                       stock_price: float,
                       strike_price: float,
                       time_to_expiry: float,
                       risk_free_rate: float,
                       volatility: float,
                       dividend_yield: float = 0.0) -> float:
        """
        Vega measures sensitivity to volatility changes.

        ν = S * e^(-qT) * φ(d1) * sqrt(T)
        """

        if time_to_expiry <= 0:
            return 0.0

        d1 = self.calculate_d1(stock_price, strike_price, time_to_expiry,
                               risk_free_rate, volatility, dividend_yield)

        vega_value = stock_price * np.exp(-dividend_yield * time_to_expiry) \
                     * norm.pdf(d1) * np.sqrt(time_to_expiry)

        # Convert to 'per 1%' volatility change
        return vega_value / 100

    # ---------------------------------------------------------
    # Parity Check
    # ---------------------------------------------------------
    def verify_put_call_parity(self,
                               stock_price: float,
                               strike_price: float,
                               time_to_expiry: float,
                               risk_free_rate: float,
                               volatility: float,
                               dividend_yield: float = 0.0) -> dict:
        """
        Checks if put-call parity holds:

        C - P = S*e^(-qT) - K*e^(-rT)
        """

        call_price = self.price_call(stock_price, strike_price, time_to_expiry,
                                     risk_free_rate, volatility, dividend_yield)
        put_price = self.price_put(stock_price, strike_price, time_to_expiry,
                                   risk_free_rate, volatility, dividend_yield)

        left = call_price - put_price
        right = stock_price * np.exp(-dividend_yield * time_to_expiry) \
                - strike_price * np.exp(-risk_free_rate * time_to_expiry)

        difference = abs(left - right)

        return {
            "call_price": call_price,
            "put_price": put_price,
            "left_side": left,
            "right_side": right,
            "difference": difference,
            "parity_holds": difference < 0.01   # tolerance level
        }


# ---------------------------------------------------------
# Convenience function for single-line pricing
# ---------------------------------------------------------
def price_option(stock_price: float,
                 strike_price: float,
                 time_to_expiry: float,
                 risk_free_rate: float,
                 volatility: float,
                 option_type: Literal["call", "put"],
                 dividend_yield: float = 0.0) -> float:
    """
    Simple wrapper to price either a call or a put.
    """

    bsm = BlackScholesModel()

    if option_type == "call":
        return bsm.price_call(stock_price, strike_price, time_to_expiry,
                              risk_free_rate, volatility, dividend_yield)
    else:
        return bsm.price_put(stock_price, strike_price, time_to_expiry,
                             risk_free_rate, volatility, dividend_yield)
