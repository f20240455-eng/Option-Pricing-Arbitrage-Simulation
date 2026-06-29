# src/pricing/implied_vol.py
# Strike Squad - FIN F311
# Implied Volatility Calculator
# Student ID: 2024A1PS0271P

"""
Implied Volatility Module

This module provides functions to estimate implied volatility from
observed option prices using the Black–Scholes–Merton model.
The goal is to recover the volatility value that makes the
BSM model price match the market price.

Two numerical approaches are supported:
1. Brent's method (robust bracketed solver)
2. Newton's method (faster when vega is stable)

This file is designed to be simple to understand and integrate
directly with the pricing modules.
"""

import numpy as np
from scipy.optimize import brentq, newton
from typing import Literal, Optional
import warnings

from src.pricing_models.black_scholes import BlackScholesModel


def calculate_implied_volatility(
    market_price: float,
    stock_price: float,
    strike_price: float,
    time_to_expiry: float,
    risk_free_rate: float,
    option_type: Literal['call', 'put'],
    dividend_yield: float = 0.0,
    initial_guess: Optional[float] = None,
    method: str = 'brent',
    max_iterations: int = 100,
    tolerance: float = 1e-6
) -> float:
    """
    Estimate implied volatility from an observed option price.

    Parameters
    ----------
    market_price : float
        Observed market option price (e.g., mid price)
    stock_price : float
        Current stock price
    strike_price : float
        Strike price of the option
    time_to_expiry : float
        Time to expiration in years
    risk_free_rate : float
        Annual risk-free rate
    option_type : {'call', 'put'}
        Type of the option
    dividend_yield : float
        Continuous dividend yield (default = 0.0)
    initial_guess : float
        Starting guess for Newton's method
    method : str
        'brent', 'newton', or 'hybrid'
    max_iterations : int
        Maximum number of iterations allowed
    tolerance : float
        Convergence tolerance

    Returns
    -------
    float
        Estimated implied volatility. Returns NaN when not solvable.
    """

    # Basic validation
    if market_price <= 0:
        warnings.warn("Invalid market price; must be positive.")
        return np.nan
    if time_to_expiry <= 0:
        warnings.warn("Time to expiry must be positive.")
        return np.nan

    bsm = BlackScholesModel()

    # Function whose root we need: model_price(sigma) - observed_price
    def objective(vol):
        if vol <= 0:
            return np.inf
        if option_type == 'call':
            price = bsm.price_call(
                stock_price, strike_price, time_to_expiry,
                risk_free_rate, vol, dividend_yield
            )
        else:
            price = bsm.price_put(
                stock_price, strike_price, time_to_expiry,
                risk_free_rate, vol, dividend_yield
            )
        return price - market_price

    # Simple approximation for initial guess if needed
    if initial_guess is None:
        initial_guess = _initial_iv_guess(
            market_price, stock_price, strike_price, time_to_expiry
        )

    try:
        if method == 'brent':
            # Volatility is usually between 1% and 300%
            return brentq(objective, 0.01, 3.0, maxiter=max_iterations, xtol=tolerance)

        elif method == 'newton':
            # Newton's method requires vega
            def vega(vol):
                return bsm.calculate_vega(
                    stock_price, strike_price, time_to_expiry,
                    risk_free_rate, vol, dividend_yield
                ) / 100  # vega from module is per 1% vol

            return newton(
                func=objective,
                x0=initial_guess,
                fprime=vega,
                maxiter=max_iterations,
                tol=tolerance
            )

        elif method == 'hybrid':
            # Try Newton first, fall back to Brent
            try:
                def vega(vol):
                    return bsm.calculate_vega(
                        stock_price, strike_price, time_to_expiry,
                        risk_free_rate, vol, dividend_yield
                    ) / 100

                return newton(
                    func=objective,
                    x0=initial_guess,
                    fprime=vega,
                    maxiter=50,
                    tol=tolerance
                )
            except Exception:
                return brentq(objective, 0.01, 3.0, maxiter=max_iterations)

        else:
            raise ValueError(f"Unknown method '{method}'")

    except Exception as e:
        warnings.warn(f"Implied volatility could not be solved: {e}")
        return np.nan


def _initial_iv_guess(option_price, stock_price, strike_price, time_to_expiry):
    """
    Simple initial guess using common approximations.
    Works reasonably well for ATM options.
    """
    atm = abs(stock_price - strike_price) < 0.05 * stock_price
    if atm:
        return np.sqrt(2 * np.pi / time_to_expiry) * (option_price / stock_price)
    else:
        return np.sqrt(2 * np.pi) * option_price / (stock_price * np.sqrt(time_to_expiry))


def test_iv():
    """
    Small test to verify the module.
    Computes IV from a price that was generated using BSM.
    """
    bsm = BlackScholesModel()

    S = 1650
    K = 1650
    T = 30/365
    r = 0.067
    true_vol = 0.25

    price = bsm.price_call(S, K, T, r, true_vol)

    recovered_vol = calculate_implied_volatility(
        market_price=price,
        stock_price=S,
        strike_price=K,
        time_to_expiry=T,
        risk_free_rate=r,
        option_type='call',
        method='hybrid'
    )

    print("True Volatility :", true_vol)
    print("Recovered Vol   :", recovered_vol)
    print("Error           :", abs(true_vol - recovered_vol))


if __name__ == "__main__":
    test_iv()
