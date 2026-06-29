import numpy as np
from src.pricing_models.black_scholes import BlackScholesModel

def test_call_put_parity_basic():
    bsm = BlackScholesModel()
    S, K, T, r, sigma = 100, 100, 30/365, 0.05, 0.2

    call = bsm.price_call(S, K, T, r, sigma)
    put = bsm.price_put(S, K, T, r, sigma)

    lhs = call - put
    rhs = S - K * np.exp(-r*T)

    assert abs(lhs - rhs) < 1e-6
