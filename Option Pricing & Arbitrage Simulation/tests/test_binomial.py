from src.pricing_models.binomial_tree import price_binomial
from src.pricing_models.black_scholes import BlackScholesModel
import numpy as np

def test_binomial_converges_to_bsm():
    S, K, T, r, sigma = 100, 100, 30/365, 0.05, 0.2

    bsm = BlackScholesModel()
    bsm_price = bsm.price_call(S, K, T, r, sigma)

    approx = price_binomial(S, K, T, r, sigma, n_steps=500, option_type='call')

    assert abs(approx - bsm_price) < 0.02   # Tight convergence margin
