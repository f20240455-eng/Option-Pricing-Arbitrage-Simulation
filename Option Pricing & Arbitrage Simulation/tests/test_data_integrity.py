import pandas as pd
import pytest

def test_no_missing_values():
    stock = pd.read_csv("data/processed/hdfc_stock_processed_271P.csv")
    assert stock.isnull().mean().max() == 0

def test_atm_lookup_columns():
    atm = pd.read_csv("data/processed/atm_lookup_271P.csv")
    required = ["trade_date", "atm_strike", "implied_vol", "call_mid"]
    for c in required:
        assert c in atm.columns
