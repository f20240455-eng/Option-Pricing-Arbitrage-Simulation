import pandas as pd
from src.preprocessing.atm_selector import _pick_atm_strike

def test_atm_picker():
    strikes = [900, 920, 940, 960]
    S = 935
    selected = _pick_atm_strike(S, strikes)

    assert selected == 940
