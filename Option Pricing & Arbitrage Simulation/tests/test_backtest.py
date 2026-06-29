import pandas as pd
from src.backtest_engine.backtest_engine import BacktestEngine

def test_backtest_basic():
    df = pd.DataFrame({
        "trade_date": ["2024-01-01"],
        "stock_price": [100],
        "atm_strike": [100],
        "implied_vol": [0.2],
        "days_to_expiry": [30],
        "risk_free_rate": [0.05],
        "actual_price": [5.5]
    })

    engine = BacktestEngine(df)
    ledger = engine.run_backtest()

    assert "daily_pnl" in ledger.columns
    assert len(ledger) == 1
