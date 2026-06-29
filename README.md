Overview
--------
This project performs end-to-end options pricing, synthetic option chain generation, ATM selection, convergence testing, and full backtesting on HDFCBANK equities.

Key Components
--------------
- Data ETL and preprocessing
- Synthetic option chain construction
- Black-Scholes & Binomial pricing
- ATM strike selection
- Convergence analysis
- Backtest engine
- Validation checker
- Unit tests

Run Instructions
----------------
pip install -r requirements.txt
pytest --maxfail=1 -q
