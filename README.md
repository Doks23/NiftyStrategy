# NiftyStrategy — VWAP option-selling backtest

Rules: 5-min Nifty spot; candle closes above VWAP -> sell ATM Put, below -> sell ATM Call;
exit on opposite-side close; 3-min cool-off; daily loss cap 1.8% of capital; EOD square-off 15:15.

    pip install pandas requests
    # .env.dhan: DHAN_CLIENT_ID=...  DHAN_ACCESS_TOKEN=...   (never commit)
    python -m niftyvwap.backtest fetch --from 2025-06-01 --to 2025-12-31
    python -m niftyvwap.backtest run --lots 1 --lot-size 75 --capital 1000000

Assumptions to confirm: index volume is 0 on Dhan, so VWAP falls back to unweighted typical price
(use Nifty futures for true VWAP); weekly nearest expiry; fixed strike tracked after entry;
roll-down technique NOT implemented (video summary was cut off); charges excluded.
Tests (`python -I tests/test_strategy.py`) use hand-built days, not market data.

## Momentum-base ORH + RVOL (pullback/orh_rvol.py)
Contraction dot (daily) -> next day: 11-min opening volume >= 8% of 50d avg volume -> buy break of opening-range high.
Stop = tightest of contraction low / day low / 2.5%; BE at 2R; book 25% at 4R; sell all below EMA21.

    python -m pullback.bhavcopy --from 2025-06-01        # daily NSE panel
    python -m pullback.orh_rvol --trade-from 2025-10-13  # needs .env.dhan; 1-min data cached in data/dhan_1m
    python -I tests/test_orh_rvol.py
