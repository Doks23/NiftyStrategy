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

---

# swingema — 10/21 EMA + Supertrend(10,3) swing backtest (NSE cash, daily, long only)

    python -m swingema.backtest fetch --from 2025-01-01 --to 2026-10-14   # ~9m warm-up + test year
    python -m swingema.backtest run   --from 2025-10-15 --to 2026-10-14 --out data_swing/base
    python -I tests/test_swingema.py

Flags: `--all-signals` (no cash limit: measures rule edge in R), `--no-scaling` (control: same entries,
all-out on stop / 2 closes < 21 EMA), `--no-st-filter`, `--no-be`, `--no-dema-filter`, `--no-charges`.
Outputs: `trades.csv` (one row per trade cycle, R-multiple), `fills.csv` (every fill + rule tag), `equity.csv`.

Universe filter (as of each signal date): 30-day avg traded value (close x volume) >= Rs 30 cr,
market cap (close x current shares outstanding) >= Rs 1,500 cr, price > Rs 20.

Implementation choices (spec gaps closed - change in `swingema/strategy.py:Params`):
- Signals on daily close fill at next open; stop / 4R fill intraday at level (open if gapped). Stop
  checked before target on the same bar.
- Full size Q = 1% equity risk / R, capped at 25% equity notional, multiple of 4; entered all at once.
  Entry skipped if full size isn't affordable (no partial slivers). Same-day candidates ranked by turnover.
- Each exit/add = Q/4. T1 fires once per cycle; T2-T4 re-arm only after their add-back. T4 sells one
  tranche. Stop -> breakeven after T1. An add raises the stop for the whole position to
  max(current stop, latest swing low - 0.25 ATR); add skipped if position risk would exceed 1% or
  Supertrend is red.
- One entry per bullish crossover. Pivots need 3 bars each side and are used only once confirmed.
- Costs: delivery STT 0.1% both sides, stamp 0.015% buy, NSE txn 0.00297%, SEBI, GST, DP Rs15.34/sell,
  zero brokerage, 0.05% slippage on market orders.

Data caveats: Yahoo daily (split-adjusted, but misses some splits/demergers -> overnight gaps outside
-40%/+67% are back-adjusted as corporate actions); current listings only (survivorship bias);
market cap uses today's share count.
