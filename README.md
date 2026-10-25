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

---

# ironfly — Monthly Nifty Iron Fly with adjustments (Dhan "Working Professionals" video, caof-oHPe4E)

    python -m ironfly.fetch_bhav 2024-01-01 2026-10-24     # NSE public F&O bhavcopy (EOD), no login needed
    python -m ironfly.backtest --from 2026-01-01 --out data_if/run_2026.csv
    python -m ironfly.sweep                                 # sensitivity grid
    python -I tests/test_ironfly.py

Rules (video summary + gaps closed; flags in `ironfly/backtest.py`): enter day after monthly expiry on the
next monthly; sell ATM CE+PE, buy wings at +/-W where W = widest wing with max loss <= 3% of capital
(Rs 2L/lot default); when spot closes beyond a breakeven, move the threatened short to the old hedge strike
and buy a new hedge one wing further; 4% hard stop on MTM; hold to expiry. Costs incl. STT 0.15% from Apr-2026.

Data caveats: EOD only (video enters 11-12 and uses intraday alerts; here entry, adjustments and stop act on
daily closes). Deep-ITM closes are stale, so ITM strikes are re-priced from the OTM twin via put-call parity
on the same-expiry future. 1 pt slippage per option traded.

Adjustment variants (`--adj-mode`): `roll` (video reading: threatened short -> old hedge strike, new hedge
one wing out), `recenter` (close all, new 3%-rule fly at ATM), `untested` (move untested side to ATM).

Results (per lot, Rs 2L capital, Feb-2024..Oct-2026, 32-33 cycles; `data_export/ironfly_sweep.txt`):

    mode                total %  2026 %  maxDD %  stops
    roll (video)          -11.7    -5.7    -31.5      9
    recenter              -53.1   -15.7    -54.1      8
    untested to ATM       -19.7    -9.9    -20.5      3
    no adjustment         +43.9   +26.3    -21.5      0   (DD = 6 straight losses Feb-Jul 2025)

Every adjustment variant loses; only the un-adjusted 3%-rule fly is positive, and it holds under 2-pt
slippage (+40%) and Rs 3L/lot (+33%). It wins when expiry lands within +/-credit of ATM (53% of months).

Regime filters on the un-adjusted fly (`python -m ironfly.fetch_vix && python -m ironfly.filters`;
thresholds = trailing-1y percentiles known at entry; p = share of random same-size skips doing as well):

    filter (skip month)                 total %  maxDD %  worst streak  p
    none                                  +43.9    -21.5             7  -
    VIX above 1y median                   +12.8    -12.1             2  0.72  (skips the best months)
    VIX < 20d realised vol                +47.0    -14.0             4  0.09  (only one with signal; weak)
    20d move in top quartile (trend)      +44.7    -16.2             3  0.19

Max-loss cap (wing rule): 1.5% +21.7/-14.9 DD, 2% +28.8/-16.7, 3% +43.9/-21.5, 4% +52.0/-25.8,
5% +56.0/-31.2 -> return/DD peaks around 3-4%; the video's 3% is a sound choice.
