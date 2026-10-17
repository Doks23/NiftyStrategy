# Chart Replay — reusable prompt

Paste everything below the line into Claude (Claude Code or claude.ai). Fill the `<<...>>` fields.
Works with any trade list; if you are in the NiftyStrategy repo, Claude can reuse
`scripts/trade_review_data.py` + `scripts/trade_review_template.html` instead of rebuilding.

---

Build me a **Chart Replay** page for reviewing trades as a visual exercise.

## Input
- Trades: <<path to CSV / pasted table / "the trades from the backtest we just ran">>
- Required columns (map mine if names differ): `sym`, `setup_date` (the signal / dot / pattern day),
  `entry_date`, `entry`, `stop` (or `stop_pct`), `exit_date`, `R`, `ret_pct`, `why` (exit reason).
  Optional: entry time, RVOL, market-filter flag, notes, any extra columns → show them as facts.
- Market: <<NSE / US / crypto>>. Daily OHLC source, in this order of preference:
  <<local bhavcopy panel data/bhav/panel.pkl | Dhan API (.env.dhan) | yfinance (.NS suffix for NSE)>>.
  Use split/bonus-adjusted prices.
- Overlay: <<EMA 21>> (line) — change if my exit uses a different MA.
- Exercise title: <<e.g. "RVOL Dot Trade Review", "Breakout Drill – Oct 2026">>

## Chart window (keep exactly)
- From ~35 trading days before `setup_date` to ~8 trading days after `exit_date`.
- Daily candles, up = green, down = red, faint price grid with labels on the right.

## Replay behaviour (the point of the page)
1. Each chart **opens on the setup day with every later candle hidden** behind a grey "future hidden" area.
2. Reveal / hide candles one at a time by:
   - dragging the chart left↔right (mouse and touch; one candle per candle-width of drag),
   - a slider under the chart (any day),
   - ◀ / ▶ buttons and keyboard arrows,
   - "Setup" button (jump back to start) and "Show all".
3. A live day label: date + tag "setup" / "entry" / "exit" when on those days.
4. Markers appear **only once the replay reaches that day** (no spoilers):
   setup-day dot above the candle, entry triangle + entry price line, dashed initial-stop line,
   shaded holding period, exit triangle, MA drawn only up to the current candle.
5. Each chart remembers its replay position when I filter or sort.
6. <<Optional blind mode: y-axis rescales to visible candles only, R/return hidden until "Show all".>>

## Page around the charts
- One row per trade: facts on the left (symbol, R coloured green/red, setup date, entry date/time,
  entry/stop/stop %, exit date + reason, return %, my extra columns) and the replay chart on the right;
  stack to one column on phones.
- "Open on TradingView" button per trade:
  `https://www.tradingview.com/chart/?symbol=<<EXCHANGE>>:<SYM>&interval=D`
  (URL-encode symbols like M&M; TradingView can't deep-link a date — tell me to use Alt+G with the entry date).
- Summary strip that follows the filter: trades, win rate, total R, return on capital at <<0.5>>% risk per trade.
- Filters: All / Winners / Losers / <<market-filter flag>>; sort by date, best R, worst R.
- Legend for the markers. Light and dark theme. No external data calls — embed the candle data in the page.

## Deliver
- Publish it as an artifact (private) and give me the link.
- Save the generator (data script + HTML template) in the repo so I can rebuild after the next backtest
  with one command; commit and push.
- Tell me in 3–5 lines what to look at first (e.g. best and worst trades, any pattern in losers).
