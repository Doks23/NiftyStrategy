"""VWAP option-selling strategy engine (pure logic, no I/O).

Rules (from the strategy video summary):
  * 5-minute Nifty spot candles, standard session VWAP.
  * Candle CLOSES above VWAP -> bullish -> SELL ATM Put.
    Candle CLOSES below VWAP -> bearish -> SELL ATM Call.
  * Hold until a candle closes on the opposite side of VWAP, then exit.
  * After an exit, wait `cooloff_min` minutes before scanning for a new entry.
  * Daily loss cap = loss_cap_pct of capital; once hit, flatten and stop for the day.
  * Hard square-off at `squareoff` (default 15:15).

Execution model: signals are evaluated at the 5-min candle close (bar start + 5).
Fills use the open of the 1-minute option bar starting at that minute, adjusted
by `slippage` points against us. The strike is fixed at entry and tracked.
Loss cap is checked every minute on 1-minute option closes (MTM).
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class Params:
    candle_min: int = 5
    cooloff_min: int = 3
    loss_cap_pct: float = 1.8
    capital: float = 1_000_000.0      # capital deployed for the day (INR)
    lots: int = 1
    lot_size: int = 75
    slippage: float = 1.0             # option points, each side, against us
    squareoff: time = time(15, 15)
    vwap_source: str = "volume"       # "volume" (needs volume) or "typical" (unweighted)
    roll_decay: float = 0.60          # roll down when premium has decayed this much from entry; 0 = off
    skip_first_candle: bool = True    # ignore the 09:15 candle for signals (still in VWAP)


@dataclass
class Trade:
    date: object
    side: str            # "SELL_CE" or "SELL_PE"
    strike: float
    entry_ts: datetime
    entry_px: float
    exit_ts: datetime = None
    exit_px: float = None
    reason: str = ""
    qty: int = 0
    pnl: float = 0.0     # INR, short option: (entry - exit) * qty

    def close(self, ts, px, reason):
        self.exit_ts, self.exit_px, self.reason = ts, px, reason
        self.pnl = (self.entry_px - px) * self.qty


def session_vwap(bars: pd.DataFrame, source: str) -> pd.Series:
    """Cumulative VWAP from the first bar of the day. `bars` has high/low/close/volume."""
    tp = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    if source == "volume" and bars["volume"].sum() > 0:
        pv = (tp * bars["volume"]).cumsum()
        v = bars["volume"].cumsum().replace(0, float("nan"))
        return pv / v
    return tp.expanding().mean()


def run_day(day, spot5: pd.DataFrame, opt1: Dict[Tuple[str, float], pd.DataFrame],
            atm_at, p: Params) -> List[Trade]:
    """Simulate one day.

    spot5  : 5-min spot bars indexed by bar-start datetime (cols high/low/close/volume)
    opt1   : {("CE"|"PE", strike): 1-min bars indexed by bar-start datetime (open/close)}
    atm_at : f(ts, "CE"|"PE") -> strike that was ATM at ts (from the data feed)
    """
    trades: List[Trade] = []
    if spot5.empty:
        return trades
    vwap = session_vwap(spot5, p.vwap_source)
    step = timedelta(minutes=p.candle_min)
    # candle-close time -> (close, vwap)
    first = spot5.index[0]
    closes = {ts + step: (row.close, vwap[ts]) for ts, row in spot5.iterrows()
              if not (p.skip_first_candle and ts == first)}
    qty = p.lots * p.lot_size
    cap_inr = p.capital * p.loss_cap_pct / 100.0

    realised, open_t, cool_until, halted = 0.0, None, None, False
    day_end = datetime.combine(day, p.squareoff)
    last_min = max(ts for df in opt1.values() for ts in df.index) if opt1 else None
    if last_min is None:
        return trades

    def px(key, ts, field):
        df = opt1.get(key)
        if df is None or ts not in df.index:
            return None
        return float(df.at[ts, field])

    ts = datetime.combine(day, time(9, 15))
    while ts <= min(day_end, last_min + timedelta(minutes=1)):
        # 1) mark-to-market loss cap
        if open_t is not None and not halted:
            mark = px((open_t.side[-2:], open_t.strike), ts - timedelta(minutes=1), "close")
            if mark is not None:
                mtm = realised + (open_t.entry_px - mark) * qty
                if mtm <= -cap_inr:
                    fill = px((open_t.side[-2:], open_t.strike), ts, "open") or mark
                    open_t.close(ts, fill + p.slippage, "LOSS_CAP")
                    realised += open_t.pnl
                    open_t, halted = None, True
        if halted:
            ts += timedelta(minutes=1)
            continue
        # 2) hard square-off
        if ts >= day_end:
            if open_t is not None:
                key = (open_t.side[-2:], open_t.strike)
                fill = px(key, ts, "open")
                if fill is None:  # fall back to last known close
                    fill = float(opt1[key]["close"].iloc[-1])
                open_t.close(ts, fill + p.slippage, "EOD")
                realised += open_t.pnl
                open_t = None
            break
        # 2b) roll-down: premium decayed >= roll_decay from this leg's entry -> re-sell at new ATM
        if open_t is not None and p.roll_decay > 0:
            typ = open_t.side[-2:]
            mark = px((typ, open_t.strike), ts - timedelta(minutes=1), "close")
            if mark is not None and mark <= open_t.entry_px * (1 - p.roll_decay):
                new_strike = atm_at(ts, typ)
                old_fill = px((typ, open_t.strike), ts, "open")
                new_fill = px((typ, new_strike), ts, "open") if new_strike is not None else None
                if new_strike != open_t.strike and old_fill is not None and new_fill is not None:
                    open_t.close(ts, old_fill + p.slippage, "ROLL")
                    realised += open_t.pnl
                    open_t = Trade(day, open_t.side, new_strike, ts, new_fill - p.slippage, qty=qty)
                    trades.append(open_t)
        # 3) signal at candle close
        if ts in closes:
            c, v = closes[ts]
            bullish = c > v
            bearish = c < v
            if open_t is not None:
                reverse = (open_t.side == "SELL_PE" and bearish) or (open_t.side == "SELL_CE" and bullish)
                if reverse:
                    key = (open_t.side[-2:], open_t.strike)
                    fill = px(key, ts, "open")
                    if fill is not None:
                        open_t.close(ts, fill + p.slippage, "REVERSAL")
                        realised += open_t.pnl
                        open_t = None
                        cool_until = ts + timedelta(minutes=p.cooloff_min)
            if open_t is None and (bullish or bearish) and (cool_until is None or ts >= cool_until):
                typ = "PE" if bullish else "CE"
                strike = atm_at(ts, typ)
                fill = px((typ, strike), ts, "open") if strike is not None else None
                if fill is not None:
                    open_t = Trade(day, f"SELL_{typ}", strike, ts, fill - p.slippage, qty=qty)
                    trades.append(open_t)
        ts += timedelta(minutes=1)
    # safety: anything still open at data end
    if open_t is not None:
        key = (open_t.side[-2:], open_t.strike)
        open_t.close(ts, float(opt1[key]["close"].iloc[-1]) + p.slippage, "DATA_END")
    return trades
