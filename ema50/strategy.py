"""EMA-50 reclaim after downtrend + next-day RVOL opening breakout (NSE cash).

Universe (on T-1): mcap = shares x close > 1500 Cr, close > 20, 30d avg(close*vol) > 25 Cr.
Downtrend (T-1): close < EMA10, EMA20, EMA50 and close < EMA50 on >= 10 of last 15 sessions.
Setup day T: (close > EMA50 or high >= EMA50) AND (gain >= 4% or >= 1.5 x ADR%) AND
             close in top 25% of range AND volume >= 1.5 x 21d avg volume.
Entry window: T+1; T+2/T+3 only if the prior day is tight (range <= 0.7 x ADR, or inside bar).
Entry (1-min): OR high = high of 09:15-09:17. Between 09:18 and 09:25 (first 11 minutes), once
             cumulative volume >= 8% of 21d avg daily volume AND price > OR high -> buy at
             max(OR high, bar open) + slippage. One trade per setup.
Stop: day low so far, clamped to 1%..2% below entry. Size = floor(risk / (entry - stop)).
Exits: 2R -> stop to breakeven; 4R -> sell 25%; daily close < EMA21 x (1 - exit_buf) -> sell at
       next open. After the entry day, daily bars are used (stop checked before targets).
"""
from dataclasses import dataclass
from datetime import time
import math
import numpy as np
import pandas as pd


@dataclass
class P:
    risk: float = 1500.0
    slip: float = 0.0015
    rvol_pct: float = 8.0
    min_sl: float = 0.01
    max_sl: float = 0.02
    exit_buf: float = 0.01
    tight_k: float = 0.7
    min_gain: float = 4.0
    adr_k: float = 1.5


def features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.sort_values(["symbol", "date"]).copy()
    g = d.groupby("symbol", group_keys=False)
    for n in (10, 20, 21, 50):
        d[f"e{n}"] = g.close.transform(lambda s: s.ewm(span=n, adjust=False).mean())
    d["pc"] = g.close.shift(1)
    d["hl"] = d.high / d.low - 1
    d["tv"] = d.close * d.volume
    g = d.groupby("symbol", group_keys=False)
    d["adr"] = g.hl.transform(lambda s: s.rolling(20).mean()) * 100
    d["av21"] = g.volume.transform(lambda s: s.rolling(21).mean().shift(1))
    d["tv30"] = g.tv.transform(lambda s: s.rolling(30).mean()) / 1e7
    d["below50"] = (d.close < d.e50).astype(int)
    d["b50_15"] = g.below50.transform(lambda s: s.rolling(15).sum())
    d["mcap"] = d.shares_cr * d.close
    return d


def setups(d: pd.DataFrame, p: P) -> pd.DataFrame:
    g = d.groupby("symbol")
    prev = g[["close", "e10", "e20", "e50", "b50_15", "mcap", "tv30"]].shift(1).add_prefix("p_")
    x = pd.concat([d, prev], axis=1)
    gain = (x.close / x.pc - 1) * 100
    rng = (x.high - x.low).replace(0, np.nan)
    ok = ((x.p_mcap > 1500) & (x.p_close > 20) & (x.p_tv30 > 25) &
          (x.p_close < x.p_e10) & (x.p_close < x.p_e20) & (x.p_close < x.p_e50) & (x.p_b50_15 >= 10) &
          ((x.close > x.e50) | (x.high >= x.e50)) &
          ((gain >= p.min_gain) | (gain >= p.adr_k * x.adr)) &
          ((x.close - x.low) / rng >= 0.75) & (x.volume >= 1.5 * x.av21))
    return x[ok][["symbol", "sid", "date"]]


def candidate_days(d, st):
    """(symbol, sid, setup_date, trade_date, k) for k = 1..3 sessions after setup."""
    out = []
    for s, grp in d.groupby("symbol"):
        dates = list(grp.date)
        for r in st[st.symbol == s].itertuples():
            i = dates.index(r.date)
            for k in (1, 2, 3):
                if i + k < len(dates):
                    out.append((s, r.sid, r.date, dates[i + k], k))
    return pd.DataFrame(out, columns=["symbol", "sid", "setup", "date", "k"])


def tight(day_prev, day_prev2, p):
    rng = (day_prev.high - day_prev.low) / day_prev.close * 100
    inside = day_prev.high <= day_prev2.high and day_prev.low >= day_prev2.low
    return rng <= p.tight_k * day_prev.adr or inside


def entry_on(m: pd.DataFrame, av21: float, p: P):
    """m = 1-min bars of the trade day. Returns (ts, entry, stop) or None."""
    m = m.sort_values("ts")
    t = m.ts.dt.time
    orb = m[(t >= time(9, 15)) & (t < time(9, 18))]
    if orb.empty or not av21:
        return None
    orh = orb.high.max()
    win = m[(t >= time(9, 18)) & (t <= time(9, 25))]
    for r in win.itertuples():
        cum = m[m.ts <= r.ts].volume.sum()
        if cum >= p.rvol_pct / 100 * av21 and r.high > orh:
            raw = max(orh, r.open)
            e = raw * (1 + p.slip)
            lod = m[m.ts <= r.ts].low.min()
            stop = min(max(lod, e * (1 - p.max_sl)), e * (1 - p.min_sl))
            return r.ts, e, stop, m[m.ts > r.ts]
    return None


def charges(buy, sell, intraday):
    """Equity charges INR for one round trip (Dhan: delivery brokerage 0, intraday min(20, 0.03%))."""
    t = buy + sell
    brk = 0.0 if not intraday else min(20, 0.0003 * buy) + min(20, 0.0003 * sell)
    stt = 0.00025 * sell if intraday else 0.001 * t
    txn = 0.0000297 * t
    sebi = t * 1e-6
    stamp = buy * (0.00003 if intraday else 0.00015)
    dp = 0 if intraday else 15.93
    return brk + stt + txn + sebi + stamp + dp + 0.18 * (brk + txn + sebi + dp)


def simulate(entry, rest_min, daily_after, p: P, sym, d0):
    ts, e, stop, after = entry
    R = e - stop
    qty = math.floor(p.risk / R)
    if qty <= 0:
        return None
    legs, left, be = [], qty, False          # legs: (qty, exit_px, ts, reason)
    cur_stop = stop
    booked = False

    def bars():
        for r in after.itertuples():
            yield r.ts, r.open, r.high, r.low, None, True
        for j, r in enumerate(daily_after.itertuples()):
            if j == 0:      # entry day: intraday already handled, only its close vs EMA21
                yield r.date, None, -1, float("inf"), r, False
            else:
                yield r.date, r.open, r.high, r.low, r, False

    pending_exit = False
    for when, o, hi, lo, drow, intra in bars():
        if pending_exit:
            legs.append((left, o * (1 - p.slip), when, "EMA21")); left = 0; break
        if o is None and not intra:
            if drow.close < drow.e21 * (1 - p.exit_buf):
                pending_exit = True
            continue
        if lo <= cur_stop:
            px = min(o, cur_stop) * (1 - p.slip)
            legs.append((left, px, when, "BE" if be else "STOP")); left = 0; break
        if not be and hi >= e + 2 * R:
            be, cur_stop = True, e
        if not booked and hi >= e + 4 * R:
            q = math.floor(qty * 0.25)
            if q:
                legs.append((q, (e + 4 * R) * (1 - p.slip), when, "4R_25%")); left -= q
            booked = True
        if drow is not None and drow.close < drow.e21 * (1 - p.exit_buf):
            pending_exit = True
    if left:
        last = daily_after.iloc[-1] if len(daily_after) else None
        px = (last.close if last is not None else after.close.iloc[-1]) * (1 - p.slip)
        legs.append((left, px, "open", "OPEN_AT_END"))
    gross = sum(q * (x - e) for q, x, _, _ in legs)
    fee = sum(charges(q * e, q * x, str(w)[:10] == str(d0)) for q, x, w, _ in legs)
    hold = legs[-1][2]
    return dict(symbol=sym, date=d0, entry_ts=ts, entry=round(e, 2), stop=round(stop, 2), qty=qty,
                R_rs=round(R * qty, 0), exit=legs[-1][3], exit_when=hold, gross=round(gross, 0),
                fees=round(fee, 0), pnl=round(gross - fee, 0), r_mult=round((gross - fee) / p.risk, 2))
