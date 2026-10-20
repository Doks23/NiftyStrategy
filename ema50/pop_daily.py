"""Daily-close version of 'pop, settle, buy again' (no intraday data needed).

Per setup (the 50 MA pop day T):
  KILL    : any close after T below the low of T -> setup cancelled.
  SETTLE  : `q` consecutive sessions after T with range% <= quiet_k x ADR.
  TRIGGER : the session right after the settle (any later day up to T+MAXK as long as the last q sessions are quiet) closes ABOVE the highest high of
            the quiet base (variant 'pop-high': above the high of T) on volume >= vol_x x the 21-day average.
  ENTRY   : at that session's close (+slippage).  STOP: lowest low of the base and trigger day, at most 8% below entry, at least 1%.
  EXITS   : 2R -> breakeven, 4R -> sell 25%, close < SMA21 x 0.99 -> sell next open.   RISK: Rs 1,500 per trade, notional capped at Rs 10 lakh.
"""
import math
import numpy as np, pandas as pd
from .strategy import P, features, charges

MAXK = 10


def load():
    d = pd.read_csv("data_eq/daily_univ.csv", parse_dates=["date"]); d["date"] = d.date.dt.date
    f = features(d, "sma"); f["rng"] = (f.high / f.low - 1) * 100
    return {s: g.reset_index(drop=True) for s, g in f.groupby("symbol")}


def find_trade(g, i, q, quiet_k=0.8, vol_x=1.5, mode="base-high", p=P()):
    low_T, high_T = g.low.iloc[i], g.high.iloc[i]
    for j in range(i + 1 + q, min(i + MAXK, len(g) - 1) + 1):
        if (g.close.iloc[i + 1:j + 1] < low_T).any():                       # a close below the pop-day low (incl. the trigger day) kills the setup
            return None
        base = range(j - q, j)
        if not all(g.rng.iloc[b] <= quiet_k * g.adr.iloc[b] for b in base): continue
        level = max(g.high.iloc[b] for b in base) if mode == "base-high" else high_T
        if g.close.iloc[j] > level and g.volume.iloc[j] >= vol_x * g.av21.iloc[j]:
            e = g.close.iloc[j] * (1 + p.slip); stop = min(min(g.low.iloc[b] for b in base), g.low.iloc[j])
            stop = max(stop, e * 0.92); stop = min(stop, e * 0.99)
            return j, e, stop
    return None


def simulate(g, j, e, stop, p=P()):
    R = e - stop; q = math.floor(p.risk / R); q = min(q, int(1_000_000 / e))
    if q <= 0: return None
    left, be, booked, legs, cs = q, False, False, [], stop
    pend = g.close.iloc[j] < g.e21.iloc[j] * (1 - p.exit_buf)
    for k in range(j + 1, len(g)):
        b = g.iloc[k]
        if pend: legs.append((left, b.open * (1 - p.slip), b.date, "EMA21")); left = 0; break
        if b.low <= cs: legs.append((left, min(b.open, cs) * (1 - p.slip), b.date, "BE" if be else "STOP")); left = 0; break
        if not be and b.high >= e + 2 * R: be, cs = True, e
        if not booked and b.high >= e + 4 * R:
            kq = math.floor(q * .25)
            if kq: legs.append((kq, (e + 4 * R) * (1 - p.slip), b.date, "4R_25%")); left -= kq
            booked = True
        if b.close < b.e21 * (1 - p.exit_buf): pend = True
    if left:
        b = g.iloc[-1]; legs.append((left, b.close * (1 - p.slip), b.date, "OPEN"))
    gross = sum(k * (px - e) for k, px, _, _ in legs); fee = sum(charges(k * e, k * px, False) for k, px, _, _ in legs)
    return dict(entry=round(e, 2), stop=round(stop, 2), qty=q, exit=legs[-1][3], exit_date=legs[-1][2], gross=round(gross), fees=round(fee), pnl=round(gross - fee),
                r=round((gross - fee) / p.risk, 2), hold=(legs[-1][2] - g.date.iloc[j]).days)


def run(by, st, **kw):
    out = []
    for r in st.itertuples():
        g = by[r.symbol]; ix = g.index[g.date == r.date]
        if not len(ix): continue
        i = ix[0]; f = find_trade(g, i, **kw)
        if f is None: continue
        j, e, stop = f; res = simulate(g, j, e, stop)
        if res: out.append(dict(symbol=r.symbol, setup=r.date, entry_date=g.date.iloc[j], days_after_pop=j - i, **res))
    return pd.DataFrame(out)


def summ(name, t):
    if t.empty: return dict(variant=name, n=0)
    t = t.assign(yr=pd.to_datetime(t.entry_date).dt.year); w, l = t[t.pnl > 0].pnl.sum(), -t[t.pnl <= 0].pnl.sum(); a, b = t[t.yr == 2025], t[t.yr == 2026]
    eq = t.sort_values("entry_date").pnl.cumsum()
    return dict(variant=name, n=len(t), win=round(100 * (t.pnl > 0).mean()), net=int(t.pnl.sum()), gross=int(t.gross.sum()), fees=int(t.fees.sum()), pf=round(w / l, 2) if l else 9.9,
                avgR=round(t.r.mean(), 2), hold=int(t.hold.median()), maxdd=int((eq - eq.cummax()).min()), y2025=f"{len(a)}/{int(a.pnl.sum())}", y2026=f"{len(b)}/{int(b.pnl.sum())}")


if __name__ == "__main__":
    by = load(); st = pd.read_csv("data_eq/setups_cl.csv", parse_dates=["date"]); st["date"] = st.date.dt.date; rows, allt = [], {}
    for name, kw in [("PRIMARY: 2 quiet days, close > base high, vol>=1.5x", dict(q=2)), ("3 quiet days, close > base high, vol>=1.5x", dict(q=3)),
                     ("2 quiet days, close > pop-day high, vol>=1.5x", dict(q=2, mode="pop-high")), ("2 quiet days, close > base high, any volume", dict(q=2, vol_x=0.0)),
                     ("1 quiet day, close > base high, vol>=1.5x", dict(q=1))]:
        t = run(by, st, **kw); allt[name] = t.assign(variant=name); rows.append(summ(name, t))
    pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 60); print(pd.DataFrame(rows).to_string(index=False))
    t = allt["PRIMARY: 2 quiet days, close > base high, vol>=1.5x"]
    print("\nPRIMARY: exits", t.exit.value_counts().to_dict(), "| entry day after pop:", t.days_after_pop.value_counts().sort_index().to_dict())
    print("by quarter:", t.groupby(pd.to_datetime(t.entry_date).dt.to_period("Q")).pnl.agg(["count", "sum"]).astype(int).T.to_string())
    print("\ntop winners:"); print(t.nlargest(5, "pnl")[["symbol", "setup", "entry_date", "exit", "pnl", "r"]].to_string(index=False))
    pd.concat(allt.values()).to_csv("data_eq/pop_daily_trades.csv", index=False)
