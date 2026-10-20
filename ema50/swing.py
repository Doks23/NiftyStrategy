"""Daily swing entry on Chartink setups: buy T+1 open (or T+1 close) ; stop = setup-day low (capped); exits as before.
Exit: 2R -> BE, 4R -> sell 25%, close < SMA21*(1-buf) -> sell next open. Costs: delivery charges + 0.15% slip/side."""
import math, sys
import numpy as np, pandas as pd
from .strategy import features, charges, P

def run(p=P(), entry="open", max_sl=0.08, wait=1, kron=None, thr=None, daily="daily_ck.csv", setups="setups_ck.csv"):
    d = features(pd.read_csv("data_eq/" + daily, parse_dates=["date"]).assign(), "sma")
    d["date"] = pd.to_datetime(d.date).dt.date
    st = pd.read_csv("data_eq/" + setups, parse_dates=["date"]).drop_duplicates(["symbol", "date"]); st["date"] = st.date.dt.date
    out = []
    for r in st.itertuples():
        x = d[d.symbol == r.symbol].reset_index(drop=True); ix = x.index[x.date == r.date]
        if not len(ix) or ix[0] + wait >= len(x): continue
        i = ix[0]; e0 = x.iloc[i + wait]
        raw = e0.open if entry == "open" else e0.close
        e = raw * (1 + p.slip); stop = max(x.low.iloc[i], e * (1 - max_sl))
        if stop >= e * 0.995: continue
        R = e - stop; q = math.floor(p.risk / R)
        if q <= 0 or q * e > 500000 * 2: q = min(q, int(1_000_000 / e))
        if q <= 0: continue
        left, be, booked, legs, cs, pend = q, False, False, [], stop, False
        for j in range(i + wait + (0 if entry == "close" else 0), len(x)):
            b = x.iloc[j]
            if j == i + wait and entry == "close":
                if b.close < b.e21 * (1 - p.exit_buf): pend = True
                continue
            if pend: legs.append((left, b.open * (1 - p.slip), b.date, "EMA21")); left = 0; break
            if b.low <= cs:
                legs.append((left, min(b.open, cs) * (1 - p.slip), b.date, "BE" if be else "STOP")); left = 0; break
            if not be and b.high >= e + 2 * R: be, cs = True, e
            if not booked and b.high >= e + 4 * R:
                k = math.floor(q * .25)
                if k: legs.append((k, (e + 4 * R) * (1 - p.slip), b.date, "4R_25%")); left -= k
                booked = True
            if b.close < b.e21 * (1 - p.exit_buf): pend = True
        if left:
            b = x.iloc[-1]; legs.append((left, b.close * (1 - p.slip), b.date, "OPEN")); 
        gross = sum(k * (px - e) for k, px, _, _ in legs); fee = sum(charges(k * e, k * px, False) for k, px, _, _ in legs)
        out.append(dict(symbol=r.symbol, setup=r.date, entry_date=x.date.iloc[i + wait], entry=round(e, 2), stop=round(stop, 2), qty=q,
                        exit=legs[-1][3], exit_date=legs[-1][2], gross=round(gross), fees=round(fee), pnl=round(gross - fee),
                        r=round((gross - fee) / p.risk, 2), hold=(legs[-1][2] - x.date.iloc[i + wait]).days))
    return pd.DataFrame(out)

def summ(t):
    w, l = t[t.pnl > 0].pnl.sum(), -t[t.pnl <= 0].pnl.sum()
    eq = t.sort_values("entry_date").pnl.cumsum()
    return dict(trades=len(t), win=round(100 * (t.pnl > 0).mean(), 1), net=int(t.pnl.sum()), gross=int(t.gross.sum()), fees=int(t.fees.sum()),
                pf=round(w / l, 2) if l else None, avgR=round(t.r.mean(), 2), dd=int((eq - eq.cummax()).min()), exits=t.exit.value_counts().to_dict())

if __name__ == "__main__":
    for name, kw in [("T+1 open", dict(entry="open")), ("T+1 close", dict(entry="close")), ("T+1 open, stop cap 5%", dict(entry="open", max_sl=.05)),
                     ("T+2 open", dict(entry="open", wait=2))]:
        t = run(**kw); print(f"\n## {name}"); [print(f"{k:>8}: {v}") for k, v in summ(t).items()]
        t.to_csv(f"data_eq/swing_{name.replace(' ','_').replace('+','').replace(',','')}.csv", index=False)
