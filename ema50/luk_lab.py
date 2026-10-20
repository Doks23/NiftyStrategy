"""Martin Luk-style swing system on the NSE universe (daily approximation), rules taken from his own USIC-2024 write-up.
Signal day t (known at the close):
  LIQUID   : mcap > 1500 Cr, close > 20, 30-day avg turnover > 20 Cr (as of t-1)
  ADR      : 20-day mean daily range% >= adr_min
  LEAD     : EMA9 > EMA21 > EMA50 and close > EMA21
  LEADER   : 30-session return in the top `leader_pct` of the liquid universe on that date
  PULLBACK : in the last 3 sessions (t-2..t) the low came within 1% of the EMA9 (or touched it)
Entry day t+1 : buy-stop above the signal-day high: enter at max(open, high_t) if t+1 trades above high_t.
Stop          : signal-day low; must be <= stop_adr x ADR below entry (his 'half ADR' rule) and >= 0.5%.
Exits         : stop (hard); sell 15% at 3R; close below the trail EMA (9 or 21) -> sell next open. Re-entry allowed once flat.
Risk Rs 1,500 per trade (0.3% of Rs 5L), notional cap Rs 10L, 0.15% slippage each side, delivery charges. Optional Nifty gate: Nifty close > its EMA21."""
import math, sys
import numpy as np, pandas as pd
from .strategy import charges

SLIP = 0.0015; RISK = 1500.0; CAP = 1_000_000


def load():
    d = pd.read_csv("data_eq/daily_univ.csv", parse_dates=["date"]); d["date"] = d.date.dt.date; d = d.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = d.groupby("symbol", group_keys=False)
    for n in (9, 21, 50): d[f"ema{n}"] = g.close.transform(lambda s, n=n: s.ewm(span=n, adjust=False).mean())
    d["hl"] = (d.high / d.low - 1) * 100; d["adr"] = g.hl.transform(lambda s: s.rolling(20).mean())
    d["tv"] = d.close * d.volume; d["tv30"] = d.groupby("symbol").tv.transform(lambda s: s.rolling(30).mean().shift(1)) / 1e7
    d["ret30"] = g.close.transform(lambda s: s / s.shift(30) - 1); d["mcap"] = d.shares_cr * d.close
    d["liquid"] = (d.mcap > 1500) & (d.close > 20) & (d.tv30 > 20)
    d["rank30"] = d[d.liquid].groupby("date").ret30.rank(pct=True); d["rank30"] = d.rank30.fillna(0)
    return d


def signals(d, adr_min, leader_pct, stop_adr, gate):
    n = pd.read_csv("data_eq/nifty_daily.csv", parse_dates=["date"]).sort_values("date"); n["date"] = n.date.dt.date; n["ok"] = n.close > n.close.ewm(span=21, adjust=False).mean(); nok = dict(zip(n.date, n.ok))
    out = {}
    for s, g in d.groupby("symbol"):
        g = g.reset_index(drop=True); lead = (g.ema9 > g.ema21) & (g.ema21 > g.ema50) & (g.close > g.ema21)
        touch = (g.low <= g.ema9 * 1.01); pull = touch | touch.shift(1, fill_value=False) | touch.shift(2, fill_value=False)
        ok = lead & pull & g.liquid & (g.adr >= adr_min) & (g.rank30 >= 1 - leader_pct)
        if gate: ok &= g.date.map(nok).fillna(False)
        out[s] = (g, np.flatnonzero(ok.values))
    return out


def trade(g, t, trail, stop_adr):
    n = len(g); e0 = max(g.open.iloc[t + 1], g.high.iloc[t]) if t + 1 < n else None
    if e0 is None or g.high.iloc[t + 1] <= g.high.iloc[t]: return None
    e = e0 * (1 + SLIP); stop = g.low.iloc[t]; dist = (e - stop) / e * 100
    if dist < 0.5 or dist > stop_adr * g.adr.iloc[t]: return None
    R = e - stop; q = min(math.floor(RISK / R), int(CAP / e))
    if q <= 0: return None
    left, legs, pend, trimmed, armed, why = q, [], False, False, False, ''; k0 = t + 1
    for k in range(k0, n):
        o, h, l, c = g.open.iloc[k], g.high.iloc[k], g.low.iloc[k], g.close.iloc[k]
        if k > k0:
            if pend: legs.append((left, o * (1 - SLIP), k)); left = 0; why = 'TRAIL'; break
            if o <= stop: legs.append((left, o * (1 - SLIP), k)); left = 0; why = 'STOP'; break
        if l <= stop: legs.append((left, stop * (1 - SLIP), k)); left = 0; why = 'STOP'; break
        if not trimmed and h >= e + 3 * R:
            kq = math.floor(q * .15)
            if kq and kq < left: legs.append((kq, (e + 3 * R) * (1 - SLIP), k)); left -= kq
            trimmed = True
        ema = g[f"ema{trail}"].iloc[k]
        if c > ema: armed = True
        if armed and c < ema: pend = True
    if left: legs.append((left, g.close.iloc[-1] * (1 - SLIP), n - 1))
    gross = sum(k * (px - e) for k, px, _ in legs); fee = sum(charges(k * e, k * px, ix == k0) for k, px, ix in legs)
    return dict(why=why or 'OPEN', entry_idx=k0, exit_idx=legs[-1][2], date=g.date.iloc[k0], pnl=gross - fee, r=(gross - fee) / RISK, gross=gross, fees=fee, days=legs[-1][2] - k0)


def run(d, adr_min, leader_pct, stop_adr, trail, gate):
    out = []
    for s, (g, idx) in signals(d, adr_min, leader_pct, stop_adr, gate).items():
        free = -1
        for t in idx:
            if t + 1 <= free or t + 1 >= len(g): continue
            r = trade(g, t, trail, stop_adr)
            if r: r.update(symbol=s); out.append(r); free = r["exit_idx"]
    return pd.DataFrame(out)


def summ(name, t):
    if t.empty: return dict(variant=name, n=0)
    t = t.assign(yr=pd.to_datetime(t.date).dt.year); w, l = t[t.pnl > 0].pnl.sum(), -t[t.pnl <= 0].pnl.sum(); a, b = t[t.yr == 2025], t[t.yr == 2026]
    eq = t.sort_values("date").pnl.cumsum(); top3 = t.nlargest(3, "pnl").pnl.sum()
    return dict(variant=name, n=len(t), win=round(100 * (t.pnl > 0).mean()), net=int(t.pnl.sum()), gross=int(t.gross.sum()), fees=int(t.fees.sum()), pf=round(w / l, 2) if l else 9.9, avgR=round(t.r.mean(), 2),
                medhold=int(t.days.median()), net_ex_top3=int(t.pnl.sum() - top3), y2025=f"{len(a)}/{int(a.pnl.sum())}", y2026=f"{len(b)}/{int(b.pnl.sum())}", dd=int((eq - eq.cummax()).min()))


if __name__ == "__main__":
    d = load(); pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 60); rows, allt = [], {}
    for adr_min in (3.5, 4.5):
        for trail in (9, 21):
            for gate in (False, True):
                name = f"ADR>={adr_min}, trail EMA{trail}, Nifty gate {'ON' if gate else 'off'}"; t = run(d, adr_min, 0.15, 0.6, trail, gate); allt[name] = t; rows.append(summ(name, t))
    print(pd.DataFrame(rows).to_string(index=False))
    best = max(allt, key=lambda k: allt[k].pnl.sum() if len(allt[k]) else -1e12); t = allt[best]
    print("\nBest by net (for reference, not selected for trading):", best); print(t.assign(q=pd.to_datetime(t.date).dt.to_period("Q")).groupby("q").pnl.agg(["count", "sum"]).astype(int).T.to_string())
    print("exits:", t.why.value_counts().to_dict()); print("R distribution: median R of winners %.1f, best %.1f, worst %.1f; losing-streak max %d" % (t[t.pnl > 0].r.median(), t.r.max(), t.r.min(), max((len(list(g)) for k, g in __import__('itertools').groupby(t.sort_values('date').pnl > 0) if not k), default=0)))
    pd.concat([x.assign(variant=k) for k, x in allt.items()]).to_csv("data_eq/luk_trades.csv", index=False)
