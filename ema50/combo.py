import numpy as np, pandas as pd
from ema50.strategy import features
t = pd.read_csv("data_eq/trades_ck.csv"); t["date"] = pd.to_datetime(t.date); t["setup"] = pd.to_datetime(t.setup)
k = pd.read_csv("data_eq/kronos_signals.csv", parse_dates=["date"])
d = features(pd.read_csv("data_eq/daily_ck.csv", parse_dates=["date"]), "sma").set_index(["symbol", "date"]).sort_index()
m = pd.read_csv("data_eq/minutes_ck.csv", parse_dates=["ts"]); m["day"] = m.ts.dt.normalize()
meta = pd.read_csv("/root/.claude/uploads/e83f47eb-44a3-589b-b7ae-3c4d2951ba20/3e69d5a1-Backtest_50_MA_POP.csv"); meta["setup"] = pd.to_datetime(meta.Date, format="%d-%m-%Y")
meta = meta.drop_duplicates(["Symbol", "setup"]).rename(columns={"Symbol": "symbol"})
t = t.merge(k.rename(columns={"date": "setup"}), on=["symbol", "setup"], how="left").merge(meta[["symbol", "setup", "Marketcapname", "Sector"]], on=["symbol", "setup"], how="left")
F = []
for r in t.itertuples():
    s = d.loc[(r.symbol, r.setup)]; prev = d.loc[(r.symbol, r.setup)]
    e = d.loc[(r.symbol, r.date)]
    mm = m[(m.symbol == r.symbol) & (m.day == r.date)].sort_values("ts")
    ets = pd.to_datetime(r.entry_ts); cum = mm[mm.ts <= ets].volume.sum()
    F.append(dict(gain=(s.close / s.pc - 1) * 100, volx=s.volume / s.av21, clv=(s.close - s.low) / (s.high - s.low), above50=(s.close / s.e50 - 1) * 100,
                  adr=s.adr, gap=(e.open / s.close - 1) * 100, risk=(r.entry - r.stop) / r.entry * 100, mins=(ets.hour * 60 + ets.minute) - 555,
                  rv=100 * cum / e.av21, tv=s.tv30, dist20=(s.close / s.e20 - 1) * 100))
t = pd.concat([t.reset_index(drop=True), pd.DataFrame(F)], axis=1)
t["win"] = (t.pnl > 0).astype(int); t["early"] = t.date < "2026-07-01"
t.to_csv("data_eq/combo_features.csv", index=False)
def row(name, s):
    if len(s) < 8: return None
    w, l = s[s.pnl > 0].pnl.sum(), -s[s.pnl <= 0].pnl.sum()
    e1, e2 = s[s.early], s[~s.early]
    return dict(rule=name, n=len(s), win=round(100 * s.win.mean()), net=int(s.pnl.sum()), pf=round(w / l, 2) if l else 9.99, avgR=round(s.r_mult.mean(), 2),
                early=f"{len(e1)}/{int(e1.pnl.sum())}", late=f"{len(e2)}/{int(e2.pnl.sum())}")
rules = {"ALL RVOL entries": t.pnl.notna()}
K = t.up5.notna()
for th in (.5, .7, .8, .9): rules[f"Kronos paths up >= {int(th*100)}%"] = K & (t.up5 >= th)
rules["Kronos paths up < 50%"] = K & (t.up5 < .5)
rules["Kronos pred5 > +5%"] = K & (t.ret5_mean > .05)
rules["Kronos spread std5 < 3%"] = K & (t.std5 < .03)
rules["Kronos >=80% AND spread <3%"] = K & (t.up5 >= .8) & (t.std5 < .03)
for name, c in [("gap-up at open > 1%", t.gap > 1), ("gap <= 1% (flat/gap-down)", t.gap <= 1), ("gap > 3%", t.gap > 3), ("entry day 1 only", t.k == 1), ("entry day 2-3", t.k > 1),
                ("entry in first 3 min after 09:18", t.mins <= 3), ("entry later than 3 min", t.mins > 3), ("RVOL at entry >= 12%", t.rv >= 12), ("RVOL at entry < 12%", t.rv < 12),
                ("setup volume >= 3x", t.volx >= 3), ("setup volume < 3x", t.volx < 3), ("setup gain >= 6%", t.gain >= 6), ("setup gain < 6%", t.gain < 6),
                ("close > 6% above 50MA", t.above50 > 6), ("close <= 6% above 50MA", t.above50 <= 6), ("stop risk >= 1.5%", t.risk >= 1.5), ("stop risk < 1.5%", t.risk < 1.5),
                ("Largecap", t.Marketcapname == "Largecap"), ("Midcap/Smallcap", t.Marketcapname != "Largecap"), ("tight setup clv >= 0.9", t.clv >= .9),
                ("turnover >= 100 Cr", t.tv >= 100), ("turnover < 100 Cr", t.tv < 100)]:
    rules[name] = c
res = [row(n, t[c]) for n, c in rules.items()]
pd.set_option("display.width", 250); print(pd.DataFrame([r for r in res if r]).to_string(index=False))
print("\n--- COMBINATIONS with Kronos >= 80% ---")
base = K & (t.up5 >= .8); res = []
for name, c in [("+ gap <= 1%", t.gap <= 1), ("+ gap > 1%", t.gap > 1), ("+ entry day 1", t.k == 1), ("+ RVOL >= 12%", t.rv >= 12), ("+ entry within 3 min", t.mins <= 3),
                ("+ setup vol >= 3x", t.volx >= 3), ("+ setup gain >= 6%", t.gain >= 6), ("+ above50 <= 6%", t.above50 <= 6), ("+ risk < 1.5%", t.risk < 1.5), ("+ Largecap", t.Marketcapname == "Largecap"),
                ("+ gap<=1% & risk<1.5%", (t.gap <= 1) & (t.risk < 1.5)), ("+ gap<=1% & day1", (t.gap <= 1) & (t.k == 1)), ("+ gap<=1% & RVOL>=12%", (t.gap <= 1) & (t.rv >= 12))]:
    res.append(row("Kronos>=80% " + name, t[base & c]))
print(pd.DataFrame([r for r in res if r]).to_string(index=False))
