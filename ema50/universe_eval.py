import numpy as np, pandas as pd
from ema50.strategy import features
t = pd.read_csv("data_eq/trades_ucl.csv"); t["date"] = pd.to_datetime(t.date); t["setup"] = pd.to_datetime(t.setup)
d0 = pd.read_csv("data_eq/daily_univ.csv", parse_dates=["date"]); d = features(d0, "sma").set_index(["symbol", "date"]).sort_index()
by = {s: g.sort_values("date").reset_index(drop=True) for s, g in d0.groupby("symbol")}
m = pd.read_csv("data_eq/minutes_univ.csv", parse_dates=["ts"]); m["day"] = m.ts.dt.normalize(); mg = {k: g.sort_values("ts") for k, g in m.groupby(["symbol", "day"])}
def smooth(sym, setup, n):
    g = by[sym]; i = g.index[g.date == setup][0]
    if i < n: return [np.nan] * 8
    w = g.iloc[i - n:i]; rng = (w.high - w.low).replace(0, np.nan); body = (w.close - w.open).abs(); ret = np.log(w.close / w.close.shift(1)).dropna(); x = np.arange(len(w)); gp = (w.open / w.close.shift(1) - 1).abs().dropna()
    return [((rng - body) / rng).mean(), abs(np.log(w.close.iloc[-1] / w.close.iloc[0])) / ret.abs().sum(), gp.max() * 100, (ret.abs() > 3 * ret.abs().median()).mean(),
            np.corrcoef(x, np.log(w.close))[0, 1] ** 2, (rng / w.close).std() / (rng / w.close).mean(), (gp > .02).mean(), (((rng - body) / rng) > .6).mean()]
F = []
for r in t.itertuples():
    s = d.loc[(r.symbol, r.setup)]; e = d.loc[(r.symbol, r.date)]; mm = mg[(r.symbol, r.date.normalize())]; ets = pd.to_datetime(r.entry_ts)
    F.append(dict(gain=(s.close / s.pc - 1) * 100, volx=s.volume / s.av21, above50=(s.close / s.e50 - 1) * 100, gap=(e.open / s.close - 1) * 100, risk=(r.entry - r.stop) / r.entry * 100,
                  rv=100 * mm[mm.ts <= ets].volume.sum() / e.av21, adr=s.adr, tv=s.tv30, **dict(zip(["wick20", "er20", "gap20", "jump20", "r220", "rv20", "gf20", "bw20"], smooth(r.symbol, r.setup, 20))),
                  **dict(zip(["wick200", "er200", "gap200", "jump200", "r2200", "rv200", "gf200", "bw200"], smooth(r.symbol, r.setup, 200)))))
t = pd.concat([t.reset_index(drop=True), pd.DataFrame(F)], axis=1); t["win"] = (t.pnl > 0).astype(int); t["yr"] = t.date.dt.year
pr = lambda s, hi: s.rank(pct=True) if hi else 1 - s.rank(pct=True)
for n in ("20", "200"):
    t["score" + n] = 100 * pd.concat([pr(t["wick" + n], 0), pr(t["er" + n], 1), pr(t["gap" + n], 0), pr(t["jump" + n], 0), pr(t["r2" + n], 1), pr(t["rv" + n], 0), pr(t["gf" + n], 0), pr(t["bw" + n], 0)], axis=1).mean(axis=1, skipna=False)
t.to_csv("data_eq/universe_features.csv", index=False)
def row(name, s):
    if len(s) < 8: return dict(rule=name, n=len(s))
    w, l = s[s.pnl > 0].pnl.sum(), -s[s.pnl <= 0].pnl.sum(); a, b = s[s.yr == 2025], s[s.yr == 2026]
    return dict(rule=name, n=len(s), win=round(100 * s.win.mean()), net=int(s.pnl.sum()), pf=round(w / l, 2) if l else 9.9, avgR=round(s.r_mult.mean(), 2), y2025=f"{len(a)}/{int(a.pnl.sum())}", y2026=f"{len(b)}/{int(b.pnl.sum())}")
pd.set_option("display.width", 250)
R = [("ALL", t.pnl.notna()), ("H1 gap-up > 1% (SKIP)", t.gap > 1), ("H1 gap <= 1% (KEEP)", t.gap <= 1), ("H2 > 6% above 50MA (SKIP)", t.above50 > 6), ("H2 <= 6% above 50MA (KEEP)", t.above50 <= 6),
     ("H3 setup volume < 3x (SKIP)", t.volx < 3), ("H3 setup volume >= 3x (KEEP)", t.volx >= 3), ("H1+H2+H3 all pass (KEEP)", (t.gap <= 1) & (t.above50 <= 6) & (t.volx >= 3)), ("H1+H3 pass", (t.gap <= 1) & (t.volx >= 3))]
print("PRE-REGISTERED RULES (n / net by year; thresholds fixed in advance)"); print(pd.DataFrame([row(n, t[c]) for n, c in R]).to_string(index=False))
print("\nSMOOTHNESS (score: higher = smoother)")
rows = []
for n in ("20", "200"):
    u = t[t["score" + n].notna()]; q = pd.qcut(u["score" + n], 3, labels=["erratic third", "middle third", "smooth third"])
    for k in q.cat.categories: rows.append(row(f"{n}-session lookback, {k}", u[q == k]))
    print(f"corr(score{n}, R) = {u['score'+n].corr(u.r_mult):.3f}")
print(pd.DataFrame(rows).to_string(index=False))
for c in ("gap", "above50", "volx", "risk", "rv", "adr", "tv"): print(f"corr({c}, R) = {t[c].corr(t.r_mult):+.3f}", end="  ")
print()
