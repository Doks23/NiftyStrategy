"""Smoothness ('controlled stock') score for each setup, from the 20 sessions BEFORE the setup day, plus Kronos forecast spread."""
import numpy as np, pandas as pd
t = pd.read_csv("data_eq/combo_features.csv"); t["date"] = pd.to_datetime(t.date); t["setup"] = pd.to_datetime(t.setup)
d = pd.read_csv("data_eq/daily_ck.csv", parse_dates=["date"]).sort_values(["symbol", "date"])
by = {s: g.reset_index(drop=True) for s, g in d.groupby("symbol")}
def metrics(sym, setup, n=20):
    g = by[sym]; i = g.index[g.date == setup][0]; w = g.iloc[i - n:i]            # sessions T-20 .. T-1, excludes the pop candle
    rng = (w.high - w.low).replace(0, np.nan); body = (w.close - w.open).abs()
    wick = ((rng - body) / rng).mean()                                           # share of each candle that is wick
    ret = np.log(w.close / w.close.shift(1)).dropna(); er = abs(np.log(w.close.iloc[-1] / w.close.iloc[0])) / ret.abs().sum()   # efficiency ratio (1 = straight line)
    gap = (w.open / w.close.shift(1) - 1).abs().dropna().max() * 100            # largest overnight gap
    jump = (ret.abs() > 3 * ret.abs().median()).mean()                           # share of days that are outsized jumps
    x = np.arange(len(w)); r2 = np.corrcoef(x, np.log(w.close))[0, 1] ** 2       # how linear the price path is
    rv = (rng / w.close).std() / (rng / w.close).mean()                          # instability of daily range
    return dict(wick=wick, er=er, gapmax=gap, jump=jump, r2=r2, rngvar=rv)
M = pd.DataFrame([metrics(r.symbol, r.setup) for r in t.itertuples()])
t = pd.concat([t, M], axis=1)
pr = lambda s, hi_good: s.rank(pct=True) if hi_good else 1 - s.rank(pct=True)
t["score_hist"] = 100 * pd.concat([pr(t.wick, 0), pr(t.er, 1), pr(t.gapmax, 0), pr(t.jump, 0), pr(t.r2, 1), pr(t.rngvar, 0)], axis=1).mean(axis=1)
t["score_kron"] = 100 * pd.concat([pr(t.wick, 0), pr(t.er, 1), pr(t.gapmax, 0), pr(t.jump, 0), pr(t.r2, 1), pr(t.rngvar, 0), pr(t.std5, 0)], axis=1).mean(axis=1, skipna=True)
t.to_csv("data_eq/smooth_features.csv", index=False)
def row(n, s):
    if len(s) < 6: return None
    w, l = s[s.pnl > 0].pnl.sum(), -s[s.pnl <= 0].pnl.sum(); e, la = s[s.early], s[~s.early]
    return dict(group=n, n=len(s), win=round(100 * s.win.mean()), net=int(s.pnl.sum()), pf=round(w / l, 2) if l else 9.99, avgR=round(s.r_mult.mean(), 2), early=f"{len(e)}/{int(e.pnl.sum())}", late=f"{len(la)}/{int(la.pnl.sum())}")
pd.set_option("display.width", 250)
print("correlation of each metric with trade R:"); print(t[["wick", "er", "gapmax", "jump", "r2", "rngvar", "std5", "score_hist", "score_kron"]].corrwith(t.r_mult).round(3).to_string())
for col in ("score_hist", "score_kron"):
    q = pd.qcut(t[col], 3, labels=["erratic (bottom third)", "middle", "smooth (top third)"]); print(f"\n== {col}")
    print(pd.DataFrame([row(str(k), t[q == k]) for k in q.cat.categories]).to_string(index=False))
print("\n== thresholds on score_hist"); print(pd.DataFrame([row(f"score_hist >= {c}", t[t.score_hist >= c]) for c in (40, 50, 60, 70)]).to_string(index=False))
print("\n== smooth + Kronos >= 80%"); print(pd.DataFrame([row(n, t[c]) for n, c in [("score_hist>=60 & Kronos>=80%", (t.score_hist >= 60) & (t.up5 >= .8)), ("score_hist>=50 & Kronos>=80%", (t.score_hist >= 50) & (t.up5 >= .8)), ("score_hist>=60 & gap<=1%", (t.score_hist >= 60) & (t.gap <= 1)), ("score_hist<40 (erratic)", t.score_hist < 40)]] ).dropna().to_string(index=False) if False else "")
