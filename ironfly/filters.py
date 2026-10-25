"""Regime filters on the un-adjusted 3%-rule iron fly: skip (or half-size) a month based on what was known
at the entry close. Thresholds are trailing-1y percentiles of DAILY values up to the entry day, so nothing
is fitted on the backtest months. Each filter is judged against skipping the same number of months at random.

Usage: python -m ironfly.filters   (needs data_export/ironfly_2024_2026_noadj.csv, data_if/vix_nifty.csv)"""
import numpy as np
import pandas as pd

t = pd.read_csv("data_export/ironfly_2024_2026_noadj.csv", parse_dates=["entry", "expiry"])
t = t[t.reason != "open"].reset_index(drop=True)
m = pd.read_csv("data_if/vix_nifty.csv", parse_dates=["date"], index_col="date")
lr = np.log(m.nifty).diff()
m["rv20"] = lr.rolling(20).std() * np.sqrt(252) * 100
m["ret20"] = (m.nifty / m.nifty.shift(20) - 1) * 100
m["vix_chg10"] = (m.vix / m.vix.shift(10) - 1) * 100
pct = lambda s, q: s.rolling(250, min_periods=120).quantile(q)   # trailing 1y, includes the entry day only
m["vix_med"], m["rv_med"] = pct(m.vix, .5), pct(m.rv20, .5)
m["absret_p75"] = pct(m.ret20.abs(), .75)

f = m.reindex(t.entry).reset_index(drop=True)
t = pd.concat([t, f], axis=1)

FILTERS = {
    "VIX above 1y median":            t.vix > t.vix_med,
    "VIX below 1y median":            t.vix < t.vix_med,
    "20d realised vol above 1y median": t.rv20 > t.rv_med,
    "VIX < 20d realised vol (no premium)": t.vix < t.rv20,
    "|20d Nifty move| above 1y p75 (trending)": t.ret20.abs() > t.absret_p75,
    "VIX up >15% in 10d":             t.vix_chg10 > 15,
}


def dd(r):
    eq = r.cumsum()
    return (eq - eq.cummax().clip(lower=0)).min()


def row(r, skip):
    out = {"months traded": int((~skip).sum()), "total %": round(r.sum(), 1), "maxDD %": round(dd(r), 1),
           "worst streak": int(max((len(list(g)) for k, g in __import__("itertools").groupby(r < 0) if k), default=0))}
    for y in (2024, 2025, 2026):
        out[f"{y} %"] = round(r[t.expiry.dt.year == y].sum(), 1)
    return out


rng = np.random.default_rng(7)
base = t.ret_pct
res = {"no filter": {**row(base, pd.Series(False, index=t.index)), "skipped-months P&L %": None, "random-skip p": None}}
for name, skip in FILTERS.items():
    skip = skip.fillna(False).astype(bool)
    k = int(skip.sum())
    for mode, w in (("skip", 0.0), ("half size", 0.5)):
        r = base.where(~skip, base * w)
        rand = np.array([base.where(~np.isin(np.arange(len(t)), rng.choice(len(t), k, replace=False)),
                                    base * w).sum() for _ in range(5000)])
        res[f"{name} -> {mode}"] = {**row(r, skip), "skipped-months P&L %": round(base[skip].sum(), 1),
                                    "random-skip p": round((rand >= r.sum()).mean(), 2)}

pd.set_option("display.width", 250, "display.max_colwidth", 60)
out = pd.DataFrame(res).T
print(out.to_string())
print("\nentry-day features vs outcome:")
print(t[["expiry", "ret_pct", "move_pct", "vix", "vix_med", "rv20", "ret20", "vix_chg10"]].round({c: 1 for c in t.columns if t[c].dtype.kind == "f"}).to_string(index=False))
out.to_csv("data_export/ironfly_filters.csv")
