"""Sensitivity grid over the assumptions the video leaves open. python -m ironfly.sweep"""
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
from ironfly.backtest import load, run_cycle

BASE = dict(capital=200000, lots=1, maxloss=0.03, sl=0.04, hedge=0, max_adj=10, slip=1.0, exit_days_before=0,
            adj_mode="roll")
MODES = {"video: roll short to hedge": dict(adj_mode="roll"),
         "recenter whole fly": dict(adj_mode="recenter"),
         "shift untested side to ATM": dict(adj_mode="untested"),
         "no adjustment": dict(max_adj=0)}
GRID = {}
for m, ov in MODES.items():
    GRID[m] = ov
    if "max_adj" not in ov:
        GRID[f"{m} | max 1 adj"] = {**ov, "max_adj": 1}
    GRID[f"{m} | slip 2pt"] = {**ov, "slip": 2.0}
    GRID[f"{m} | Rs3L/lot"] = {**ov, "capital": 300000}
    GRID[f"{m} | exit T-1"] = {**ov, "exit_days_before": 1}


def cycles(spot, monthly, chain, lots, a):
    tdays = list(spot.index); rows = []
    for prev, exp in zip(monthly, monthly[1:]):
        if prev < tdays[0] or prev >= tdays[-1]:
            continue
        entry = next(d for d in tdays if d > prev)
        days = [d for d in tdays if entry < d <= exp]
        r = run_cycle(entry, exp, days, spot, chain, int(lots[(entry, exp)]), a)
        if r and r["reason"] != "open":
            rows.append(r)
    return pd.DataFrame(rows)


def stats(t):
    t = t.assign(yr=pd.to_datetime(t.expiry).dt.year)
    out = {}
    for y, g in t.groupby("yr"):
        out[f"{y} %"] = round(g.ret_pct.sum(), 1)
    out.update({"n": len(t), "win%": round(100 * (t.net > 0).mean()), "total %": round(t.ret_pct.sum(), 1),
                "avg %": round(t.ret_pct.mean(), 2), "worst %": t.ret_pct.min(), "stops": (t.reason == "stop").sum(),
                "maxDD %": round((eq := t.ret_pct.cumsum()).sub(eq.cummax().clip(lower=0)).min(), 1),
                "ann %": round(12 * t.ret_pct.mean(), 1),
                "sharpe": round(np.sqrt(12) * t.ret_pct.mean() / t.ret_pct.std(), 2),
                "costs/gross wins %": round(100 * t.costs.sum() / max(t.gross.clip(lower=0).sum(), 1))})
    return out


if __name__ == "__main__":
    data = load()
    res = {}
    for name, ov in GRID.items():
        a = SimpleNamespace(**{**BASE, **ov})
        res[name] = stats(cycles(*data, a))
    pd.set_option("display.width", 250)
    print(pd.DataFrame(res).T.to_string())
