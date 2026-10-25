"""Sensitivity grid over the assumptions the video leaves open. python -m ironfly.sweep"""
import sys
from types import SimpleNamespace
import pandas as pd
from ironfly.backtest import load, run_cycle

BASE = dict(capital=200000, lots=1, maxloss=0.03, sl=0.04, hedge=0, max_adj=10, slip=1.0, exit_days_before=0)
GRID = {
    "base (Rs2L/lot, 3% wing, 4% SL, adjust on BE)": {},
    "no adjustments (plain iron fly)": dict(max_adj=0),
    "max 1 adjustment": dict(max_adj=1),
    "capital Rs1.5L/lot": dict(capital=150000),
    "capital Rs3L/lot": dict(capital=300000),
    "hedge 2x wing after adjust": dict(hedge=-2),
    "no stop loss": dict(sl=9.0),
    "exit 1 day before expiry": dict(exit_days_before=1),
    "zero slippage": dict(slip=0.0),
    "slippage 2 pts": dict(slip=2.0),
    "plain fly, capital Rs1.5L/lot": dict(max_adj=0, capital=150000),
    "plain fly, capital Rs3L/lot": dict(max_adj=0, capital=300000),
    "plain fly, slippage 2 pts": dict(max_adj=0, slip=2.0),
    "plain fly, exit 1 day before": dict(max_adj=0, exit_days_before=1),
}


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
                "costs % of gross wins": round(100 * t.costs.sum() / max(t.gross.clip(lower=0).sum(), 1))})
    return out


if __name__ == "__main__":
    data = load()
    res = {}
    for name, ov in GRID.items():
        a = SimpleNamespace(**{**BASE, **ov})
        res[name] = stats(cycles(*data, a))
    pd.set_option("display.width", 250)
    print(pd.DataFrame(res).T.to_string())
