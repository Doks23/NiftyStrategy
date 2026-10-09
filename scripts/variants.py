"""Variant sweep with train/test split. Slippage + full charges always ON."""
import sys, itertools
from datetime import date, time
sys.path.insert(0, ".")
import pandas as pd
from niftyvwap import data
from niftyvwap.backtest import build_atm_lookup
from niftyvwap.strategy import Params, run_day

sp, op = data.load_cache(__import__("pathlib").Path("data6m"))
sp["day"], op["day"] = sp.ts.dt.date, op.ts.dt.date
odays = dict(tuple(op.groupby("day")))
DAYS = []
for day, s in sp.groupby("day"):
    if day in odays:
        o = odays[day]
        book = {k: g.set_index("ts")[["open", "close"]].sort_index() for k, g in o.groupby(["type", "strike"])}
        DAYS.append((day, s.set_index("ts")[["high", "low", "close", "volume"]], book, build_atm_lookup(o)))
SPLIT = date(2026, 7, 1)


def evaluate(p, days):
    tr = [t for d, s, b, a in days for t in run_day(d, s, b, a, p)]
    if not tr:
        return dict(n=0, net=0, pf=0, dd=0)
    df = pd.DataFrame([t.__dict__ for t in tr])
    daily = df.groupby("date").pnl.sum()
    eq = daily.cumsum()
    w, l = df[df.pnl > 0].pnl.sum(), -df[df.pnl <= 0].pnl.sum()
    return dict(n=len(df), net=round(df.pnl.sum()), pf=round(w / l, 2) if l else 9, dd=round((eq - eq.cummax()).min()),
                gross=round(df.gross.sum()), fees=round(df.fees.sum()))


def sweep(name, grid):
    train = [x for x in DAYS if x[0] < SPLIT]; test = [x for x in DAYS if x[0] >= SPLIT]
    rows = []
    for kw in grid:
        p = Params(**kw)
        a, b = evaluate(p, train), evaluate(p, test)
        rows.append({"cfg": str(kw), "train_net": a["net"], "train_pf": a["pf"], "test_net": b["net"], "test_pf": b["pf"],
                     "test_dd": b["dd"], "trades": a["n"] + b["n"], "all_net": a["net"] + b["net"]})
    out = pd.DataFrame(rows)
    print(f"\n### {name}"); print(out.to_string(index=False)); return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("single", "all"):
        sweep("baseline + single features", [
            {}, {"band_pct": 0.03}, {"band_pct": 0.06}, {"band_pct": 0.10},
            {"confirm_n": 2}, {"confirm_n": 3},
            {"opt_sl": 0.5}, {"opt_sl": 1.0},
            {"max_entries": 2}, {"max_entries": 3},
            {"last_entry": time(13, 0)}, {"last_entry": time(14, 0)},
            {"skip_weekdays": (1,)}, {"skip_weekdays": (3,)},
            {"roll_decay": 0.0}, {"roll_decay": 0.5}, {"roll_decay": 0.7},
        ])
    if which in ("combo", "all"):
        grid = [dict(band_pct=b, max_entries=m, confirm_n=c, roll_decay=r)
                for b in (0.03, 0.05) for m in (1, 2, 3) for c in (1, 2) for r in (0.6,)]
        out = sweep("combinations (select on TRAIN only)", grid)
