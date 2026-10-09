"""CLI:  python -m niftyvwap.backtest fetch --from 2025-01-01 --to 2025-12-31
        python -m niftyvwap.backtest run   [--lots 1 --lot-size 75 --capital 1000000 ...]"""
import argparse
from datetime import date, time
from pathlib import Path

import pandas as pd

from . import data
from .strategy import Params, run_day


def build_atm_lookup(day_opts: pd.DataFrame):
    atm = day_opts[day_opts["offset"] == "ATM"]
    lut = {(r.ts, r.type): r.strike for r in atm.itertuples()}
    return lambda ts, typ: lut.get((ts, typ))


def run(spot: pd.DataFrame, opts: pd.DataFrame, p: Params) -> pd.DataFrame:
    out = []
    spot = spot.assign(day=spot.ts.dt.date)
    opts = opts.assign(day=opts.ts.dt.date)
    odays = dict(tuple(opts.groupby("day")))
    for day, s in spot.groupby("day"):
        if day not in odays:
            continue
        o = odays[day]
        book = {k: g.set_index("ts")[["open", "close"]].sort_index()
                for k, g in o.groupby(["type", "strike"])}
        out += run_day(day, s.set_index("ts")[["high", "low", "close", "volume"]], book,
                       build_atm_lookup(o), p)
    return pd.DataFrame([t.__dict__ for t in out])


def stats(tr: pd.DataFrame, capital: float) -> dict:
    if tr.empty:
        return {"trades": 0}
    d = tr.groupby("date").pnl.sum()
    eq = d.cumsum()
    dd = (eq - eq.cummax()).min()
    w, l = tr[tr.pnl > 0].pnl, tr[tr.pnl <= 0].pnl
    return {"days": len(d), "trades": len(tr), "net_pnl": round(tr.pnl.sum(), 0),
            "win_rate_%": round(100 * len(w) / len(tr), 1),
            "profit_factor": round(w.sum() / -l.sum(), 2) if l.sum() < 0 else None,
            "avg_trade": round(tr.pnl.mean(), 0), "best_day": round(d.max(), 0),
            "worst_day": round(d.min(), 0), "max_drawdown": round(dd, 0),
            "return_%_on_capital": round(100 * tr.pnl.sum() / capital, 2),
            "loss_cap_days": int((tr.reason == "LOSS_CAP").sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "run"])
    ap.add_argument("--from", dest="start", type=date.fromisoformat)
    ap.add_argument("--to", dest="end", type=date.fromisoformat)
    ap.add_argument("--cache", default="data")
    ap.add_argument("--expiry", default="WEEK", choices=["WEEK", "MONTH"])
    ap.add_argument("--width", type=int, default=6)
    ap.add_argument("--lots", type=int, default=1)
    ap.add_argument("--lot-size", type=int, default=75)
    ap.add_argument("--capital", type=float, default=1_000_000)
    ap.add_argument("--loss-cap", type=float, default=1.8)
    ap.add_argument("--cooloff", type=int, default=3)
    ap.add_argument("--slippage", type=float, default=1.0)
    ap.add_argument("--vwap", default="volume", choices=["volume", "typical"])
    a = ap.parse_args()
    d = Path(a.cache)
    if a.cmd == "fetch":
        d.mkdir(exist_ok=True)
        h = data.load_env()
        data.fetch_spot(h, a.start, a.end).to_csv(d / "spot_5m.csv", index=False)
        data.fetch_options(h, a.start, a.end, a.expiry, 1, a.width).to_csv(d / "options_1m.csv", index=False)
        print("cached to", d)
        return
    sp, op = data.load_cache(d)
    if a.start: sp, op = sp[sp.ts.dt.date >= a.start], op[op.ts.dt.date >= a.start]
    if a.end: sp, op = sp[sp.ts.dt.date <= a.end], op[op.ts.dt.date <= a.end]
    if sp.volume.sum() == 0 and a.vwap == "volume":
        print("WARNING: index volume is zero -> VWAP falls back to unweighted typical price.")
    p = Params(cooloff_min=a.cooloff, loss_cap_pct=a.loss_cap, capital=a.capital, lots=a.lots,
               lot_size=a.lot_size, slippage=a.slippage, vwap_source=a.vwap)
    tr = run(sp, op, p)
    tr.to_csv(d / "trades.csv", index=False)
    for k, v in stats(tr, a.capital).items():
        print(f"{k:>22}: {v}")


if __name__ == "__main__":
    main()
