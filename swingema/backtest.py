"""CLI:  python -m swingema.backtest fetch --from 2025-01-01 --to 2026-10-14
        python -m swingema.backtest run   --from 2025-10-15 --to 2026-10-14 [--capital 1000000 ...]
`fetch` pulls ~9 months of warm-up before the test window so EMA/DEMA/Supertrend are settled."""
import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import data
from .strategy import Params, prepare, simulate

INDEX = "^NSEI"


def stats(tr: pd.DataFrame, eq: pd.DataFrame, ev: pd.DataFrame, capital: float, bench,
          all_signals=False) -> dict:
    curve = eq.set_index("date").equity
    dd = (curve / curve.cummax() - 1).min()
    if all_signals:      # no cash limit -> equity curve is meaningless; report in R
        out = {"mode": "ALL SIGNALS (no cash limit, each sized at 1R of starting capital)",
               "total_R": round(tr.r_mult.sum(), 1) if not tr.empty else 0,
               "nifty50_return_%": bench, "max_concurrent": int(eq.positions.max())}
    else:
        out = {"start_equity": capital, "end_equity": int(curve.iloc[-1]),
               "return_%": round(100 * (curve.iloc[-1] / capital - 1), 2),
               "nifty50_return_%": bench, "max_drawdown_%": round(100 * dd, 2),
               "avg_invested_%": round(100 * (eq.invested / eq.equity).mean(), 1),
               "max_positions": int(eq.positions.max())}
    if tr.empty:
        return out | {"trades": 0}
    w, l = tr[tr.pnl > 0], tr[tr.pnl <= 0]
    out |= {"trades": len(tr), "still_open": int((tr.exit_reason == "OPEN_MTM").sum()),
            "win_rate_%": round(100 * len(w) / len(tr), 1),
            "profit_factor": round(w.pnl.sum() / -l.pnl.sum(), 2) if l.pnl.sum() < 0 else None,
            "avg_R": round(tr.r_mult.mean(), 2), "median_R": round(tr.r_mult.median(), 2),
            "best_R": tr.r_mult.max(), "worst_R": tr.r_mult.min(),
            "avg_win": int(w.pnl.mean()) if len(w) else 0, "avg_loss": int(l.pnl.mean()) if len(l) else 0,
            "avg_hold_days": round(tr.hold_days.mean(), 1),
            "entries_retest/breakout": f"{(tr.trigger == 'RETEST').sum()}/{(tr.trigger == 'BREAKOUT').sum()}",
            "total_charges": int(tr.fees.sum())}
    tags = ev.tag.value_counts()
    out["fills_by_rule"] = ", ".join(f"{k}={tags.get(k, 0)}" for k in
                                     ["ENTRY", "T1", "T2", "T3", "T4", "STOP", "STOP_GAP",
                                      "A_T1", "A_T2", "A_T3", "A_T4"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "run"])
    ap.add_argument("--from", dest="start", type=date.fromisoformat)
    ap.add_argument("--to", dest="end", type=date.fromisoformat)
    ap.add_argument("--cache", default="data_swing")
    ap.add_argument("--out", default=None, help="output dir (default = cache)")
    ap.add_argument("--capital", type=float, default=1_000_000)
    ap.add_argument("--risk", type=float, default=1.0, help="% equity risked per position")
    ap.add_argument("--max-pos", type=float, default=25.0, help="max % equity per position")
    ap.add_argument("--min-turnover", type=float, default=30.0, help="30d avg traded value, Rs crore")
    ap.add_argument("--min-mcap", type=float, default=1500.0, help="market cap, Rs crore")
    ap.add_argument("--min-price", type=float, default=20.0)
    ap.add_argument("--slippage", type=float, default=0.05, help="% per market order")
    ap.add_argument("--no-charges", action="store_true")
    ap.add_argument("--no-be", action="store_true", help="keep initial stop after T1")
    ap.add_argument("--no-st-filter", action="store_true", help="allow adds when Supertrend red")
    ap.add_argument("--no-dema-filter", action="store_true")
    ap.add_argument("--no-scaling", action="store_true",
                    help="control: same entries, exit all on stop or 2 closes < 21 EMA, no adds")
    ap.add_argument("--all-signals", action="store_true",
                    help="ignore cash: take every signal, sized off starting capital (edge test)")
    a = ap.parse_args()
    d = Path(a.cache)
    if a.cmd == "fetch":
        d.mkdir(exist_ok=True)
        syms = data.universe()
        data.fetch_shares(syms).to_csv(d / "shares.csv", index=False)
        data.fetch_ohlcv(syms + [INDEX], a.start, a.end).to_csv(d / "ohlcv.csv", index=False)
        print("cached to", d)
        return

    px, shares = data.load_cache(d)
    if a.end:
        px = px[px.date.dt.date <= a.end]
    p = Params(capital=a.capital, risk_pct=a.risk, max_pos_pct=a.max_pos, min_turnover_cr=a.min_turnover,
               min_mcap_cr=a.min_mcap, min_price=a.min_price, slippage_pct=a.slippage,
               charges_on=not a.no_charges, be_after_t1=not a.no_be, st_filter=not a.no_st_filter,
               dema_filter=not a.no_dema_filter, ignore_cash=a.all_signals, scale_rules=not a.no_scaling, start=a.start, end=a.end)
    bench = None
    stocks = {}
    n_adj = 0
    for sym, g in px.groupby("symbol"):
        g, k = data.adjust_breaks(g.sort_values("date"))
        n_adj += k
        if sym == INDEX:
            w = g[g.date.dt.date >= a.start] if a.start else g
            bench = round(100 * (w.close.iloc[-1] / w.close.iloc[0] - 1), 2)
            continue
        if len(g) > p.warmup:
            stocks[sym] = prepare(g, shares.get(sym), p)
    elig = sum(bool(np.any(s["entry"][[i for i, x in enumerate(s["date"]) if not a.start or x >= a.start]]))
               for s in stocks.values())
    print(f"symbols loaded: {len(stocks)}  | with >=1 entry signal in window: {elig}"
          f"  | corporate-action gaps back-adjusted: {n_adj}")
    tr, ev, eq, skipped = simulate(stocks, p)
    o = Path(a.out or d)
    o.mkdir(exist_ok=True)
    tr.to_csv(o / "trades.csv", index=False)
    ev.to_csv(o / "fills.csv", index=False)
    eq.to_csv(o / "equity.csv", index=False)
    for k, v in (stats(tr, eq, ev, a.capital, bench, a.all_signals) | {"skipped_entries": skipped}).items():
        print(f"{k:>24}: {v}")


if __name__ == "__main__":
    main()
