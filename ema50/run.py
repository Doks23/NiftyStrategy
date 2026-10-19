"""python -m ema50.run scan   -> data_eq/setups.csv, candidates.csv
   python -m ema50.run test   -> data_eq/trades.csv + summary"""
import sys
from datetime import date
import pandas as pd
from .strategy import P, features, setups, candidate_days, tight, entry_on, simulate

D = "data_eq/"
T0, T1 = date(2026, 1, 1), date(2026, 9, 30)


def load(p=P()):
    d = pd.read_csv(D + "daily.csv", parse_dates=["date"])
    d["date"] = d.date.dt.date
    return features(d, p.ma)


def scan(p=P()):
    d = load(p)
    st = setups(d, p)
    st = st[(st.date >= T0) & (st.date <= T1)]
    st.to_csv(D + "setups.csv", index=False)
    c = candidate_days(d, st, p)
    c.to_csv(D + "candidates.csv", index=False)
    print(len(st), "setups;", len(c), "candidate days;", st.symbol.nunique(), "symbols")


def test(p=P(), quiet=False):
    d = load(p).set_index(["symbol", "date"]).sort_index()
    st = pd.read_csv(D + "setups.csv", parse_dates=["date"]); st["date"] = st.date.dt.date
    c = pd.read_csv(D + "candidates.csv", parse_dates=["setup", "date"])
    c["setup"], c["date"] = c.setup.dt.date, c.date.dt.date
    m = pd.read_csv(D + "minutes.csv", parse_dates=["ts"]); m["day"] = m.ts.dt.date
    mins = dict(tuple(m.groupby(["symbol", "day"])))
    trades, skipped = [], {"no_min": 0, "not_tight": 0, "no_trigger": 0}
    for (sym, setup), grp in c.groupby(["symbol", "setup"]):
        ds = d.loc[sym]
        dates = list(ds.index)
        for r in grp.sort_values("k").itertuples():
            i = dates.index(r.date)
            if r.k >= 2 and not tight(ds.iloc[i - 1], ds.iloc[i - 2], p):
                skipped["not_tight"] += 1; continue
            mm = mins.get((sym, r.date))
            if mm is None or mm.empty:
                skipped["no_min"] += 1; continue
            e = entry_on(mm, ds.iloc[i].av21, p)
            if e is None:
                skipped["no_trigger"] += 1; continue
            res = simulate(e, None, ds.iloc[i:].reset_index(), p, sym, r.date)
            if res:
                res["setup"], res["k"] = setup, r.k
                trades.append(res)
            break
    t = pd.DataFrame(trades)
    t.to_csv(D + "trades.csv", index=False)
    if not quiet:
        print("skips:", skipped)
    return t


def summary(t, capital=500000):
    if t.empty:
        return {"trades": 0}
    w, l = t[t.pnl > 0].pnl.sum(), -t[t.pnl <= 0].pnl.sum()
    eq = t.sort_values("date").groupby("date").pnl.sum().cumsum()
    return {"trades": len(t), "win_rate_%": round(100 * (t.pnl > 0).mean(), 1),
            "net_pnl": t.pnl.sum(), "gross": t.gross.sum(), "fees": t.fees.sum(),
            "profit_factor": round(w / l, 2) if l else None, "avg_R": round(t.r_mult.mean(), 2),
            "best_R": t.r_mult.max(), "worst_R": t.r_mult.min(),
            "max_dd": round((eq - eq.cummax()).min(), 0), "return_%": round(100 * t.pnl.sum() / capital, 2),
            "exits": t.exit.value_counts().to_dict()}


if __name__ == "__main__":
    if sys.argv[1] == "scan":
        scan()
    else:
        t = test()
        for k, v in summary(t).items():
            print(f"{k:>14}: {v}")
