"""Setup lifecycle rules after the 50 MA pop (setup day T), then the usual RVOL opening entry.

Rules (fixed in advance):
  KILL     : any close after T that is below the low of T cancels the setup.
  VOLATILE : skip an entry day if the previous session's range% > vol_k x ADR (ADR = 20-day mean daily range %). Applies from T+2 on.
  ENTRY A  : T+1 and T+2 are open for the RVOL entry ("big move" continuation).
  ENTRY B  : T+3 .. T+8 are open only after `quiet_n` consecutive quiet sessions right before the entry day
             (quiet = range% <= quiet_k x ADR), i.e. the stock settled down and the buyers come again.
  TRIGGER  : unchanged - 8% RVOL inside the first 11 minutes and a break of the 09:15-09:17 high; stop = day low clamped 1..2%.

CLI:  python -m ema50.pop_rules elig | fetch | test
"""
import os, sys
import numpy as np, pandas as pd
from .strategy import P, features, entry_on, simulate

MAXK = 8


def load():
    d = pd.read_csv("data_eq/daily_univ.csv", parse_dates=["date"]); d["date"] = d.date.dt.date
    f = features(d, "sma"); f["rng"] = (f.high / f.low - 1) * 100
    return f


def eligible(f, st, vol_k=1.5, quiet_k=0.8, quiet_ns=(2, 3)):
    """Return DataFrame of (symbol, sid, setup, date, k, type, quiet_n_ok) candidate days."""
    by = {s: g.reset_index(drop=True) for s, g in f.groupby("symbol")}; out = []
    for r in st.itertuples():
        g = by[r.symbol]; ix = g.index[g.date == r.date]
        if not len(ix): continue
        i = ix[0]; low_T = g.low.iloc[i]
        for k in range(1, MAXK + 1):
            j = i + k
            if j >= len(g): break
            if k >= 2 and g.close.iloc[j - 1] < low_T:          # a close after the pop is below the pop-day low -> setup is dead
                break
            prev = g.iloc[j - 1]
            volatile = k >= 2 and prev.rng > vol_k * prev.adr
            if volatile: continue
            q = {n: bool(k >= n + 1 and all(g.rng.iloc[j - m] <= quiet_k * g.adr.iloc[j - m] for m in range(1, n + 1))) for n in quiet_ns}
            typ = "A" if k <= 2 else "B"
            if typ == "B" and not any(q.values()): continue
            out.append(dict(symbol=r.symbol, sid=r.sid, setup=r.date, date=g.date.iloc[j], k=k, type=typ, **{f"q{n}": q[n] for n in quiet_ns}))
    return pd.DataFrame(out)


def run_variant(f, c, p, mins, name, quiet_n):
    """First day (in order) whose RVOL trigger fires -> one trade per setup."""
    fi = f.set_index(["symbol", "date"]).sort_index(); by = {s: g.reset_index(drop=True) for s, g in f.groupby("symbol")}; trades = []
    use = c[(c.type == "A") | (c[f"q{quiet_n}"] if quiet_n else False)] if quiet_n else c[c.type == "A"]
    for (sym, setup), grp in use.groupby(["symbol", "setup"]):
        ds = by[sym]
        for r in grp.sort_values("k").itertuples():
            mm = mins.get((sym, pd.Timestamp(r.date)))
            if mm is None or mm.empty: continue
            i = ds.index[ds.date == r.date][0]
            e = entry_on(mm, ds.av21.iloc[i], p)
            if e is None: continue
            res = simulate(e, None, ds.iloc[i:].reset_index(drop=True), p, sym, r.date)
            if res: res.update(setup=setup, k=r.k, type=r.type, variant=name); trades.append(res)
            break
    return pd.DataFrame(trades)


def summarize(name, t):
    if t.empty: return dict(variant=name, n=0)
    t = t.assign(yr=pd.to_datetime(t.date).dt.year); w, l = t[t.pnl > 0].pnl.sum(), -t[t.pnl <= 0].pnl.sum(); a, b = t[t.yr == 2025], t[t.yr == 2026]
    return dict(variant=name, n=len(t), win=round(100 * (t.pnl > 0).mean()), net=int(t.pnl.sum()), gross=int(t.gross.sum()), fees=int(t.fees.sum()), pf=round(w / l, 2) if l else 9.9,
                avgR=round(t.r_mult.mean(), 2), y2025=f"{len(a)}/{int(a.pnl.sum())}", y2026=f"{len(b)}/{int(b.pnl.sum())}")


if __name__ == "__main__":
    step = sys.argv[1]; f = load(); st = pd.read_csv("data_eq/setups_cl.csv", parse_dates=["date"]); st["date"] = st.date.dt.date
    if step == "elig":
        c = eligible(f, st); c.to_csv("data_eq/pop_candidates.csv", index=False)
        print(len(st), "setups ->", len(c), "candidate days;", c.groupby("type").size().to_dict(), "| quiet2:", int(c.q2.sum()), "quiet3:", int(c.q3.sum()))
        have = pd.read_csv("data_eq/minutes_univ.csv", usecols=["symbol", "ts"]); have = set(zip(have.symbol, have.ts.str[:10]))
        need = c[[(r.symbol, str(r.date)) not in have for r in c.itertuples()]][["symbol", "sid", "date"]].drop_duplicates()
        need.to_csv("data_eq/pop_need.csv", index=False); print("1-min days still to fetch:", len(need))
    elif step == "fetch":
        from niftyvwap.data import load_env, _post, _ist
        need = pd.read_csv("data_eq/pop_need.csv"); out = "data_eq/minutes_univ.csv"; h = load_env(); buf = []
        def flush():
            global buf
            if buf:
                pd.DataFrame(buf, columns=["symbol", "ts", "open", "high", "low", "close", "volume"]).to_csv(out, mode="a", header=False, index=False); buf = []
        for i, r in enumerate(need.itertuples()):
            try:
                j = _post(h, "/charts/intraday", {"securityId": str(r.sid), "exchangeSegment": "NSE_EQ", "instrument": "EQUITY", "interval": "1",
                          "fromDate": f"{r.date} 09:00:00", "toDate": f"{r.date} 15:30:00"}, pace=0.21)
            except RuntimeError as e:
                if "401" in str(e) or "DH-901" in str(e): print("TOKEN EXPIRED at", i, flush=True); flush(); sys.exit(2)
                print(r.symbol, r.date, str(e)[:60]); continue
            for k, ts in enumerate(j.get("timestamp", [])): buf.append((r.symbol, _ist(ts), j["open"][k], j["high"][k], j["low"][k], j["close"][k], j["volume"][k]))
            if i % 150 == 0: flush(); print(i, flush=True)
        flush(); print("done")
    else:
        c = pd.read_csv("data_eq/pop_candidates.csv", parse_dates=["setup", "date"]); c["setup"] = c.setup.dt.date
        m = pd.read_csv("data_eq/minutes_univ.csv", parse_dates=["ts"]).drop_duplicates(["symbol", "ts"]); m["day"] = m.ts.dt.normalize()
        mins = {k: g.sort_values("ts") for k, g in m.groupby(["symbol", "day"])}; p = P(); res = []; allt = {}
        for name, q in [("A only: T+1..T+2", 0), ("A + B after 2 quiet days", 2), ("A + B after 3 quiet days", 3)]:
            t = run_variant(f, c, p, mins, name, q); allt[name] = t; res.append(summarize(name, t))
        pd.set_option("display.width", 250); print(pd.DataFrame(res).to_string(index=False))
        for name, t in allt.items():
            if name.startswith("A +"): print(f"\n{name}: by entry type"); print(t.groupby("type").agg(n=("pnl", "size"), win=("pnl", lambda v: round(100 * (v > 0).mean())), net=("pnl", "sum"), avgR=("r_mult", "mean")).round(2).to_string())
        pd.concat(allt.values()).to_csv("data_eq/pop_trades.csv", index=False)
