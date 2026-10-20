"""Anchored VWAP (from daily hlc3 x volume) added to the 50 MA pop setup.
 A) FILTERS (on setup-day close): AVWAP anchored at the swing HIGH before the decline (highest high of the 60 sessions before the pop),
    AVWAP anchored at the swing LOW (lowest low of those 60 sessions), AVWAP anchored at the pop day.
 B) EXITS: trail / stop on the pop-day AVWAP.
 C) ENTRY: after the pop, buy when a session tests the pop-day AVWAP (low within 1%) and closes back above it with a green candle.
All thresholds fixed in advance."""
import sys
import numpy as np, pandas as pd
from . import exit_lab as X


def avwap(d, a):
    tp = (d["high"] + d["low"] + d["close"]) / 3; v = d["vol"]; out = np.full(len(tp), np.nan)
    cv = np.cumsum(v[a:]); out[a:] = np.cumsum(tp[a:] * v[a:]) / np.where(cv == 0, np.nan, cv); return out


def swing_anchors(d, si, n=60):
    lo = max(0, si - n); w_h = d["high"][lo:si]; w_l = d["low"][lo:si]
    return lo + int(np.argmax(w_h)), lo + int(np.argmin(w_l))


def attach(trs, D, setups):
    """setups: list of (symbol, setup_date) aligned with trs"""
    for tr, (sym, su) in zip(trs, setups):
        d = D[sym]; si = d["date"].index(pd.Timestamp(su).date()); ah, al = swing_anchors(d, si)
        tr["si"] = si; tr["av_pop"] = avwap(d, si); tr["av_hi"] = avwap(d, ah); tr["av_lo"] = avwap(d, al); tr["avwap"] = tr["av_pop"]
        c = d["close"][si]; tr["f_hi"] = c / tr["av_hi"][si] - 1; tr["f_lo"] = c / tr["av_lo"][si] - 1; tr["f_pop"] = c / tr["av_pop"][si] - 1
        tr["f_entry_pop"] = tr["e"] / tr["av_pop"][tr["i0"] - (1 if tr["la"] is not None else 0)] - 1   # no look-ahead: opening entries use the previous close
    return trs


def rep(tag, rs):
    rs = [r for r in rs if r]
    if len(rs) < 8: return dict(rule=tag, n=len(rs))
    t = pd.DataFrame(rs); t["yr"] = pd.to_datetime(t.date).dt.year; w, l = t[t.pnl > 0].pnl.sum(), -t[t.pnl <= 0].pnl.sum(); a, b = t[t.yr == 2025], t[t.yr == 2026]
    return dict(rule=tag, n=len(t), win=round(100 * (t.pnl > 0).mean()), net=int(t.pnl.sum()), pf=round(w / l, 2) if l else 9.9, avgR=round(t.r.mean(), 2), y2025=f"{len(a)}/{int(a.pnl.sum())}", y2026=f"{len(b)}/{int(b.pnl.sum())}")


if __name__ == "__main__":
    D = X.load_daily(); pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
    to = pd.read_csv("data_eq/trades_ucl.csv"); ts = pd.read_csv("data_eq/swing_univ_close.csv")
    opening = attach(X.prep_opening(D), D, list(zip(to.symbol, to.setup))); swing = attach(X.prep_swing(D), D, list(zip(ts.symbol, ts.setup)))
    B_OPEN, B_SW = X.BASE, X.BASE_SWING; BEST = dict(stop=("atr", 3.0), be=2.0, book=[(4.0, 0.25)], trail=("sma50", 0.0), tstop=None)
    for name, trs, base in (("OPENING-RANGE ENTRIES", opening, B_OPEN), ("SWING ENTRIES (T+1 close)", swing, B_SW)):
        sims = [(t, X.sim(t, base, D)) for t in trs]; ok = [(t, r) for t, r in sims if r]
        print(f"\n################ {name}: baseline exits"); rows = [rep("ALL (baseline)", [r for _, r in ok])]
        for tag, key, cuts in [("A1 setup close vs AVWAP from swing HIGH", "f_hi", [(-9, -0.15), (-0.15, 0), (0, 9)]), ("A2 setup close vs AVWAP from swing LOW", "f_lo", [(-9, 0.05), (0.05, 0.20), (0.20, 9)]),
                               ("A3 setup close vs pop-day AVWAP", "f_pop", [(-9, 0), (0, 0.03), (0.03, 9)]), ("A4 entry price vs pop-day AVWAP", "f_entry_pop", [(-9, 0), (0, 0.05), (0.05, 9)])]:
            for lo, hi in cuts: rows.append(rep(f"{tag}: {lo:+.2f}..{hi:+.2f}", [r for t, r in ok if lo <= t[key] < hi]))
        print(pd.DataFrame(rows).to_string(index=False))
        print(f"\n---- {name}: B) AVWAP exits (pop-day AVWAP)")
        V = {"current exits": base, "trail: close < pop-day AVWAP": {**base, "trail": ("avwap", 0.0)}, "trail: close < pop-day AVWAP -2%": {**base, "trail": ("avwap", 0.02)},
             "stop: pop-day AVWAP -1% (max 12%) + SMA21 trail": {**base, "stop": ("avwap", 0.01)}, "stop AVWAP -1% + trail AVWAP": {**base, "stop": ("avwap", 0.01), "trail": ("avwap", 0.0)},
             "REF best earlier: 3xADR stop + SMA50 trail": BEST, "3xADR stop + AVWAP trail": {**BEST, "trail": ("avwap", 0.0)}}
        print(pd.DataFrame([dict(rule=n, **{k: v for k, v in X.summ([X.sim(t, v, D) for t in trs]).items() if k in ("n", "win", "net", "pf", "avgR", "y2025", "y2026")}) for n, v in V.items()]).to_string(index=False))
    # C) entry on an AVWAP test
    print("\n################ C) ENTRY on a pop-day AVWAP test (daily close entry, up to 10 sessions after the pop)")
    st = pd.read_csv("data_eq/setups_cl.csv", parse_dates=["date"]); st["date"] = st.date.dt.date; ent = []
    for r in st.itertuples():
        d = D[r.symbol]; si = d["date"].index(r.date); av = avwap(d, si); low_T = d["low"][si]
        for j in range(si + 2, min(si + 10, len(d["close"]) - 1) + 1):
            if (d["close"][si + 1:j + 1] < low_T).any(): break
            if d["low"][j] <= av[j] * 1.01 and d["close"][j] > av[j] and d["close"][j] > d["open"][j]:
                e = d["close"][j] * 1.0015; ent.append(dict(sym=r.symbol, i0=j, e=e, lod=d["low"][j], la=None, ha=None, setup_low=low_T, intraday=False, avwap=av, date_=d["date"][j])); break
    print(len(ent), "entries from", len(st), "setups")
    for tr in ent: tr["i0"] = tr["i0"]
    rows = []
    for n, v in {"stop: day low (max 8%), SMA21 trail": dict(stop=("clamp", 1, 8), be=2.0, book=[(4.0, 0.25)], trail=("sma21", 0.01), tstop=None),
                 "stop: AVWAP -1%, SMA21 trail": dict(stop=("avwap", 0.01), be=2.0, book=[(4.0, 0.25)], trail=("sma21", 0.01), tstop=None),
                 "stop: AVWAP -1%, AVWAP trail": dict(stop=("avwap", 0.01), be=2.0, book=[(4.0, 0.25)], trail=("avwap", 0.0), tstop=None),
                 "stop: 3xADR, SMA50 trail": BEST}.items():
        rows.append(rep(n, [X.sim(t, v, D) for t in ent]))
    print(pd.DataFrame(rows).to_string(index=False))
