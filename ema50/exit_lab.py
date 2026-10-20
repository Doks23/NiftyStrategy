"""Exit lab: replay the SAME entries with different exit rules.
Sources: (1) 667 RVOL opening entries (trades_ucl.csv, entry-day path from 1-min bars), (2) 1,578 swing entries at the T+1 close (swing_univ_close.csv).
Exit spec (dict):  stop  = ("clamp", lo, hi) day-low clamped lo..hi % below entry | ("setup", cap) setup-day low capped | ("atr", k) k x ADR
                   be    = R multiple at which the stop moves to entry (or None)
                   book  = [(R multiple, fraction), ...] partial profit levels
                   trail = ("sma21", buf) | ("sma10", buf) | ("sma50", buf) | ("atr", k) close-based, exits next open | None
                   tstop = (N sessions, R) -> if after N sessions the close is below entry + R*risk, exit next open | None
Costs: 0.15% slippage on exits (entry slippage is already in the entry price), delivery charges (intraday charges when exited the same day)."""
import math, sys
import numpy as np, pandas as pd
from .strategy import features, charges

SLIP = 0.0015; RISK = 1500.0; CAP = 1_000_000


def load_daily():
    d = pd.read_csv("data_eq/daily_univ.csv", parse_dates=["date"]); d["date"] = d.date.dt.date
    f = features(d, "sma"); out = {}
    for s, g in f.groupby("symbol"):
        g = g.reset_index(drop=True)
        out[s] = dict(date=list(g.date), open=g.open.values, high=g.high.values, low=g.low.values, close=g.close.values, s10=g.e10.values, s21=g.e21.values, s50=g.e50.values, adr=g.adr.values)
    return out


def prep_opening(D):
    t = pd.read_csv("data_eq/trades_ucl.csv"); m = pd.read_csv("data_eq/minutes_univ.csv", parse_dates=["ts"]).drop_duplicates(["symbol", "ts"]); m["day"] = m.ts.dt.date
    mg = {k: g.sort_values("ts") for k, g in m.groupby(["symbol", "day"])}; out = []
    for r in t.itertuples():
        d = D[r.symbol]; dt = pd.Timestamp(r.date).date(); i = d["date"].index(dt); su = pd.Timestamp(r.setup).date(); si = d["date"].index(su)
        mm = mg[(r.symbol, dt)]; ets = pd.Timestamp(r.entry_ts); aft = mm[mm.ts > ets]; pre = mm[mm.ts <= ets]
        out.append(dict(sym=r.symbol, i0=i, e=r.entry, lod=pre.low.min(), la=aft.low.min() if len(aft) else np.inf, ha=aft.high.max() if len(aft) else -np.inf, setup_low=d["low"][si], intraday=True))
    return out


def prep_swing(D):
    t = pd.read_csv("data_eq/swing_univ_close.csv"); out = []
    for r in t.itertuples():
        d = D[r.symbol]; i = d["date"].index(pd.Timestamp(r.entry_date).date()); si = d["date"].index(pd.Timestamp(r.setup).date())
        out.append(dict(sym=r.symbol, i0=i, e=r.entry, lod=d["low"][i], la=None, ha=None, setup_low=d["low"][si], intraday=False))
    return out


def stop_price(tr, spec, d):
    e = tr["e"]
    if spec[0] == "clamp": s = min(max(tr["lod"], e * (1 - spec[2] / 100)), e * (1 - spec[1] / 100))
    elif spec[0] == "setup": s = max(tr["setup_low"], e * (1 - spec[1]))
    else: s = e * (1 - max(0.01, spec[1] * d["adr"][tr["i0"]] / 100))
    return min(s, e * 0.99)


def sim(tr, v, D):
    d = D[tr["sym"]]; e = tr["e"]; stop = stop_price(tr, v["stop"], d); R = e - stop
    q = min(math.floor(RISK / R), int(CAP / e))
    if q <= 0: return None
    left, legs, cs, be_done, pend, maxr = q, [], stop, False, False, 0.0
    done = {lv: False for lv, _ in v["book"]}; hc = e; n = len(d["close"]); i0 = tr["i0"]
    def leg(k, px, idx, why): legs.append((k, px, d["date"][idx], why, idx))
    for k in range(i0, n):
        o, h, l, c = d["open"][k], d["high"][k], d["low"][k], d["close"][k]
        if k == i0:
            if tr["la"] is not None:                                  # opening entry: only the part of the day after entry counts
                if tr["la"] <= cs: leg(left, cs * (1 - SLIP), k, "STOP"); left = 0; break
                h = tr["ha"]
            else:                                                     # entry at the close: nothing happens until tomorrow, but close-based rules apply
                h = -np.inf
        else:
            if pend: leg(left, o * (1 - SLIP), k, "TRAIL"); left = 0; break
            if o <= cs: leg(left, o * (1 - SLIP), k, "STOP" if not be_done else "BE"); left = 0; break
            if l <= cs: leg(left, cs * (1 - SLIP), k, "STOP" if not be_done else "BE"); left = 0; break
        maxr = max(maxr, (h - e) / R) if h > -np.inf else maxr
        if v["be"] is not None and not be_done and h >= e + v["be"] * R: be_done, cs = True, max(cs, e)
        for lv, fr in v["book"]:
            if not done[lv] and h >= e + lv * R:
                kq = math.floor(q * fr)
                if kq and kq < left: leg(kq, (e + lv * R) * (1 - SLIP), k, "TARGET"); left -= kq
                done[lv] = True
        hc = max(hc, c); tr_spec = v["trail"]
        if tr_spec:
            if tr_spec[0] == "atr": line = hc - tr_spec[1] * d["adr"][k] / 100 * c
            else: line = d[tr_spec[0].replace("sma", "s")][k] * (1 - tr_spec[1])
            if c < line: pend = True
        if v["tstop"] and k - i0 == v["tstop"][0] and c < e + v["tstop"][1] * R: pend = True
    if left:
        k = n - 1; leg(left, d["close"][k] * (1 - SLIP), k, "OPEN")
    gross = sum(k * (px - e) for k, px, _, _, _ in legs)
    fee = sum(charges(k * e, k * px, tr["intraday"] and idx == i0) for k, px, _, _, idx in legs)
    return dict(pnl=gross - fee, r=(gross - fee) / RISK, why=legs[-1][3], date=d["date"][i0], maxr=maxr, gross=gross)


BASE = dict(stop=("clamp", 1, 2), be=2.0, book=[(4.0, 0.25)], trail=("sma21", 0.01), tstop=None)


def variants(base=None, tag="BASELINE (current rules)"):
    base = base or BASE
    V = {tag: base}
    def add(name, **kw): V[name] = {**base, **kw}
    for lo, hi in [(1.5, 3), (2, 4), (3, 5)]: add(f"stop: day low clamped {lo}-{hi}%", stop=("clamp", lo, hi))
    for cap in (0.05, 0.08): add(f"stop: setup-day low, max {int(cap*100)}%", stop=("setup", cap))
    for k in (1.0, 1.5, 2.0): add(f"stop: {k} x ADR", stop=("atr", k))
    for b in (None, 1.0, 1.5, 3.0): add(f"breakeven at {b}R" if b else "no breakeven", be=b)
    add("no profit booking", book=[]); add("book 50% at 2R", book=[(2.0, 0.5)]); add("book 50% at 3R", book=[(3.0, 0.5)]); add("book 1/3 at 2R and 1/3 at 4R", book=[(2.0, 1 / 3), (4.0, 1 / 3)])
    add("trail: none (stop/BE/targets only)", trail=None); add("trail: SMA10 close", trail=("sma10", 0.0)); add("trail: SMA21 close, no buffer", trail=("sma21", 0.0)); add("trail: SMA21 close -3%", trail=("sma21", 0.03))
    add("trail: SMA50 close", trail=("sma50", 0.0)); add("trail: 2 x ADR below highest close", trail=("atr", 2.0)); add("trail: 3 x ADR below highest close", trail=("atr", 3.0))
    for n, r in [(3, 0.5), (5, 0.5), (5, 1.0), (10, 1.0)]: add(f"time stop: day {n} below +{r}R", tstop=(n, r))
    if base is not BASE: return V
    V["COMBO 1: stop 2-4%, BE 1.5R, 50% at 2R, SMA10 trail"] = dict(stop=("clamp", 2, 4), be=1.5, book=[(2.0, 0.5)], trail=("sma10", 0.0), tstop=None)
    V["COMBO 2: stop 1.5-3%, BE 1R, time stop day 5"] = dict(stop=("clamp", 1.5, 3), be=1.0, book=[(4.0, 0.25)], trail=("sma21", 0.01), tstop=(5, 0.5))
    V["COMBO 3: stop 1.5 x ADR, BE 1.5R, 50% at 3R, SMA21"] = dict(stop=("atr", 1.5), be=1.5, book=[(3.0, 0.5)], trail=("sma21", 0.01), tstop=None)
    V["COMBO 4: setup-low stop 8%, no BE, SMA21 -2%"] = dict(stop=("setup", 0.08), be=None, book=[(4.0, 0.25)], trail=("sma21", 0.02), tstop=None)
    return V


def summ(res):
    t = pd.DataFrame([r for r in res if r]); t["yr"] = pd.to_datetime(t.date).dt.year; w, l = t[t.pnl > 0].pnl.sum(), -t[t.pnl <= 0].pnl.sum(); a, b = t[t.yr == 2025], t[t.yr == 2026]
    eq = t.sort_values("date").pnl.cumsum()
    return dict(n=len(t), win=round(100 * (t.pnl > 0).mean()), net=int(t.pnl.sum()), pf=round(w / l, 2) if l else 9.9, avgR=round(t.r.mean(), 2), y2025=int(a.pnl.sum()), y2026=int(b.pnl.sum()), dd=int((eq - eq.cummax()).min()))


BASE_SWING = dict(stop=("setup", 0.08), be=2.0, book=[(4.0, 0.25)], trail=("sma21", 0.01), tstop=None)
BASE_ATR2 = dict(stop=("atr", 2.0), be=2.0, book=[(4.0, 0.25)], trail=("sma21", 0.01), tstop=None)


def round2(src, D):
    """Exploratory second round AROUND the best first-round stop (2 x ADR); chosen after looking, so treat as hypotheses."""
    V = {"2 x ADR stop + current exits": BASE_ATR2}
    def add(n, **kw): V[n] = {**BASE_ATR2, **kw}
    add("stop 2.5 x ADR", stop=("atr", 2.5)); add("stop 3.0 x ADR", stop=("atr", 3.0)); add("stop 2 x ADR, no breakeven", be=None); add("stop 2 x ADR, BE 3R", be=3.0)
    add("stop 2 x ADR, no booking", book=[]); add("stop 2 x ADR, 50% at 3R", book=[(3.0, 0.5)]); add("stop 2 x ADR, SMA50 trail", trail=("sma50", 0.0)); add("stop 2 x ADR, trail 3 x ADR", trail=("atr", 3.0))
    add("stop 2 x ADR, no trail", trail=None); add("stop 3 x ADR, SMA50 trail", stop=("atr", 3.0), trail=("sma50", 0.0)); add("stop 3 x ADR, no BE, SMA50 trail", stop=("atr", 3.0), be=None, trail=("sma50", 0.0))
    rows = [dict(variant=n, **summ([sim(t, v, D) for t in src])) for n, v in V.items()]; df = pd.DataFrame(rows); df["both_years_pos"] = (df.y2025 > 0) & (df.y2026 > 0); return df


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "r2":
    D = load_daily(); pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
    for sname, trs in {"OPENING-RANGE ENTRIES (667)": prep_opening(D), "SWING ENTRIES at T+1 close (1,578)": prep_swing(D)}.items():
        print(f"\n######## ROUND 2: {sname}"); df = round2(trs, D); print(df.sort_values("net", ascending=False).to_string(index=False)); df.to_csv(f"data_eq/exit_lab_r2_{'open' if 'OPEN' in sname else 'swing'}.csv", index=False)
    raise SystemExit
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "swing":
    D = load_daily(); trs = prep_swing(D); pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
    bt = pd.DataFrame([r for r in [sim(t, BASE_SWING, D) for t in trs] if r]); st = bt[bt.why == "STOP"]
    print("SWING true baseline: %d of %d exit on the initial stop; of those %.0f%% first reached +1R, %.0f%% +2R; share of all trades ever reaching +1R %.0f%%, +2R %.0f%%, +4R %.0f%%" % (len(st), len(bt), 100*(st.maxr>=1).mean(), 100*(st.maxr>=2).mean(), 100*(bt.maxr>=1).mean(), 100*(bt.maxr>=2).mean(), 100*(bt.maxr>=4).mean()))
    V = variants(BASE_SWING, "BASELINE (setup-low stop 8%, current exits)")
    for n, st_ in [("stop: 1.5 x ADR", ("atr", 1.5)), ("stop: 2.0 x ADR", ("atr", 2.0)), ("stop: setup-day low, max 5%", ("setup", .05)), ("stop: setup-day low, max 12%", ("setup", .12))]: V[n] = {**BASE_SWING, "stop": st_}
    df = pd.DataFrame([dict(variant=n, **summ([sim(t, v, D) for t in trs])) for n, v in V.items()]); df["both_years_pos"] = (df.y2025 > 0) & (df.y2026 > 0)
    print(df.sort_values("net", ascending=False).to_string(index=False)); df.to_csv("data_eq/exit_lab_swing_true.csv", index=False); raise SystemExit
if __name__ == "__main__":
    D = load_daily(); src = {"OPENING-RANGE ENTRIES (667)": prep_opening(D), "SWING ENTRIES at T+1 close (1,578)": prep_swing(D)}; V = variants(); pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
    for sname, trs in src.items():
        base = [sim(t, BASE, D) for t in trs]; bt = pd.DataFrame([r for r in base if r])
        print(f"\n######## {sname}")
        st = bt[bt.why == "STOP"]
        print("DIAGNOSTIC (baseline): %d of %d trades exit on the initial stop; of those, %.0f%% first reached +1R, %.0f%% reached +2R; median best excursion of ALL trades = %.2fR; share of all trades ever reaching +1R: %.0f%%, +2R: %.0f%%, +4R: %.0f%%" % (
            len(st), len(bt), 100 * (st.maxr >= 1).mean(), 100 * (st.maxr >= 2).mean(), bt.maxr.median(), 100 * (bt.maxr >= 1).mean(), 100 * (bt.maxr >= 2).mean(), 100 * (bt.maxr >= 4).mean()))
        rows = [dict(variant=n, **summ([sim(t, v, D) for t in trs])) for n, v in V.items()]
        df = pd.DataFrame(rows); df["both_years_pos"] = (df.y2025 > 0) & (df.y2026 > 0)
        print(df.sort_values("net", ascending=False).to_string(index=False))
        df.to_csv(f"data_eq/exit_lab_{'open' if 'OPENING' in sname else 'swing'}.csv", index=False)
