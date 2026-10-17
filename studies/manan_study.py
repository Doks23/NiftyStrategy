"""Study Manan Arora's trade days: setup context + would PP VOL / RVOL->HOD have given an entry?

Entry rules are line-by-line ports of the user's Pine scripts:
  PP VOL  - "PP VOL - Trigger" (src_pp/pp_scanner.py defaults, config_combined.json 2026-09-29)
  RVOL    - "RVOL -> HOD Trigger" (pullback/orh_rvol.hod_trigger)
No exits. Daily candles come straight from NSE bhavcopy files (EQ + BE series, so trade-for-trade
names are included); 1-minute candles from DhanHQ (prev session + trade day, so the 20-bar pocket-pivot
window and the 50-bar volume EMA are warm at 09:15).

Run (repo root):  python -m studies.manan_study
"""
from __future__ import annotations
import json, os
from collections import deque
from dataclasses import dataclass
import numpy as np
import pandas as pd

from niftyvwap.data import load_env, _post, _ist
from pullback.orh_rvol import Cfg as HodCfg, hod_trigger, contraction_days, Cfg as DotCfg

RAW = "data/bhav/raw"
CACHE = "data/dhan_1m_2d"
ALIAS = {"JKBANK": "J&KBANK"}


# ───────── daily candles from bhavcopy (EQ + BE) ─────────
def daily_bars(symbols: set[str]) -> dict[str, pd.DataFrame]:
    rows = []
    for f in sorted(os.listdir(RAW)):
        if not f.endswith(".csv"):
            continue
        df = pd.read_csv(f"{RAW}/{f}")
        df.columns = [c.strip() for c in df.columns]
        if "SERIES" in df.columns:
            df = df[df.SERIES.str.strip().isin(["EQ", "BE"]) & df.SYMBOL.str.strip().isin(symbols)]
            sub = pd.DataFrame(dict(sym=df.SYMBOL.str.strip(), Open=df.OPEN, High=df.HIGH, Low=df.LOW,
                                    Close=df.CLOSE, Volume=df.TOTTRDQTY))
        else:
            df = df[df.SctySrs.str.strip().isin(["EQ", "BE"]) & df.TckrSymb.str.strip().isin(symbols)]
            sub = pd.DataFrame(dict(sym=df.TckrSymb.str.strip(), Open=df.OpnPric, High=df.HghPric, Low=df.LwPric,
                                    Close=df.ClsPric, Volume=df.TtlTradgVol))
        sub["date"] = pd.Timestamp(f[:8])
        rows.append(sub)
    long = pd.concat(rows).drop_duplicates(["sym", "date"])
    out = {}
    for s, g in long.groupby("sym"):
        g = g.set_index("date").sort_index()[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        # back-adjust splits/bonuses: opening gap down > 30% vs prior close
        r = g.Open / g.Close.shift()
        f = r.where(r < 0.7, 1.0).fillna(1.0)
        cum = f[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
        for k in ["Open", "High", "Low", "Close"]:
            g[k] = g[k] * cum
        g["Volume"] = g.Volume / cum
        out[s] = g
    return out


# ───────── DhanHQ 1-minute, previous session + trade day ─────────
def scrip_ids_all() -> dict:
    m = pd.read_csv("data/dhan_scrip_master.csv", low_memory=False)
    m = m[(m.SEM_EXM_EXCH_ID == "NSE") & (m.SEM_SEGMENT == "E") & (m.SEM_INSTRUMENT_NAME == "EQUITY")]
    m = m.sort_values("SEM_SERIES")  # BE before EQ; EQ wins on duplicates below
    return dict(zip(m.SEM_TRADING_SYMBOL.str.strip(), m.SEM_SMST_SECURITY_ID.astype(str)))


def minutes_2d(h, sec_id, sym, d_prev, d):
    os.makedirs(CACHE, exist_ok=True)
    path = f"{CACHE}/{sym.replace('&', '_')}_{d:%Y%m%d}.csv"
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["ts"])
    j = _post(h, "/charts/intraday", {"securityId": sec_id, "exchangeSegment": "NSE_EQ", "instrument": "EQUITY",
                                      "interval": "1", "oi": False,
                                      "fromDate": f"{d_prev:%Y-%m-%d} 09:15:00", "toDate": f"{d:%Y-%m-%d} 15:30:00"})
    ts = j.get("timestamp") or []
    df = pd.DataFrame(dict(ts=[_ist(t) for t in ts], open=j.get("open", []), high=j.get("high", []),
                           low=j.get("low", []), close=j.get("close", []), volume=j.get("volume", [])))
    df.to_csv(path, index=False)
    return df


# ───────── PP VOL trigger (port of "PP VOL - Trigger") ─────────
@dataclass
class PP:
    pp_len: int = 20; trig_no: int = 3; ratio_pct: float = 90.0; vol_ema_n: int = 50
    pp1_x: float = 1.5; pp2_x: float = 4.0; rvol_pp1: float = 1.0; rvol_pp2: float = 10.0
    ref_from: int = 9 * 60 + 17; sig_until: int = 11 * 60 + 15; ent_until: int = 11 * 60 + 15
    max_run: float = 5.0; max_above: float = 6.0; min_sl: float = 1.5; max_sl: float = 3.0


def pp_trigger(m: pd.DataFrame, day: pd.Timestamp, prev_close, prev_sma, avg21, k: PP = PP()):
    """Replay prev session (warm-up) + trade day. Returns dict(entry...) or dict(blocked=reason)."""
    alpha = 2.0 / (k.vol_ema_n + 1)
    win = deque(maxlen=k.pp_len)  # (vol, is_red)
    v_ema = None
    state, piv, pp1v, cum, day_hi, day_lo = 0, 0, None, 0.0, None, None
    level = arm_low = armed_t = None
    reason, n_pp, best = "", 0, 0
    m = m.sort_values("ts")
    for r in m.itertuples(index=False):
        o, h, l, c, v, t = r.open, r.high, r.low, r.close, r.volume, r.ts
        mins = t.hour * 60 + t.minute
        mr = max([vv for vv, red in win if red], default=0.0)
        is_pp = len(win) >= k.pp_len and c > o and v > mr
        v_ema = v if v_ema is None else v * alpha + v_ema * (1 - alpha)
        today = t.normalize() == day
        if today:
            if day_hi is None:
                day_hi, day_lo, cum = h, l, v
                if k.max_above > 0 and prev_close > 0 and prev_sma > 0:
                    above = (prev_close - prev_sma) / prev_sma * 100
                    if above > k.max_above:
                        state, reason = -1, f"prev close {above:.1f}% above 10d SMA (max {k.max_above}%)"
            else:
                day_hi, day_lo, cum = max(day_hi, h), min(day_lo, l), cum + v
            rvol = cum / avg21 * 100 if avg21 > 0 else np.nan
            ok_rv1 = k.rvol_pp1 <= 0 or np.isnan(rvol) or rvol >= k.rvol_pp1
            ok_rv2 = k.rvol_pp2 <= 0 or np.isnan(rvol) or rvol >= k.rvol_pp2
            ok_x1 = k.pp1_x <= 0 or v_ema <= 0 or v >= k.pp1_x * v_ema
            ok_x2 = k.pp2_x <= 0 or v_ema <= 0 or v >= k.pp2_x * v_ema
            if is_pp and mins < k.sig_until:
                n_pp += 1
            if state == 0:
                if mins >= k.sig_until:
                    state = -1
                elif is_pp and mins >= k.ref_from and ok_x1 and ok_rv1:
                    state, piv, pp1v, best = 1, 1, v, 1
            elif state == 1:
                if mins >= k.sig_until:
                    state = -1
                elif is_pp:
                    piv += 1
                    best = max(best, piv)
                    if piv >= k.trig_no:
                        ok_ratio = k.ratio_pct <= 0 or v >= pp1v * k.ratio_pct / 100
                        if ok_ratio and ok_x2 and ok_rv2:
                            run = (day_hi - prev_close) / prev_close * 100 if prev_close > 0 else 0
                            if k.max_run > 0 and run > k.max_run:
                                state, reason = -1, f"max run {run:.1f}% > {k.max_run}%"
                            else:
                                state, level, arm_low, armed_t = 2, day_hi, day_lo, t
                                armed_rvol, armed_n, armed_x = rvol, piv, v / v_ema if v_ema else np.nan
                        elif not reason:
                            why = [n for n, ok in [("vol<90% PP1", ok_ratio), ("vol<4x EMA", ok_x2), ("RVOL<10%", ok_rv2)] if not ok]
                            reason = f"PP{piv} at {t:%H:%M} failed: " + ", ".join(why)
            elif state == 2 and t > armed_t:
                if mins >= k.ent_until:
                    state, reason = -1, f"armed PP{armed_n} {armed_t:%H:%M} @ {level:.2f}, no break by 11:15"
                elif h >= level:
                    s = min(max(arm_low, level * (1 - k.max_sl / 100)), level * (1 - k.min_sl / 100))
                    return dict(entry=round(level, 2), time=f"{t:%H:%M}", armed=f"PP{armed_n} {armed_t:%H:%M}",
                                stop=round(s, 2), stop_pct=round((1 - s / level) * 100, 2),
                                rvol=round(armed_rvol, 1), vol_x=round(armed_x, 1))
        win.append((v, c < o))
    if state == 2:
        reason = f"armed PP{armed_n} {armed_t:%H:%M} @ {level:.2f}, no break by 11:15"
    if not reason:
        reason = "no qualifying PP1 (vol ≥1.5× EMA after 09:17)" if best == 0 else f"only {best} pivot(s) before 11:15 (need {k.trig_no})"
    return dict(blocked=reason)


# ───────── setup context on the day before ─────────
def atr(df, n):
    tr = pd.concat([df.High - df.Low, (df.High - df.Close.shift()).abs(), (df.Low - df.Close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def context(df: pd.DataFrame, i: int, dots: set) -> dict:
    C, H, L, V = df.Close, df.High, df.Low, df.Volume
    e10, e21, e50 = (C.ewm(span=n, adjust=False).mean() for n in (10, 21, 50))
    p = i - 1
    a5, a20 = atr(df, 5), atr(df, 20)
    rng = H - L
    hi60 = H.iloc[max(0, p - 59):p + 1].max()
    look = [df.index[j] for j in range(max(0, i - 3), i)]
    return dict(
        stack=bool(e10.iat[p] > e21.iat[p] > e50.iat[p]),
        above50=bool(C.iat[p] > e50.iat[p]),
        d10=round((C.iat[p] / e10.iat[p] - 1) * 100, 1), d21=round((C.iat[p] / e21.iat[p] - 1) * 100, 1),
        roc20=round((C.iat[p] / C.iat[max(0, p - 20)] - 1) * 100, 1), roc63=round((C.iat[p] / C.iat[max(0, p - 63)] - 1) * 100, 1),
        off_hi60=round((1 - C.iat[p] / hi60) * 100, 1),
        atr_ratio=round(a5.iat[p] / a20.iat[p], 2),
        vol_dry=round(V.iloc[p - 4:p + 1].mean() / V.iloc[max(0, p - 49):p + 1].mean(), 2),
        inside=bool(H.iat[p] < H.iat[p - 1] and L.iat[p] > L.iat[p - 1]),
        nr7=bool(rng.iat[p] <= rng.iloc[p - 6:p + 1].min()),
        dot3=bool(any(d in dots for d in look)),
        gap=round((df.Open.iat[i] / C.iat[p] - 1) * 100, 1),
        day_chg=round((C.iat[i] / C.iat[p] - 1) * 100, 1),
        day_rvol=round(V.iat[i] / V.iloc[max(0, i - 21):i].mean() * 100),
        fwd10=round((H.iloc[i + 1:i + 11].max() / C.iat[i] - 1) * 100, 1) if i + 1 < len(df) else None,
    )


def main():
    t = pd.read_csv("studies/manan_trades.csv", parse_dates=["date"])
    t["skip"] = t.symbol.str.contains(" ")
    eq = t[~t.skip].copy()
    eq["sym"] = eq.symbol.map(lambda s: ALIAS.get(s, s))
    daily = daily_bars(set(eq.sym))
    ids = scrip_ids_all()
    hdr = load_env()
    hk = HodCfg()
    rows = []
    for r in eq.itertuples():
        s, d = r.sym, r.date
        df = daily.get(s)
        base = dict(date=str(d.date()), symbol=r.symbol)
        if df is None or d not in df.index:
            rows.append(base | dict(note="no daily data")); continue
        i = df.index.get_loc(d)
        if i < 25:
            rows.append(base | dict(note=f"only {i} days of history (new listing)")); continue
        dots = {ev["date"] for ev in contraction_days(df, DotCfg())} if i >= 60 else set()
        ctx = context(df, i, dots) if i >= 25 else {}
        prev_close, prev_low = df.Close.iat[i - 1], df.Low.iat[i - 1]
        sma10 = df.Close.iloc[i - 10:i].mean()
        avg21 = df.Volume.iloc[max(0, i - 21):i].mean()
        m = minutes_2d(hdr, ids[s], s, df.index[i - 1], d) if s in ids else pd.DataFrame()
        if m.empty:
            rows.append(base | ctx | dict(note="no 1-min data")); continue
        m_day = m[m.ts.dt.normalize() == d]
        # minute prices are raw; rescale to the adjusted daily series
        f = df.Open.iat[i] / m_day.sort_values("ts").open.iat[0] if len(m_day) else 1.0
        if not 0.2 < f < 5: f = 1.0
        m = m.assign(open=m.open * f, high=m.high * f, low=m.low * f, close=m.close * f, volume=m.volume / f)
        m_day = m[m.ts.dt.normalize() == d]
        pp = pp_trigger(m, d, prev_close, sma10, avg21)
        hod = hod_trigger(m_day, avg21, prev_close, prev_low, hk)
        rows.append(base | ctx | dict(
            pp_entry=pp.get("entry"), pp_time=pp.get("time"), pp_stop_pct=pp.get("stop_pct"), pp_armed=pp.get("armed"),
            pp_block=pp.get("blocked"),
            rvol_entry=round(hod["entry"], 2) if "entry" in hod else None,
            rvol_time=f"{hod['t_entry']:%H:%M}" if "entry" in hod else None,
            rvol_stop_pct=round(hod["sl_pct"], 2) if "entry" in hod else None,
            rvol_at=round(hod["rvol"], 1) if "entry" in hod else None,
            rvol_block=hod.get("blocked")))
    out = pd.DataFrame(rows)
    out.to_csv("studies/manan_study.csv", index=False)
    # chart data for the replay page
    charts = []
    for r in out.itertuples():
        s = ALIAS.get(r.symbol, r.symbol); df = daily.get(s)
        if df is None or pd.Timestamp(r.date) not in df.index:
            continue
        i = df.index.get_loc(pd.Timestamp(r.date)); a, b = max(0, i - 40), min(len(df), i + 9)
        w = df.iloc[a:b]; C = df.Close
        dots = {str(ev["date"].date()) for ev in contraction_days(df, DotCfg())} if i >= 60 else set()
        rec = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r._asdict().items() if k != "Index"}
        rec.update(sym=s, d=[x.strftime("%Y-%m-%d") for x in w.index], o=w.Open.round(2).tolist(),
                   h=w.High.round(2).tolist(), l=w.Low.round(2).tolist(), c=w.Close.round(2).tolist(),
                   e10=C.ewm(span=10, adjust=False).mean().iloc[a:b].round(2).tolist(),
                   e21=C.ewm(span=21, adjust=False).mean().iloc[a:b].round(2).tolist(),
                   e50=C.ewm(span=50, adjust=False).mean().iloc[a:b].round(2).tolist(),
                   dots=[x for x in (str(z.date()) for z in w.index) if x in dots])
        charts.append(rec)
    json.dump(charts, open("studies/manan_charts.json", "w"), separators=(",", ":"), default=str)
    print(out.drop(columns=[c for c in out.columns if c.startswith("pp_armed")]).to_string())


if __name__ == "__main__":
    main()
