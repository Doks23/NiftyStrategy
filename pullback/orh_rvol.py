"""Next-day ORH + RVOL entry after a contraction dot (yellow/orange dot), DhanHQ 1-min data.

Flow
  1. Daily state machine (same rules/defaults as pine/momentum_base_breakout_indicator.pine) on the
     bhavcopy panel -> contraction days inside a base.
  2. Day after each contraction (only if still in the base): pull 1-min candles from DhanHQ.
  3-5. Entry = exact port of pine/rvol_hod_trigger.pine (RVOL vs 21d avg volume >= 9%, 09:18-09:26
     window, HOD break +0.05%, stop = LOD widened to >=1.5% and capped at 3.1%, gap / run-high /
     prev-low filters). Same-day stop checked on the remaining 1-min bars.
  6. Following days on daily bars: breakeven at 2R, book 25% at 4R, sell all next open after a
     close below EMA21. Costs 0.15% per side.

Run (repo root):
  python -m pullback.bhavcopy --from 2025-06-01            # daily panel (already cached in data/bhav)
  python -m pullback.orh_rvol --trade-from 2025-10-13       # needs .env.dhan (DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN)
"""
from __future__ import annotations
import argparse, io, os
from dataclasses import dataclass
from datetime import date, datetime
import numpy as np
import pandas as pd
import requests

from niftyvwap.data import load_env, _post, _ist

SCRIP_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
CACHE = "data/dhan_1m"
ETF_PAT = "ETF|BEES|GOLD|SILVER|LIQUID|NIFTY|SENSEX|^MON|IETF|CASE|^MAFANG|^MAHKTECH"


@dataclass
class Cfg:
    # momentum
    gap_pct: float = 4.0; gap_vol: float = 2.5; roc_len: int = 20; roc_min: float = 25.0
    up_len: int = 10; up_min: int = 7; mom_valid: int = 40
    # base
    pb_min: float = 5.0; pb_max: float = 25.0; min_base_bars: int = 3
    # contraction
    atr_ratio: float = 0.85; ema_comp: float = 1.0; tight_len: int = 3; tight_atr: float = 0.8
    # entry: RVOL -> HOD trigger (pine/rvol_hod_trigger.pine / opening_momentum_super_order.py)
    or_mins: int = 11; rvol_pct: float = 9.0; rvol_lookback: int = 21; start_delay: int = 3
    max_sl_pct: float = 3.1; min_sl_pct: float = 1.5; min_gap_pct: float = 0.2; max_hod_pc_pct: float = 5.0
    prev_low_breach_pct: float = 1.0; confirm_pct: float = 0.05; tick: float = 0.05
    # exits
    max_stop: float = 0.025; be_R: float = 2.0; part_R: float = 4.0; part_frac: float = 0.25; trail: int = 21
    cost: float = 0.0015; min_turnover_cr: float = 10.0


# ───────── daily setups ─────────
def _atr(h, l, c, n):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def contraction_days(df: pd.DataFrame, k: Cfg) -> list[dict]:
    """Return contraction events whose NEXT day is still in the base (eligible for ORH entry)."""
    O, H, L, C, V = (df[x] for x in ["Open", "High", "Low", "Close", "Volume"])
    e10, e21, e50 = (C.ewm(span=n, adjust=False).mean() for n in (10, 21, 50))
    aF, aS = _atr(H, L, C, 5), _atr(H, L, C, 20)
    vA = V.rolling(50, min_periods=20).mean()
    hiT, loT = H.rolling(k.tight_len).max(), L.rolling(k.tight_len).min()
    upc = (C > C.shift()).astype(int).rolling(k.up_len).sum()
    roc = (C / C.shift(k.roc_len) - 1) * 100
    mom = ((O >= H.shift() * (1 + k.gap_pct / 100)) & (V >= k.gap_vol * vA) & (C > O)) | \
          ((roc >= k.roc_min) & (upc >= k.up_min) & (C > e10) & (e10 > e21) & (e21 > e50))
    tight = (C.rolling(k.tight_len).max() - C.rolling(k.tight_len).min()) <= k.tight_atr * aS
    inside = (H < H.shift()) & (L > L.shift())
    ctest = (aF <= k.atr_ratio * aS) & ((e10 - e21).abs() <= k.ema_comp * aS) & (tight | inside)
    st, mh, mhb, out = 0, np.nan, 0, []
    for i in range(60, len(df)):
        h, l, c = H.iat[i], L.iat[i], C.iat[i]
        if mom.iat[i]:
            st = st or 1
            if np.isnan(mh) or h > mh:
                mh, mhb = h, i
        elif st and h > mh:
            mh, mhb = h, i
        if st and i - mhb > k.mom_valid:
            st, mh = 0, np.nan
        pb = np.nan if np.isnan(mh) else (mh - l) / mh * 100
        if st == 1 and pb >= k.pb_min and i - mhb >= k.min_base_bars:
            st = 2
        if st == 2 and (pb > k.pb_max or c < e50.iat[i]):
            st, mh = 0, np.nan
        if st == 2 and ctest.iat[i] and i - mhb >= k.min_base_bars and i + 1 < len(df):
            out.append(dict(i=i, date=df.index[i], next_date=df.index[i + 1], contr_low=loT.iat[i],
                            vavg=vA.iat[i], e21=e21))
    return out


# ───────── DhanHQ 1-min ─────────
def scrip_ids() -> dict:
    path = "data/dhan_scrip_master.csv"
    if not os.path.exists(path):
        os.makedirs("data", exist_ok=True)
        open(path, "wb").write(requests.get(SCRIP_URL, timeout=120).content)
    m = pd.read_csv(path, low_memory=False)
    m = m[(m.SEM_EXM_EXCH_ID == "NSE") & (m.SEM_SEGMENT == "E") & (m.SEM_SERIES == "EQ")]
    return dict(zip(m.SEM_TRADING_SYMBOL.str.strip(), m.SEM_SMST_SECURITY_ID.astype(str)))


def minute_bars(h, sec_id: str, sym: str, d: date) -> pd.DataFrame:
    os.makedirs(CACHE, exist_ok=True)
    path = f"{CACHE}/{sym}_{d:%Y%m%d}.csv"
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["ts"])
    j = _post(h, "/charts/intraday", {"securityId": sec_id, "exchangeSegment": "NSE_EQ", "instrument": "EQUITY",
                                      "interval": "1", "oi": False,
                                      "fromDate": f"{d} 09:15:00", "toDate": f"{d} 15:30:00"})
    ts = j.get("timestamp") or []
    df = pd.DataFrame(dict(ts=[_ist(t) for t in ts], open=j.get("open", []), high=j.get("high", []),
                           low=j.get("low", []), close=j.get("close", []), volume=j.get("volume", [])))
    df.to_csv(path, index=False)
    return df


# ───────── entry-day simulation (pure; unit-tested) ─────────
def orh_entry(m: pd.DataFrame, vavg: float, contr_low: float, k: Cfg):
    """Return dict(entry, stop, same_day_stop, or_vol_pct, t_entry) or None if no trade."""
    if m.empty or not vavg or np.isnan(vavg):
        return None
    m = m.sort_values("ts").reset_index(drop=True)
    t0 = m.ts.iloc[0].normalize() + pd.Timedelta(hours=9, minutes=15)
    win = m.ts < t0 + pd.Timedelta(minutes=k.or_mins)
    if not win.any():
        return None
    or_vol = m.volume[win].sum()
    if or_vol < k.rvol_pct / 100 * vavg:
        return None
    orh = m.high[win].max()
    low_pre = m.low[win].min()
    after = m[~win]
    for idx, b in after.iterrows():
        low_pre = min(low_pre, b.low)
        if b.high > orh:
            entry = max(orh, b.open)
            stop = max(contr_low, low_pre, entry * (1 - k.max_stop))
            if stop >= entry:
                return None
            rest = m.loc[idx + 1:]
            sds = (not rest.empty) and rest.low.min() <= stop
            return dict(entry=entry, stop=stop, same_day_stop=bool(sds), or_vol_pct=or_vol / vavg * 100,
                        t_entry=b.ts)
    return None


def follow_daily(df: pd.DataFrame, i_entry: int, entry: float, stop: float, e21: pd.Series, k: Cfg):
    """Manage trade on daily bars from the day after entry. Returns (pnl_per_share, why, exit_date)."""
    H, L, C, O, E = df.High.values, df.Low.values, df.Close.values, df.Open.values, e21.values
    R = entry - stop
    sp, be, part, pnl, rem = stop, False, False, 0.0, 1.0
    # entry-day close below EMA21 -> exit next open
    j = i_entry + 1
    if C[i_entry] < E[i_entry] and j < len(df):
        return (O[j] - entry), "21EMA", df.index[j]
    while j < len(df):
        if O[j] <= sp:
            return pnl + rem * (O[j] - entry), ("BE" if be else "stop"), df.index[j]
        if L[j] <= sp:
            return pnl + rem * (sp - entry), ("BE" if be else "stop"), df.index[j]
        if k.part_R and not part and H[j] >= entry + k.part_R * R:
            pnl += k.part_frac * (max(O[j], entry + k.part_R * R) - entry); rem -= k.part_frac; part = True
        if k.be_R and not be and H[j] >= entry + k.be_R * R:
            be, sp = True, max(sp, entry)
        if C[j] < E[j] and j + 1 < len(df):
            return pnl + rem * (O[j + 1] - entry), "21EMA", df.index[j + 1]
        j += 1
    return pnl + rem * (C[-1] - entry), "open", df.index[-1]


def _eval_row(ltp, hod, lod, rv, prev_close, prev_low, k: Cfg):
    """row_blockers() + build_trade_plan(): returns (blocker, entry, stop, sl_pct)."""
    e = np.ceil(hod / k.tick - 1e-9) * k.tick
    raw = min(lod, e * (1 - k.min_sl_pct / 100))
    stp = np.floor(max(raw, e * (1 - k.max_sl_pct / 100)) / k.tick + 1e-9) * k.tick
    sl = (e - stp) / e * 100 if e > stp > 0 else np.nan
    hod_pc = (hod - prev_close) / prev_close * 100
    if ltp <= prev_close * (1 + k.min_gap_pct / 100): blk = "LTP_BLW_PRV_CLS"
    elif rv < k.rvol_pct: blk = "RVOL_LOW"
    elif hod_pc > k.max_hod_pc_pct: blk = "RUN_HIGH"
    elif prev_low and lod < prev_low * (1 - k.prev_low_breach_pct / 100): blk = "PRV_LOW_BREACH"
    else: blk = ""
    return blk, e, stp, sl


def hod_trigger(m: pd.DataFrame, avg_vol: float, prev_close: float, prev_low: float, k: Cfg):
    """Port of the RVOL -> HOD Trigger Pine replay. Returns dict or dict(blocked=reason)."""
    if m.empty or not avg_vol or np.isnan(avg_vol) or not prev_close:
        return dict(blocked="NO_DATA")
    m = m.sort_values("ts").reset_index(drop=True)
    open_t = m.ts.iloc[0].normalize() + pd.Timedelta(hours=9, minutes=15)
    cum, rhod, rlod = 0.0, np.nan, np.nan
    known, pending, eligible, trig, break_blk, rv_met = np.nan, np.nan, False, None, "", False
    for r in m.itertuples(index=False):
        mins = (r.ts - open_t).total_seconds() / 60
        if mins < 0:
            continue
        o, h, l, c = r.open, r.high, r.low, r.close
        hod_b, lod_b = rhod, rlod
        cum += r.volume
        rhod = h if np.isnan(rhod) else max(rhod, h)
        rlod = l if np.isnan(rlod) else min(rlod, l)
        if trig:
            if l <= trig["stop"]:
                trig["same_day_stop"] = True
                break
            continue
        if mins >= k.or_mins:
            break
        rv = cum / avg_vol * 100
        kh = 0.0 if np.isnan(known) else known
        thresh = kh * (1 + k.confirm_pct / 100)
        if mins >= k.start_delay and eligible and kh > 0 and h >= thresh:
            lod_x = min(l if np.isnan(lod_b) else lod_b, l) if c >= o else min(o if np.isnan(lod_b) else lod_b, o)
            hod_x = max(thresh if np.isnan(hod_b) else hod_b, thresh)
            blk, e, stp, sl = _eval_row(thresh, hod_x, lod_x, rv, prev_close, prev_low, k)
            if blk == "":
                trig = dict(entry=e, stop=stp, sl_pct=sl, rvol=rv, t_entry=r.ts, same_day_stop=False)
                if c < o and l <= stp:
                    trig["same_day_stop"] = True
                    break
                continue
            break_blk = break_blk or blk
        blk, _, _, _ = _eval_row(c, rhod, rlod, rv, prev_close, prev_low, k)
        if blk == "":
            if not eligible:
                known = np.nanmax([kh, pending if not np.isnan(pending) else 0.0, rhod])
                pending, eligible = rhod, True
        else:
            eligible = False
            known = max(kh, 0.0 if np.isnan(pending) else pending)
            pending = rhod
        rv_met = rv_met or rv >= k.rvol_pct
    if trig:
        return trig
    return dict(blocked=break_blk or ("NO_BREAK" if rv_met else "RVOL_LOW"))


# ───────── driver ─────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/bhav/panel.pkl")
    ap.add_argument("--trade-from", default="2025-10-13")
    ap.add_argument("--rvol", type=float, default=9.0)
    ap.add_argument("--or-mins", type=int, default=11)
    ap.add_argument("--out", default="pullback/results_orh")
    a = ap.parse_args()
    k = Cfg(rvol_pct=a.rvol, or_mins=a.or_mins)
    hdr = load_env()
    ids = scrip_ids()
    pn = pd.read_pickle(a.panel)
    to = (pn["Turnover"] / 1e7).rolling(20, min_periods=15).mean()
    t0 = pd.Timestamp(a.trade_from)
    import yfinance as yf
    nf = yf.download("^NSEI", start=str(pn["Close"].index[0].date()), progress=False, auto_adjust=True)["Close"].squeeze()
    mkt = nf > nf.ewm(span=50, adjust=False).mean()
    rows, skipped = [], dict(no_id=0, no_rvol_or_break=0)
    syms = [s for s in pn["Close"].columns if not pd.Series([s]).str.contains(ETF_PAT).iat[0]]
    for s in syms:
        df = pd.DataFrame({x: pn[x][s] for x in ["Open", "High", "Low", "Close", "Volume"]}).dropna()
        if len(df) < 120:
            continue
        busy_until = None
        for ev in contraction_days(df, k):
            d1 = ev["next_date"]
            if d1 < t0 or (busy_until is not None and d1 <= busy_until):
                continue
            if to[s].asof(ev["date"]) < k.min_turnover_cr:
                continue
            if s not in ids:
                skipped["no_id"] += 1; continue
            m = minute_bars(hdr, ids[s], s, d1.date())
            i1 = df.index.get_loc(d1)
            if m.empty:
                skipped["no_rvol_or_break"] += 1; continue
            f = df.Open.iat[i1] / m.sort_values("ts").open.iat[0]   # panel is split-adjusted; Dhan 1-min is raw
            if not 0.2 < f < 5: f = 1.0
            m = m.assign(open=m.open * f, high=m.high * f, low=m.low * f, close=m.close * f, volume=m.volume / f)
            avg21 = df.Volume.iloc[i1 - k.rvol_lookback:i1].mean()
            e = hod_trigger(m, avg21, df.Close.iat[i1 - 1], df.Low.iat[i1 - 1], k)
            if "blocked" in e:
                skipped[e["blocked"]] = skipped.get(e["blocked"], 0) + 1; continue
            if e["same_day_stop"]:
                pnl, why, xd = e["stop"] - e["entry"], "same-day stop", d1
            else:
                pnl, why, xd = follow_daily(df, i1, e["entry"], e["stop"], ev["e21"], k)
            pnl -= 2 * k.cost * e["entry"]
            R = e["entry"] - e["stop"]
            rows.append(dict(sym=s, contraction=ev["date"].date(), entry_date=d1.date(), t_entry=e["t_entry"],
                             rvol_pct=round(e["rvol"], 1), entry=round(e["entry"], 2),
                             stop_pct=round(R / e["entry"] * 100, 2), R=pnl / R, ret_pct=pnl / e["entry"] * 100,
                             why=why, exit_date=xd.date(), mkt=bool(mkt.asof(d1))))
            busy_until = xd
    os.makedirs(a.out, exist_ok=True)
    t = pd.DataFrame(rows)
    t.to_csv(f"{a.out}/trades.csv", index=False)
    print("skipped:", skipped)
    if t.empty:
        print("no trades"); return
    t["m"] = pd.to_datetime(t.entry_date).dt.to_period("M")
    for lab, g in [("ALL", t), ("NIFTY>EMA50", t[t.mkt])]:
        if g.empty:
            continue
        w = g[g.R > 0]
        print(f"\n{lab}: n={len(g)} win={len(w)/len(g)*100:.0f}% avgW={w.R.mean():.2f}R avgL={g[g.R<=0].R.mean():.2f}R "
              f"exp={g.R.mean():+.2f}R sum={g.R.sum():+.1f}R ~cap@0.5%={g.R.sum()*0.5:+.1f}%")
        mm = g.groupby("m").agg(trades=("R", "size"), wins=("R", lambda x: int((x > 0).sum())), sum_R=("R", "sum"))
        mm["ret_pct"] = (mm.sum_R * 0.5).round(1); mm["cum_pct"] = mm.ret_pct.cumsum().round(1); mm["sum_R"] = mm.sum_R.round(1)
        print(mm.to_string())
    print("\nexits:", t.why.value_counts().to_dict())


if __name__ == "__main__":
    main()
