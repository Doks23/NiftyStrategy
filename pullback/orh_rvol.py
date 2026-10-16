"""Next-day ORH + RVOL entry after a contraction dot (yellow/orange dot), DhanHQ 1-min data.

Flow
  1. Daily state machine (same rules/defaults as pine/momentum_base_breakout_indicator.pine) on the
     bhavcopy panel -> contraction days inside a base.
  2. Day after each contraction (only if still in the base): pull 1-min candles from DhanHQ.
  3. Opening window 09:15-09:25 (11 x 1-min bars): volume >= rvol_pct of 50d avg daily volume.
  4. Entry: first 1-min bar after the window whose high > opening-range high; fill = max(ORH, bar open).
  5. Stop = tightest of contraction low, day low up to the entry bar, entry*(1-max_stop).
     Same-day stop checked on the remaining 1-min bars.
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
    # entry
    or_mins: int = 11; rvol_pct: float = 8.0
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


# ───────── driver ─────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/bhav/panel.pkl")
    ap.add_argument("--trade-from", default="2025-10-13")
    ap.add_argument("--rvol", type=float, default=8.0)
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
            e = orh_entry(m, ev["vavg"], ev["contr_low"], k)
            if not e:
                skipped["no_rvol_or_break"] += 1; continue
            i1 = df.index.get_loc(d1)
            if e["same_day_stop"]:
                pnl, why, xd = e["stop"] - e["entry"], "same-day stop", d1
            else:
                pnl, why, xd = follow_daily(df, i1, e["entry"], e["stop"], ev["e21"], k)
            pnl -= 2 * k.cost * e["entry"]
            R = e["entry"] - e["stop"]
            rows.append(dict(sym=s, contraction=ev["date"].date(), entry_date=d1.date(), t_entry=e["t_entry"],
                             or_vol_pct=round(e["or_vol_pct"], 1), entry=round(e["entry"], 2),
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
