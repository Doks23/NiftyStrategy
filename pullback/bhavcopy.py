"""NSE cash-market bhavcopy downloader -> split/bonus-adjusted OHLCV panel (EQ series, all stocks).

Old format (to 2024-07-05): archives.nseindia.com/content/historical/EQUITIES/YYYY/MON/cmDDMONYYYYbhav.csv.zip
New UDiFF format (2024-01-01 on): nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_YYYYMMDD_F_0000.csv.zip

Adjustment: NSE publishes PREVCLOSE already adjusted on the ex-date, so
factor_t = PREVCLOSE_t / CLOSE_{t-1}; prices before t are multiplied by the product of later factors.
"""
from __future__ import annotations
import argparse, io, os, time, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import requests

RAW = "data/bhav/raw"
UA = {"User-Agent": "Mozilla/5.0", "Accept": "*/*"}
NEW_FROM = pd.Timestamp("2024-07-08")


def url(d: pd.Timestamp) -> str:
    if d >= NEW_FROM:
        return f"https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
    mon = d.strftime("%b").upper()
    return f"https://archives.nseindia.com/content/historical/EQUITIES/{d:%Y}/{mon}/cm{d:%d}{mon}{d:%Y}bhav.csv.zip"


def fetch(d: pd.Timestamp) -> str:
    path = f"{RAW}/{d:%Y%m%d}.csv"
    if os.path.exists(path) or os.path.exists(path + ".none"):
        return "cached"
    for attempt in range(4):
        try:
            r = requests.get(url(d), headers=UA, timeout=30)
            if r.status_code == 403 and attempt >= 1:
                return "forbidden"  # NSE does not serve this date (e.g. pre-2016 archives)
            if r.status_code == 404 or (r.status_code == 200 and not r.content.startswith(b"PK")):
                open(path + ".none", "w").close()  # holiday
                return "holiday"
            if r.status_code == 200:
                z = zipfile.ZipFile(io.BytesIO(r.content))
                with open(path, "wb") as f:
                    f.write(z.read(z.namelist()[0]))
                return "ok"
        except Exception:
            pass
        time.sleep(2 ** attempt)
    return "fail"


def download(start: str, end: str, workers: int = 4):
    os.makedirs(RAW, exist_ok=True)
    days = pd.bdate_range(start, end)
    res = []
    with ThreadPoolExecutor(workers) as ex:
        for n, r in enumerate(ex.map(fetch, days), 1):
            res.append(r)
            if n % 250 == 0:
                print(n, "/", len(days), {k: res.count(k) for k in set(res)}, flush=True)
    print({k: res.count(k) for k in set(res)})


def parse(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "SERIES" in df.columns:
        df = df[df.SERIES.str.strip() == "EQ"]
        out = pd.DataFrame(dict(sym=df.SYMBOL.str.strip(), o=df.OPEN, h=df.HIGH, l=df.LOW, c=df.CLOSE,
                                pc=df.PREVCLOSE, v=df.TOTTRDQTY, val=df.TOTTRDVAL))
    else:
        df = df[df.SctySrs.str.strip() == "EQ"]
        out = pd.DataFrame(dict(sym=df.TckrSymb.str.strip(), o=df.OpnPric, h=df.HghPric, l=df.LwPric,
                                c=df.ClsPric, pc=df.PrvsClsgPric, v=df.TtlTradgVol, val=df.TtlTrfVal))
    out["date"] = pd.Timestamp(os.path.basename(path)[:8])
    return out


def adj_factor(close: pd.DataFrame, prevclose: pd.DataFrame, tol: float = 0.02) -> pd.DataFrame:
    """Multiplier per (date, sym) that back-adjusts raw prices for splits/bonuses.
    Ex-date factor = PREVCLOSE_t / last actual close; ignore <2% noise; product of factors strictly after t."""
    f = prevclose / close.ffill().shift(1)
    f = f.where((f - 1).abs() > tol, 1.0).fillna(1.0)
    return f[::-1].cumprod()[::-1].shift(-1).fillna(1.0)


def yf_split_factors(syms, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Split/bonus ratios from yfinance (new UDiFF bhavcopy does NOT adjust PrvsClsgPric)."""
    import yfinance as yf
    out = pd.DataFrame(1.0, index=dates, columns=syms)
    tick = [s + ".NS" for s in syms]
    for k in range(0, len(tick), 200):
        d = yf.download(tick[k:k + 200], start=dates[0], end=dates[-1] + pd.Timedelta(days=2), actions=True,
                        progress=False, group_by="ticker", threads=True)
        for t in tick[k:k + 200]:
            try:
                sp = d[t]["Stock Splits"].dropna()
            except KeyError:
                continue
            sp.index = pd.DatetimeIndex(sp.index).tz_localize(None)
            for dt, r in sp[sp > 0].items():
                if dt in out.index:
                    out.loc[dt, t[:-3]] = 1.0 / r  # multiplier for earlier prices
    return out


def gap_split_factors(close: pd.DataFrame, open_: pd.DataFrame, tol: float = 0.04) -> pd.DataFrame:
    """Fallback: overnight ratio within tol of a common split/bonus ratio -> treat as corporate action."""
    r = close.ffill().shift(1) / open_
    f = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    for k in (2, 3, 4, 5, 10, 1.5):
        f = f.mask(((r / k - 1).abs() < tol) & (f == 1.0), 1.0 / k)
    return f


def build_panel(start="2000-01-01", out="data/bhav/panel.pkl"):
    s0 = pd.Timestamp(start).strftime("%Y%m%d")
    files = sorted(f for f in os.listdir(RAW) if f.endswith(".csv") and f[:8] >= s0)
    long = pd.concat([parse(f"{RAW}/{f}") for f in files], ignore_index=True)
    long = long.drop_duplicates(["date", "sym"])
    P = {k: long.pivot(index="date", columns="sym", values=k).astype(float) for k in ["o", "h", "l", "c", "pc", "v", "val"]}
    cum = adj_factor(P["c"], P["pc"])  # old-format files (PREVCLOSE adjusted on ex-date)
    new = P["c"].index >= NEW_FROM
    if new.any():
        liquid = list(P["val"].columns[(P["val"].loc[new].mean() / 1e7) >= 1])  # >= Rs 1 cr/day
        f = gap_split_factors(P["c"], P["o"]).loc[new]
        yfx = yf_split_factors(liquid, P["c"].index[new])
        f[liquid] = yfx.where(yfx != 1.0, f[liquid])
        f = f.reindex(P["c"].index).fillna(1.0)
        cum = cum * f[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
    # second pass: any remaining opening gap down >30% vs adjusted prior close is a corporate action
    # (bonus+split combos yfinance under-reports, demergers); intraday crashes are left alone
    ac = (P["c"] * cum).ffill().shift(1)
    g = (P["o"] * cum) / ac
    f2 = g.where(g < 0.7, 1.0).fillna(1.0)
    cum = cum * f2[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
    panel = dict(Open=P["o"] * cum, High=P["h"] * cum, Low=P["l"] * cum, Close=P["c"] * cum,
                 Volume=P["v"] / cum, Turnover=P["val"])
    pd.to_pickle(panel, out)
    print("panel", panel["Close"].shape, "->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="2025-06-01")
    ap.add_argument("--to", dest="end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    a = ap.parse_args()
    download(a.start, a.end)
    build_panel(a.start)
