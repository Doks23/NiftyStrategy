"""NSE daily data via Yahoo Finance + CSV cache.

Universe : NSE EQUITY_L.csv, series EQ (current listings -> survivorship bias for delisted names)
OHLCV    : Yahoo v8 chart, daily, split-adjusted OHLC (not dividend-adjusted), volume in shares
Shares   : Yahoo v7 quote sharesOutstanding (current), .NS then .BO fallback.
           Historical market cap = close_t x current shares (approximation).
"""
import io
import time as _t
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

IST = timezone(timedelta(hours=5, minutes=30))
UA = {"User-Agent": "Mozilla/5.0"}
EQUITY_L = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
QUOTE = "https://query2.finance.yahoo.com/v7/finance/quote"


def universe() -> list:
    r = requests.get(EQUITY_L, headers=UA, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.strip() for c in df.columns]
    return sorted(df[df["SERIES"].str.strip() == "EQ"]["SYMBOL"].str.strip())


def _get(s, url, params, tries=4):
    for i in range(tries):
        try:
            r = s.get(url, params=params, headers=UA, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        _t.sleep(2 ** i)
    return None


def fetch_one(s, sym: str, start: date, end: date) -> pd.DataFrame:
    p1 = int(datetime(start.year, start.month, start.day, tzinfo=IST).timestamp())
    p2 = int(datetime(end.year, end.month, end.day, 23, 59, tzinfo=IST).timestamp())
    j = _get(s, CHART.format(sym=sym), {"period1": p1, "period2": p2, "interval": "1d"})
    res = ((j or {}).get("chart") or {}).get("result")
    if not res or "timestamp" not in res[0]:
        return pd.DataFrame()
    q = res[0]["indicators"]["quote"][0]
    df = pd.DataFrame({"date": [datetime.fromtimestamp(t, IST).date() for t in res[0]["timestamp"]],
                       **{k: q[k] for k in ("open", "high", "low", "close", "volume")}})
    df = df.dropna().drop_duplicates("date", keep="last")
    df = df[(df.volume > 0) & (df.high >= df.low)]
    return df.assign(symbol=sym.removesuffix(".NS").removesuffix(".BO"))


def fetch_ohlcv(symbols, start: date, end: date, workers=8) -> pd.DataFrame:
    s = requests.Session()
    with ThreadPoolExecutor(workers) as ex:
        parts = list(ex.map(lambda x: fetch_one(s, x if x.startswith("^") else f"{x}.NS", start, end), symbols))
    out = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    print(f"ohlcv: {out.symbol.nunique()}/{len(symbols)} symbols, {len(out)} rows")
    return out


def fetch_shares(symbols) -> pd.DataFrame:
    s = requests.Session()
    s.get("https://fc.yahoo.com", headers=UA, timeout=30)
    crumb = s.get("https://query2.finance.yahoo.com/v1/test/getcrumb", headers=UA, timeout=30).text
    got = {}
    for suffix in (".NS", ".BO"):
        todo = [x for x in symbols if x not in got]
        for i in range(0, len(todo), 50):
            j = _get(s, QUOTE, {"symbols": ",".join(f"{x}{suffix}" for x in todo[i:i + 50]), "crumb": crumb})
            for r in ((j or {}).get("quoteResponse") or {}).get("result") or []:
                if r.get("sharesOutstanding"):
                    got[r["symbol"].rsplit(".", 1)[0]] = r["sharesOutstanding"]
            _t.sleep(0.3)
    print(f"shares: {len(got)}/{len(symbols)} symbols")
    return pd.DataFrame(sorted(got.items()), columns=["symbol", "shares"])


def adjust_breaks(g: pd.DataFrame, lo=0.6, hi=1 / 0.6):
    """Back-adjust un-adjusted corporate actions (splits, bonuses, demergers) Yahoo misses.
    An overnight gap open/prev_close outside [lo, hi] is treated as a corporate action (NSE circuit
    bands make real gaps that size near-impossible): prior OHLC x f, prior volume / f, f = open/prev_close.
    Returns (adjusted frame, number of breaks)."""
    g = g.reset_index(drop=True).copy()
    f = (g.open / g.close.shift()).values
    breaks = [i for i in range(1, len(g)) if not lo <= f[i] <= hi]
    for i in breaks:
        g.loc[:i - 1, ["open", "high", "low", "close"]] *= f[i]
        g.loc[:i - 1, "volume"] /= f[i]
    return g, len(breaks)


def load_cache(d: Path):
    px = pd.read_csv(d / "ohlcv.csv", parse_dates=["date"])
    sh = pd.read_csv(d / "shares.csv")
    return px, dict(zip(sh.symbol, sh.shares))
