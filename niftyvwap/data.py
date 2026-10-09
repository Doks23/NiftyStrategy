"""Dhan data access + CSV cache.

Spot      : /v2/charts/intraday     (IDX_I, securityId 13 = NIFTY 50), 5-min, <=90 days/call
Options   : /v2/charts/rollingoption (expired + live options, ATM-relative strikes),
            1-min, <=30 days/call. Strikes ATM-N..ATM+N are pulled so a strike picked at
            entry can be followed even after spot moves away from it.
Credentials: DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN env vars or ./.env.dhan
"""
import os
import time as _t
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.dhan.co/v2"
IST = timezone(timedelta(hours=5, minutes=30))
NIFTY_ID = 13


def load_env(path=".env.dhan"):
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"\''))
    cid, tok = os.environ.get("DHAN_CLIENT_ID"), os.environ.get("DHAN_ACCESS_TOKEN")
    if not cid or not tok:
        raise SystemExit("Set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN (env or .env.dhan)")
    return {"access-token": tok, "client-id": cid, "Content-Type": "application/json",
            "Accept": "application/json"}


def _post(headers, path, body, pace=0.25, tries=4):
    for i in range(tries):
        r = requests.post(f"{BASE}{path}", json=body, headers=headers, timeout=60)
        _t.sleep(pace)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            _t.sleep(2 ** i)
            continue
        raise RuntimeError(f"Dhan {path} {r.status_code}: {r.text[:300]}")
    raise RuntimeError(f"Dhan {path} failed after retries")


def _ist(epoch):
    return datetime.fromtimestamp(epoch, IST).replace(tzinfo=None)


def chunks(start: date, end: date, span: int):
    cur = start
    while cur <= end:
        nxt = min(cur + timedelta(days=span - 1), end)
        yield cur, nxt
        cur = nxt + timedelta(days=1)


def fetch_spot(h, start: date, end: date, interval=5) -> pd.DataFrame:
    rows = []
    for a, b in chunks(start, end, 85):
        j = _post(h, "/charts/intraday", {
            "securityId": str(NIFTY_ID), "exchangeSegment": "IDX_I", "instrument": "INDEX",
            "interval": str(interval), "fromDate": f"{a} 09:15:00", "toDate": f"{b} 15:30:00"})
        for i, ts in enumerate(j.get("timestamp", [])):
            rows.append((_ist(ts), j["open"][i], j["high"][i], j["low"][i], j["close"][i],
                         j.get("volume", [0] * len(j["timestamp"]))[i]))
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    return df.drop_duplicates("ts").sort_values("ts")


def fetch_options(h, start: date, end: date, flag="WEEK", code=1, width=6, interval=1) -> pd.DataFrame:
    offs = ["ATM"] + [f"ATM{s}{k}" for k in range(1, width + 1) for s in "+-"]
    rows = []
    for a, b in chunks(start, end, 28):
        for off in offs:
            for typ in ("CALL", "PUT"):
                j = _post(h, "/charts/rollingoption", {
                    "exchangeSegment": "NSE_FNO", "interval": str(interval), "securityId": NIFTY_ID,
                    "instrument": "OPTIDX", "expiryFlag": flag, "expiryCode": code,
                    "strike": off, "drvOptionType": typ,
                    "requiredData": ["open", "high", "low", "close", "volume", "strike", "spot"],
                    "fromDate": str(a), "toDate": str(b)})
                side = (j.get("data") or {}).get("ce" if typ == "CALL" else "pe") or {}
                for i, ts in enumerate(side.get("timestamp") or []):
                    rows.append((_ist(ts), "CE" if typ == "CALL" else "PE", off,
                                 side["strike"][i], side["open"][i], side["high"][i],
                                 side["low"][i], side["close"][i], side["spot"][i]))
    df = pd.DataFrame(rows, columns=["ts", "type", "offset", "strike", "open", "high", "low", "close", "spot"])
    return df.drop_duplicates(["ts", "type", "strike"]).sort_values("ts")


def load_cache(d: Path):
    sp = pd.read_csv(d / "spot_5m.csv", parse_dates=["ts"])
    op = pd.read_csv(d / "options_1m.csv", parse_dates=["ts"])
    return sp, op
