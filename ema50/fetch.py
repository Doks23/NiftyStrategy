"""Fetch NSE equity daily bars (and 1-min bars for candidate days) from Dhan into ./data_eq.
Usage: python -m ema50.fetch daily SHARES_JSON  |  python -m ema50.fetch minutes"""
import json, sys, io
from datetime import date, datetime, timedelta
from pathlib import Path
import pandas as pd, requests
from niftyvwap.data import load_env, _post, _ist

OUT = Path("data_eq")
START, END = date(2025, 9, 1), date(2026, 9, 30)


def scrip_master():
    r = requests.get("https://images.dhan.co/api-data/api-scrip-master.csv", timeout=120)
    df = pd.read_csv(io.StringIO(r.text), low_memory=False)
    df = df[(df.SEM_EXM_EXCH_ID == "NSE") & (df.SEM_SEGMENT == "E") & (df.SEM_SERIES == "EQ")]
    return dict(zip(df.SEM_TRADING_SYMBOL.str.upper(), df.SEM_SMST_SECURITY_ID.astype(str)))


def daily(shares_json):
    OUT.mkdir(exist_ok=True)
    h = load_env()
    shares = json.load(open(shares_json))
    ids = scrip_master()
    syms = [s for s, v in shares.items() if v["mcap_crore"] > 700 and s in ids]
    print(len(syms), "symbols"); rows = []
    for i, s in enumerate(syms):
        try:
            j = _post(h, "/charts/historical", {"securityId": ids[s], "exchangeSegment": "NSE_EQ",
                      "instrument": "EQUITY", "expiryCode": 0, "fromDate": str(START),
                      "toDate": str(END + timedelta(days=1))}, pace=0.21)
        except RuntimeError as e:
            print(s, e); continue
        for k, ts in enumerate(j.get("timestamp", [])):
            rows.append((s, ids[s], _ist(ts).date(), j["open"][k], j["high"][k], j["low"][k],
                         j["close"][k], j["volume"][k]))
        if i % 200 == 0: print(i, flush=True)
    df = pd.DataFrame(rows, columns=["symbol", "sid", "date", "open", "high", "low", "close", "volume"])
    df["shares_cr"] = df.symbol.map({s: v["shares_crore"] for s, v in shares.items()})
    df.to_csv(OUT / "daily.csv", index=False)


def minutes():
    h = load_env()
    cand = pd.concat([pd.read_csv(f) for f in OUT.glob("candidates*.csv")]).drop_duplicates(["symbol", "date"])
    rows = []
    for i, r in enumerate(cand.itertuples()):
        d = r.date
        try:
            j = _post(h, "/charts/intraday", {"securityId": str(r.sid), "exchangeSegment": "NSE_EQ",
                      "instrument": "EQUITY", "interval": "1", "fromDate": f"{d} 09:00:00",
                      "toDate": f"{d} 15:30:00"}, pace=0.21)
        except RuntimeError as e:
            print(r.symbol, d, e); continue
        for k, ts in enumerate(j.get("timestamp", [])):
            rows.append((r.symbol, _ist(ts), j["open"][k], j["high"][k], j["low"][k], j["close"][k], j["volume"][k]))
        if i % 100 == 0: print(i, flush=True)
    df = pd.DataFrame(rows, columns=["symbol", "ts", "open", "high", "low", "close", "volume"])
    first = df.groupby([df.symbol, pd.to_datetime(df.ts).dt.date]).ts.min().astype(str).str[11:16]
    print("first-bar times:", first.value_counts().head(3).to_dict())
    df.to_csv(OUT / "minutes.csv", index=False)


if __name__ == "__main__":
    daily(sys.argv[2]) if sys.argv[1] == "daily" else minutes()
