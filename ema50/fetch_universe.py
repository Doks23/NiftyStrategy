"""Daily candles (Jul 2024 -> today) for the whole liquid NSE universe -> data_eq/daily_univ.csv"""
import json, sys
import pandas as pd
from niftyvwap.data import load_env, _post, _ist
from ema50.fetch import scrip_master
shares = json.load(open(sys.argv[1])); ids = scrip_master(); h = load_env()
syms = [s for s, v in shares.items() if v["mcap_crore"] > 700 and s in ids]; print(len(syms), "symbols", flush=True); rows = []
for i, s in enumerate(syms):
    try:
        j = _post(h, "/charts/historical", {"securityId": ids[s], "exchangeSegment": "NSE_EQ", "instrument": "EQUITY", "expiryCode": 0,
                  "fromDate": "2024-07-01", "toDate": "2026-10-21"}, pace=0.21)
    except RuntimeError as e:
        print(s, str(e)[:60]); continue
    for k, ts in enumerate(j.get("timestamp", [])):
        rows.append((s, ids[s], _ist(ts).date(), j["open"][k], j["high"][k], j["low"][k], j["close"][k], j["volume"][k]))
    if i % 100 == 0: print(i, flush=True)
d = pd.DataFrame(rows, columns=["symbol", "sid", "date", "open", "high", "low", "close", "volume"])
d["shares_cr"] = d.symbol.map({s: v["shares_crore"] for s, v in shares.items()}); d.to_csv("data_eq/daily_univ.csv", index=False); print("done", len(d))
