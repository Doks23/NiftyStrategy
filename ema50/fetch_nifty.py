"""NIFTY 50 daily candles (Jan 2024 -> today) from Dhan -> data_eq/nifty_daily.csv"""
import pandas as pd
from niftyvwap.data import load_env, _post, _ist

h = load_env()
j = _post(h, "/charts/historical", {"securityId": "13", "exchangeSegment": "IDX_I", "instrument": "INDEX", "expiryCode": 0,
                                    "fromDate": "2024-01-01", "toDate": "2026-10-21"})
n = pd.DataFrame({"date": [_ist(t).date() for t in j["timestamp"]], "open": j["open"], "high": j["high"], "low": j["low"], "close": j["close"]})
n = n.drop_duplicates("date").sort_values("date")
n.to_csv("data_eq/nifty_daily.csv", index=False)
print(len(n), n.date.min(), n.date.max(), n.close.iloc[-1])
