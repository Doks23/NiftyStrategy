import sys, os
import pandas as pd
from niftyvwap.data import load_env, _post, _ist
from ema50.strategy import features, candidate_days, tight, P
d = pd.read_csv("data_eq/daily_univ.csv", parse_dates=["date"]); d["date"] = d.date.dt.date
f = features(d, "sma"); st = pd.read_csv("data_eq/setups_cl.csv", parse_dates=["date"]); st["date"] = st.date.dt.date
c = candidate_days(f, st, None); fi = f.set_index(["symbol", "date"]).sort_index(); p = P(); keep = []
dates = {s: list(g.date) for s, g in f.groupby("symbol")}
for r in c.itertuples():
    if r.k == 1: keep.append(True); continue
    ds = dates[r.symbol]; i = ds.index(r.date)
    keep.append(bool(tight(fi.loc[(r.symbol, ds[i - 1])], fi.loc[(r.symbol, ds[i - 2])], p)))
c = c[keep]; c.to_csv("data_eq/candidates_univ_cl.csv", index=False); print(len(c), "candidate days;", c.groupby("k").size().to_dict(), flush=True)
need = c[["symbol", "sid", "date"]].drop_duplicates(); out = "data_eq/minutes_univ.csv"
have = set()
if os.path.exists(out):
    old = pd.read_csv(out, usecols=["symbol", "ts"]); have = set(zip(old.symbol, old.ts.str[:10]))
need = need[[(r.symbol, str(r.date)) not in have for r in need.itertuples()]]; print("to fetch", len(need), flush=True)
h = load_env(); buf = []
def flush():
    global buf
    if buf:
        pd.DataFrame(buf, columns=["symbol", "ts", "open", "high", "low", "close", "volume"]).to_csv(out, mode="a", header=not os.path.exists(out), index=False); buf = []
for i, r in enumerate(need.itertuples()):
    try:
        j = _post(h, "/charts/intraday", {"securityId": str(r.sid), "exchangeSegment": "NSE_EQ", "instrument": "EQUITY", "interval": "1",
                  "fromDate": f"{r.date} 09:00:00", "toDate": f"{r.date} 15:30:00"}, pace=0.21)
    except RuntimeError as e:
        msg = str(e)
        if "401" in msg or "DH-901" in msg: print("TOKEN EXPIRED at", i, flush=True); flush(); sys.exit(2)
        print(r.symbol, r.date, msg[:60]); continue
    for k, ts in enumerate(j.get("timestamp", [])):
        buf.append((r.symbol, _ist(ts), j["open"][k], j["high"][k], j["low"][k], j["close"][k], j["volume"][k]))
    if i % 150 == 0: flush(); print(i, flush=True)
flush(); print("done")
