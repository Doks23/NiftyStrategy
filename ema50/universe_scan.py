"""Scan the full liquid universe 2025-01 -> 2026-10 for the user's Chartink-style setup (SMA based, loose downtrend), write setups + candidate days."""
import pandas as pd
from ema50.strategy import features, setups, candidate_days, P
d = pd.read_csv("data_eq/daily_univ.csv", parse_dates=["date"]); d["date"] = d.date.dt.date
p = P(ma="sma", min_tv=20.0, strict_down=False, stack=False)
d = features(d, "sma")
st = setups(d, p); st = st[st.date >= pd.Timestamp("2025-01-02").date()]
st.to_csv("data_eq/setups_univ.csv", index=False)
c = candidate_days(d, st, None); c.to_csv("data_eq/candidates_univ.csv", index=False)
print(len(st), "setups across", st.symbol.nunique(), "stocks;", len(c), "candidate days")
print(st.groupby(pd.to_datetime(st.date).dt.to_period("Q")).size().to_string())
