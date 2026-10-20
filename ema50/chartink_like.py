"""Rebuild of the user's Chartink screen from what its hits actually satisfy (verified on the 425 matched hits):
 mcap>1500, close>20, SMA30(volume*close)>20 Cr as of previous session, gain>=4%, volume>2x SMA21(volume) (excl. today), below EMA50 on each of the 10 prior sessions,
 close > 0.75 x highest close of last 20 sessions."""
import numpy as np, pandas as pd
d = pd.read_csv("data_eq/daily_univ.csv", parse_dates=["date"]).sort_values(["symbol", "date"]).reset_index(drop=True)
g = d.groupby("symbol", group_keys=False)
d["ema50"] = g.close.transform(lambda s: s.ewm(span=50, adjust=False).mean()); d["pc"] = g.close.shift(1)
d["av21"] = g.volume.transform(lambda s: s.rolling(21).mean().shift(1)); d["tvv"] = d.close * d.volume
d["tv30"] = d.groupby("symbol").tvv.transform(lambda s: s.rolling(30).mean().shift(1)) / 1e7   # as of the previous session
d["b"] = (d.close < d.ema50).astype(int); d["below10"] = d.groupby("symbol").b.transform(lambda s: s.rolling(10).sum().shift(1))
d["max20"] = g.close.transform(lambda s: s.rolling(20).max()); d["mcap"] = d.shares_cr * d.close
def screen(d, near=0.75):
    return ((d.mcap > 1500) & (d.close > 20) & (d.tv30 > 20) & (d.close >= 1.04 * d.pc) & (d.volume > 2 * d.av21) & (d.below10 == 10) & (d.close > near * d.max20))
cl = pd.read_csv("/root/.claude/uploads/e83f47eb-44a3-589b-b7ae-3c4d2951ba20/3e69d5a1-Backtest_50_MA_POP.csv"); cl["date"] = pd.to_datetime(cl.Date, format="%d-%m-%Y").dt.date
cl = cl[cl.date <= pd.Timestamp("2026-10-16").date()]; b = set(zip(cl.Symbol, cl.date))
for near in (0.0, 0.75):
    st = d[screen(d, near)][["symbol", "sid", "date"]].copy(); st["date"] = st.date.dt.date
    mine = st[(st.date >= pd.Timestamp("2026-02-24").date()) & (st.date <= pd.Timestamp("2026-10-16").date())]; a = set(zip(mine.symbol, mine.date))
    print(f"near-max rule {near}: mine={len(a)} chartink={len(b)} overlap={len(a & b)} (recall {100*len(a&b)/len(b):.0f}%, precision {100*len(a&b)/len(a):.0f}%)")
st = d[screen(d, 0.75)][["symbol", "sid", "date"]].copy(); st["date"] = st.date.dt.date
st = st[st.date >= pd.Timestamp("2025-01-02").date()]; st.to_csv("data_eq/setups_cl.csv", index=False)
print(len(st), "setups since Jan 2025 across", st.symbol.nunique(), "stocks")
mine = st[(st.date >= pd.Timestamp("2026-02-24").date()) & (st.date <= pd.Timestamp("2026-10-16").date())]; a = set(zip(mine.symbol, mine.date))
print("only in Chartink:", sorted(b - a)[:5]); print("only in mine:", sorted(a - b)[:5])
print(st.groupby(pd.to_datetime(st.date).dt.to_period("Q")).size().to_string())
