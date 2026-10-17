"""Q3 2026 (Jul-Sep) NSE winners -> launch days -> same study as Manan's trades.

Winner = largest low-to-high run inside 2026-07-01..2026-09-30 (low before high), liquid names only
(>= Rs 10 cr avg daily turnover over the quarter), ETFs excluded. Launch day = first close above the
prior 15-day high after the low (else the day after the low).
Run (repo root):  python -m studies.q3_winners
"""
import numpy as np
import pandas as pd
from pullback.orh_rvol import ETF_PAT
from studies import manan_study

Q0, Q1, TOP, MIN_TO_CR = "2026-07-01", "2026-09-30", 50, 10.0

pn = pd.read_pickle("data/bhav/panel.pkl")
H, L, C, TO = pn["High"], pn["Low"], pn["Close"], pn["Turnover"]
q = (C.index >= Q0) & (C.index <= Q1)
liq = (TO.loc[q].mean() / 1e7) >= MIN_TO_CR
rows = []
for s in C.columns[liq]:
    if pd.Series([s]).str.contains(ETF_PAT).iat[0]:
        continue
    h, l, c = H[s], L[s], C[s]
    hq, lq = h.loc[q].dropna(), l.loc[q].dropna()
    if len(hq) < 40:
        continue
    runmin = lq.cummin()
    gain = hq / runmin - 1
    peak_d = gain.idxmax(); g = gain.max()
    low_d = lq.loc[:peak_d].idxmin()
    # launch: first close above the prior 15-day high after the low
    idx = c.dropna().index
    i_low, i_pk = idx.get_loc(low_d), idx.get_loc(peak_d)
    launch = None
    for i in range(i_low + 1, i_pk + 1):
        if c.iat[i] > h.iloc[i - 15:i].max():
            launch = idx[i]; break
    launch = launch or idx[min(i_low + 1, i_pk)]
    rows.append(dict(symbol=s, run_pct=round(g * 100, 1), low_date=low_d.date(), peak_date=peak_d.date(),
                     date=launch.date(), launch_to_peak=round((h.loc[peak_d] / c.loc[launch] - 1) * 100, 1),
                     days_to_peak=i_pk - idx.get_loc(launch)))
w = pd.DataFrame(rows).sort_values("run_pct", ascending=False).head(TOP)
w.to_csv("studies/q3_winners.csv", index=False)
w[["date", "symbol"]].to_csv("studies/q3_trades.csv", index=False)
print(w.to_string(index=False))
after = {r.symbol: min(70, r.days_to_peak + 8) for r in w.itertuples()}
manan_study.main("studies/q3_trades.csv", "studies/q3", after)
