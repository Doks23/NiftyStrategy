"""Build trade-review JSON (daily candles around each trade) for the review page."""
import json, sys
import pandas as pd
t = pd.read_csv(sys.argv[1]); pn = pd.read_pickle("data/bhav/panel.pkl")
out = []
for r in t.sort_values("entry_date").itertuples():
    s = r.sym
    df = pd.DataFrame({k: pn[k][s] for k in ["Open", "High", "Low", "Close"]}).dropna()
    e21 = df.Close.ewm(span=21, adjust=False).mean()
    i0 = df.index.get_loc(pd.Timestamp(r.contraction)); i2 = df.index.get_loc(pd.Timestamp(r.exit_date))
    a, b = max(0, i0 - 35), min(len(df), i2 + 8)
    w = df.iloc[a:b]
    out.append(dict(sym=s, dot=str(r.contraction), entry_date=str(r.entry_date), exit_date=str(r.exit_date),
                    t=str(r.t_entry)[11:16], rvol=r.rvol_pct, entry=r.entry, stop=round(r.entry * (1 - r.stop_pct / 100), 2),
                    stop_pct=r.stop_pct, R=round(r.R, 2), ret=round(r.ret_pct, 2), why=r.why, mkt=bool(r.mkt),
                    d=[x.strftime("%Y-%m-%d") for x in w.index],
                    o=w.Open.round(2).tolist(), h=w.High.round(2).tolist(), l=w.Low.round(2).tolist(),
                    c=w.Close.round(2).tolist(), e=e21.iloc[a:b].round(2).tolist()))
print(json.dumps(out, separators=(",", ":")))
