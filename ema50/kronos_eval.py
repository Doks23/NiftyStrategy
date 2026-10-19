import numpy as np, pandas as pd
s = pd.read_csv("data_eq/kronos_signals.csv", parse_dates=["date"])
d = pd.read_csv("data_eq/daily_ck.csv", parse_dates=["date"]).sort_values(["symbol", "date"])
rows = []
for r in s.itertuples():
    x = d[d.symbol == r.symbol].reset_index(drop=True); i = x.index[x.date == r.date][0]
    if i + 11 >= len(x): continue
    e = x.open.iloc[i + 1]
    rows.append((r.Index, x.close.iloc[i + 6] / e - 1, x.close.iloc[i + 11] / e - 1))
f = pd.DataFrame(rows, columns=["ix", "real5", "real10"]).set_index("ix"); a = s.join(f, how="inner")
COST = 0.0030 + 0.0015 * 2      # charges ~0.3% round trip + 0.15% slippage each side
print("n =", len(a), "| baseline (all setups): mean fwd5 %.2f%%  win %.0f%% | fwd10 %.2f%%" % (100*a.real5.mean(), 100*(a.real5>0).mean(), 100*a.real10.mean()))
print("corr(pred ret5, real5) = %.3f | corr(up5, real5) = %.3f" % (a.ret5_mean.corr(a.real5), a.up5.corr(a.real5)))
a["bucket"] = pd.cut(a.up5, [-.01, .2, .5, .8, 1.01], labels=["<=20% paths up", "20-50%", "50-80%", ">80% paths up"])
g = a.groupby("bucket", observed=True).agg(n=("real5", "size"), fwd5=("real5", "mean"), win5=("real5", lambda v: (v > 0).mean()), fwd10=("real10", "mean"))
g["net5_after_costs"] = g.fwd5 - COST; print((g * [1, 100, 100, 100, 100]).round(2).to_string())
a["q"] = pd.qcut(a.ret5_mean, 3, labels=["bottom third", "middle", "top third"])
g = a.groupby("q", observed=True).agg(n=("real5", "size"), pred=("ret5_mean", "mean"), fwd5=("real5", "mean"), win5=("real5", lambda v: (v > 0).mean()), fwd10=("real10", "mean"))
print((g * [1, 100, 100, 100, 100]).round(2).to_string())
