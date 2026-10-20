import sys, json; sys.path.insert(0, "/home/user/shiyu-coder/kronos")
import numpy as np, pandas as pd, torch
from model import Kronos, KronosTokenizer, KronosPredictor
torch.set_num_threads(4)
s = pd.read_csv("data_eq/kronos_signals.csv", parse_dates=["date"])
d = pd.read_csv("data_eq/daily_ck.csv", parse_dates=["date"]).sort_values(["symbol", "date"])
rows = []
for r in s.itertuples():
    x = d[d.symbol == r.symbol].reset_index(drop=True); ix = x.index[x.date == r.date]
    if not len(ix) or ix[0] + 6 >= len(x): continue
    i = ix[0]; real5 = x.close.iloc[i + 5] / x.close.iloc[i] - 1       # from setup close, same base as the forecast
    rows.append(dict(symbol=r.symbol, date=r.date, up5=r.up5, pred5=r.ret5_mean, real5=real5))
a = pd.DataFrame(rows)
bull, bear = a[a.up5 >= .9], a[a.up5 <= .1]
def pick(df, cond, n=3):
    g = df[cond].sort_values("date"); step = max(1, len(g) // n); return g.iloc[::step].head(n)
ex = pd.concat([pick(bull, bull.real5 > .03).assign(kind="Bullish, and it rose"), pick(bull, bull.real5 < -.02).assign(kind="Bullish, but it fell"),
                pick(bear, bear.real5 < -.02).assign(kind="Bearish, and it fell"), pick(bear, bear.real5 > .03).assign(kind="Bearish, but it rose")])
tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base"); m = Kronos.from_pretrained("NeoQuasar/Kronos-small")
pr = KronosPredictor(m, tok, device="cpu", max_context=512); L, H, N = 100, 10, 30; out = []
for r in ex.itertuples():
    x = d[d.symbol == r.symbol].reset_index(drop=True); i = x.index[x.date == r.date][0]; h = x.iloc[i + 1 - L:i + 1]
    fut = pd.Series(pd.bdate_range(h.date.iloc[-1] + pd.Timedelta(days=1), periods=H))
    outs = pr.predict_batch([h[["open", "high", "low", "close", "volume"]].reset_index(drop=True)] * N, [h.date.reset_index(drop=True)] * N, [fut] * N, pred_len=H, T=1.0, top_p=0.9, sample_count=1, verbose=False)
    P = np.array([o.close.values for o in outs]); c0 = h.close.iloc[-1]
    out.append(dict(kind=r.kind, s=r.symbol, su=r.date.strftime("%Y-%m-%d"), up5=round(r.up5, 2), pred5=round(r.pred5 * 100, 1), real5=round(r.real5 * 100, 1), c0=round(c0, 2),
                    fd=[t.strftime("%Y-%m-%d") for t in fut], med=[round(float(v), 2) for v in np.median(P, 0)], p10=[round(float(v), 2) for v in np.quantile(P, .1, 0)], p90=[round(float(v), 2) for v in np.quantile(P, .9, 0)]))
    print(out[-1]["kind"], out[-1]["s"], out[-1]["su"], out[-1]["up5"], out[-1]["pred5"], out[-1]["real5"], flush=True)
json.dump(out, open("data_eq/kronos_examples.json", "w"))
