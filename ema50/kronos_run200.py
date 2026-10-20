import sys, time; sys.path.insert(0, "/home/user/shiyu-coder/kronos")
import numpy as np, pandas as pd, torch
from model import Kronos, KronosTokenizer, KronosPredictor
torch.set_num_threads(4)
L, H, N, B = 200, 10, 20, 4
tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base"); m = Kronos.from_pretrained("NeoQuasar/Kronos-small")
pr = KronosPredictor(m, tok, device="cpu", max_context=512)
d = pd.read_csv("data_eq/daily_long.csv", parse_dates=["date"]).sort_values(["symbol", "date"])
t = pd.read_csv("data_eq/trades_ck.csv", parse_dates=["setup"]).drop_duplicates(["symbol", "setup"])
jobs = []
for r in t.itertuples():
    x = d[d.symbol == r.symbol].reset_index(drop=True); ix = x.index[x.date == r.setup]
    if len(ix) and ix[0] + 1 >= L: i = ix[0]; jobs.append((r.symbol, r.setup, x.iloc[i + 1 - L:i + 1], x.close.iloc[i]))
print(len(jobs), "forecasts", flush=True); rows = []; t0 = time.time()
for b in range(0, len(jobs), B):
    ch = jobs[b:b + B]; dfs, xs, ys = [], [], []
    for s, dt, h, c0 in ch:
        for _ in range(N):
            dfs.append(h[["open", "high", "low", "close", "volume"]].reset_index(drop=True)); xs.append(h.date.reset_index(drop=True))
            ys.append(pd.Series(pd.bdate_range(h.date.iloc[-1] + pd.Timedelta(days=1), periods=H)))
    outs = pr.predict_batch(dfs, xs, ys, pred_len=H, T=1.0, top_p=0.9, sample_count=1, verbose=False)
    for k, (s, dt, h, c0) in enumerate(ch):
        P = np.array([o.close.values for o in outs[k * N:(k + 1) * N]]) / c0 - 1
        rows.append(dict(symbol=s, setup=dt.date(), up5_200=(P[:, 4] > 0).mean(), ret5_200=P[:, 4].mean(), std5_200=P[:, 4].std()))
    pd.DataFrame(rows).to_csv("data_eq/kronos200.csv", index=False); print(len(rows), "/", len(jobs), round(time.time() - t0), "s", flush=True)
print("done")
