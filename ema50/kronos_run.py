"""Kronos-small forecasts for every Chartink setup (setup-day close -> next 10 sessions, N sampled paths)."""
import sys, time; sys.path.insert(0, "/home/user/shiyu-coder/kronos")
import numpy as np, pandas as pd, torch
from model import Kronos, KronosTokenizer, KronosPredictor
torch.set_num_threads(4)
L, H, N, B = 100, 10, 30, 8
tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base"); m = Kronos.from_pretrained("NeoQuasar/Kronos-small")
pr = KronosPredictor(m, tok, device="cpu", max_context=512)
d = pd.read_csv("data_eq/daily_ck.csv", parse_dates=["date"]).sort_values(["symbol", "date"])
st = pd.read_csv("data_eq/setups_ck.csv", parse_dates=["date"]).drop_duplicates(["symbol", "date"])
jobs = []
for r in st.itertuples():
    x = d[d.symbol == r.symbol].reset_index(drop=True); ix = x.index[x.date == r.date]
    if len(ix) and ix[0] + 1 >= L:
        i = ix[0]; jobs.append((r.symbol, r.date, x.iloc[i + 1 - L:i + 1], x.close.iloc[i]))
print(len(jobs), "forecasts", flush=True); rows = []; t0 = time.time()
for b in range(0, len(jobs), B):
    ch = jobs[b:b + B]; dfs, xs, ys = [], [], []
    for s, dt, h, c0 in ch:
        for _ in range(N):
            dfs.append(h[["open", "high", "low", "close", "volume"]].reset_index(drop=True)); xs.append(h.date.reset_index(drop=True))
            ys.append(pd.Series(pd.bdate_range(h.date.iloc[-1] + pd.Timedelta(days=1), periods=H)))
    outs = pr.predict_batch(dfs, xs, ys, pred_len=H, T=1.0, top_p=0.9, sample_count=1, verbose=False)
    for k, (s, dt, h, c0) in enumerate(ch):
        P = np.array([o.close.values for o in outs[k * N:(k + 1) * N]]) / c0 - 1       # N x H
        rows.append(dict(symbol=s, date=dt.date(), ret5_mean=P[:, 4].mean(), up5=(P[:, 4] > 0).mean(), ret10_mean=P[:, 9].mean(),
                         up10=(P[:, 9] > 0).mean(), std5=P[:, 4].std(), q25_5=np.quantile(P[:, 4], .25), q75_5=np.quantile(P[:, 4], .75)))
    pd.DataFrame(rows).to_csv("data_eq/kronos_signals.csv", index=False)
    print(len(rows), "/", len(jobs), round(time.time() - t0), "s", flush=True)
print("done")
