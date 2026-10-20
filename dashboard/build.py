import json, pandas as pd
d = pd.read_csv("data_eq/daily_ck.csv", parse_dates=["date"]).sort_values(["symbol", "date"])
k = pd.read_csv("data_eq/kronos_signals.csv", parse_dates=["date"]); k["date"] = k.date.dt.strftime("%Y-%m-%d")
kk = {(r.symbol, r.date): (round(r.up5, 2), round(r.ret5_mean * 100, 1)) for r in k.itertuples()}
last = d.date.max().strftime("%Y-%m-%d")

def pack(df, kind):
    out = []
    for r in df.itertuples():
        up = kk.get((r.symbol, r.setup))
        out.append(dict(s=r.symbol, su=r.setup, en=r.entry_date, ep=round(r.entry, 2), sl=round(r.stop, 2), q=int(r.qty), ex=r.exit,
                        xd=r.exit_date if r.exit_date != "open" else last, pnl=int(r.pnl), r=round(r.r, 2), fee=int(r.fees),
                        ku=up[0] if up else None, kr=up[1] if up else None))
    return out

sw = pd.read_csv("data_eq/swing_T1_close.csv")
sw["exit_date"] = sw.exit_date.astype(str).str[:10]; sw["entry_date"] = sw.entry_date.astype(str).str[:10]
op = pd.read_csv("data_eq/trades_ck.csv")
op = op.rename(columns={"date": "entry_date", "r_mult": "r"}); op["exit_date"] = op.exit_when.astype(str).str[:10].replace({"open": "open"})
op["entry_date"] = op.entry_date.astype(str).str[:10]
trades = {"swing": pack(sw, "swing"), "orb": pack(op, "orb")}
EX = json.load(open("data_eq/kronos_examples.json"))
syms = sorted({t["s"] for v in trades.values() for t in v} | {e["s"] for e in EX})
bars = {}
for s in syms:
    x = d[d.symbol == s]
    bars[s] = [[a, round(o, 2), round(h, 2), round(l, 2), round(c, 2)] for a, o, h, l, c in
               zip(x.date.dt.strftime("%Y-%m-%d"), x.open, x.high, x.low, x.close)]
# Kronos tables
ev = pd.read_csv("data_eq/kronos_signals.csv", parse_dates=["date"]); dd = d.set_index(["symbol", "date"])
rows = []
for r in ev.itertuples():
    x = d[d.symbol == r.symbol].reset_index(drop=True); ix = x.index[x.date == r.date]
    if not len(ix) or ix[0] + 11 >= len(x): continue
    i = ix[0]; e = x.open.iloc[i + 1]
    rows.append((r.up5, r.ret5_mean, x.close.iloc[i + 6] / e - 1, x.close.iloc[i + 11] / e - 1))
a = pd.DataFrame(rows, columns=["up5", "pred", "r5", "r10"])
a["b"] = pd.cut(a.up5, [-.01, .2, .5, .8, 1.01], labels=["≤20%", "20–50%", "50–80%", ">80%"])
kb = [dict(b=str(b), n=int(len(g)), f5=round(100 * g.r5.mean(), 2), w5=round(100 * (g.r5 > 0).mean(), 1), f10=round(100 * g.r10.mean(), 2)) for b, g in a.groupby("b", observed=True)]
kron = dict(buckets=kb, n=len(a), corr=round(a.pred.corr(a.r5), 3), base5=round(100 * a.r5.mean(), 2))
tpl = open("dashboard/template.html").read()
html = tpl.replace("/*DATA*/", "const TRADES=%s;const BARS=%s;const KRON=%s;const EX=%s;const LAST='%s';" % (json.dumps(trades, separators=(",", ":")), json.dumps(bars, separators=(",", ":")), json.dumps(kron), json.dumps(EX, separators=(",", ":")), last))
open("dashboard/nse_dashboard.html", "w").write(html); print(len(html) / 1e6, "MB", len(syms), "symbols")
