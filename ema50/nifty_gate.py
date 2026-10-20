"""Market gate: only take 50 MA pop trades when NIFTY 50 closes above BOTH its EMA10 and EMA20.
Regime is read from the last COMPLETED Nifty session before the entry day (no look-ahead).
Applies to the RVOL opening-entry trades (trades_ucl.csv) and the swing T+1-close trades (swing_univ_close.csv)."""
import numpy as np, pandas as pd

n = pd.read_csv("data_eq/nifty_daily.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
n["e10"] = n.close.ewm(span=10, adjust=False).mean(); n["e20"] = n.close.ewm(span=20, adjust=False).mean()
n["above"] = (n.close > n.e10) & (n.close > n.e20)
n["above_prev"] = n.above.shift(1)                       # state known at the open of this session
reg = n.set_index("date").above_prev


def row(name, s, ycol="yr"):
    if len(s) < 5:
        return dict(rule=name, n=len(s))
    w, l = s[s.pnl > 0].pnl.sum(), -s[s.pnl <= 0].pnl.sum(); a, b = s[s[ycol] == 2025], s[s[ycol] == 2026]
    return dict(rule=name, n=len(s), win=round(100 * (s.pnl > 0).mean()), net=int(s.pnl.sum()), pf=round(w / l, 2) if l else 9.9,
                y2025=f"{len(a)}/{int(a.pnl.sum())}", y2026=f"{len(b)}/{int(b.pnl.sum())}")


def tag(df, datecol):
    df = df.copy(); df["d"] = pd.to_datetime(df[datecol]).dt.normalize(); df["yr"] = df.d.dt.year
    df["gate"] = df.d.map(reg).fillna(False).astype(bool); return df


pd.set_option("display.width", 250)
print("Nifty above EMA10 & EMA20 (state at the previous close): %.0f%% of sessions since Jan 2025" % (100 * n[n.date >= "2025-01-02"].above_prev.mean()))
rv = tag(pd.read_csv("data_eq/trades_ucl.csv"), "date")
sw = tag(pd.read_csv("data_eq/swing_univ_close.csv"), "entry_date")
out = []
for name, df in (("RVOL opening entry", rv), ("Swing, buy T+1 close", sw)):
    out += [row(f"{name}: ALL", df), row(f"{name}: Nifty ABOVE both EMAs", df[df.gate]), row(f"{name}: Nifty BELOW (skipped)", df[~df.gate])]
print(pd.DataFrame(out).to_string(index=False))
# robustness: gate on the setup day instead of the entry-eve, and a stricter/looser variant, to see if the result depends on the exact definition
rv["setup_d"] = pd.to_datetime(rv.setup).dt.normalize()
rv["gate_setup"] = rv.setup_d.map(n.set_index("date").above).fillna(False).astype(bool)
print()
print(pd.DataFrame([row("RVOL: gate read on SETUP day close", rv[rv.gate_setup]), row("RVOL: gate = close > EMA20 only", rv[rv.d.map(n.set_index("date").close.shift(1) > n.set_index("date").e20.shift(1)).fillna(False).astype(bool)]),
                    row("RVOL: gate = close > EMA10 only", rv[rv.d.map(n.set_index("date").close.shift(1) > n.set_index("date").e10.shift(1)).fillna(False).astype(bool)])]).to_string(index=False))
g = rv[rv.gate]
print("\nGated RVOL trades by quarter:"); print(g.groupby(g.d.dt.to_period("Q")).pnl.agg(["count", "sum"]).T.to_string())
print("\nTop 5 gated winners:"); print(g.nlargest(5, "pnl")[["symbol", "date", "exit", "pnl", "r_mult"]].to_string(index=False))
print("without top 3 winners:", int(g.nsmallest(len(g) - 3, "pnl", keep="all").pnl.sum()) if len(g) > 3 else "n/a")
