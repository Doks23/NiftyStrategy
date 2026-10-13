"""Martin Luk-style momentum pullback backtest (daily bars, portfolio level).

Setup (signal day t):
  - Leader: EMA9 > EMA21 > EMA50, close > EMA50, return over 63d >= min_mom (default 30%)
  - Volatile enough: ADR(20) >= min_adr
  - Flush-and-reclaim: day's low dips to/through a support (EMA9, EMA21 or AVWAP anchored at the
    60d high) and close finishes back above that support.
Entry:
  - "close": buy at signal-day close (proxy for the intraday reclaim entry)
  - "orh":   buy next day only if price trades above signal-day high (break of high), at that high
Stop: signal-day low. Skip if stop distance > min(max_stop, adr_frac * ADR).
Sizing: risk_pct of equity per trade, capped at max_pos of equity, max_open positions.
Exits (variant):
  - hard stop intraday (gap-through fills at open)
  - optional partial: sell part_frac at part_R (limit), then stop -> breakeven
  - trail: close below EMA(trail_ema) -> exit next open
Costs: cost_bps per side (covers brokerage/STT/slippage).
"""
from __future__ import annotations
import argparse, json, os, sys
from dataclasses import dataclass, asdict, replace
import numpy as np
import pandas as pd

NSE = """RELIANCE TCS HDFCBANK ICICIBANK INFY BHARTIARTL SBIN ITC LT HINDUNILVR BAJFINANCE KOTAKBANK AXISBANK
MARUTI SUNPHARMA TITAN M&M NTPC ULTRACEMCO ONGC TATASTEEL POWERGRID ADANIENT ADANIPORTS COALINDIA BAJAJFINSV
HCLTECH WIPRO TECHM ASIANPAINT NESTLEIND JSWSTEEL GRASIM HINDALCO DRREDDY CIPLA EICHERMOT HEROMOTOCO BPCL
BRITANNIA DIVISLAB APOLLOHOSP TATACONSUM INDUSINDBK SBILIFE HDFCLIFE BAJAJ-AUTO TRENT BEL HAL TVSMOTOR
DLF VEDL PIDILITIND SIEMENS ABB GODREJCP DABUR HAVELLS AMBUJACEM SHREECEM IOC GAIL TATAPOWER CHOLAFIN
BANKBARODA PNB CANBK IRCTC IRFC RECLTD PFC BHEL ZYDUSLIFE LUPIN AUROPHARMA TORNTPHARM MANKIND POLYCAB
DIXON PERSISTENT COFORGE LTIM MPHASIS KPITTECH TATAELXSI CUMMINSIND BOSCHLTD MOTHERSON ASHOKLEY BHARATFORG
INDHOTEL JUBLFOOD VOLTAS PAGEIND MRF APLAPOLLO ASTRAL SUPREMEIND CGPOWER SUZLON BSE CDSL ANGELONE MAZDOCK
COCHINSHIP BDL SOLARINDS KAYNES PRESTIGE GODREJPROP OBEROIRLTY LODHA PHOENIXLTD MUTHOOTFIN SHRIRAMFIN
INDIGO NAUKRI POLICYBZR PAYTM TATACOMM IDEA JINDALSTEL SAIL NMDC NATIONALUM HINDZINC LALPATHLAB MAXHEALTH""".split()

US = """NVDA AMD SMCI TSLA COIN MSTR PLTR SOUN IONQ RGTI QBTS HOOD SOFI AFRM UPST CVNA APP RDDT ARM AVGO META
NFLX SHOP CRWD NET SNOW DDOG MDB ZS PANW ANET VRT CELH ELF DKNG RKLB ASTS LUNR OKLO SMR CCJ VST CEG TLN
GME AMC LMND MARA RIOT CLSK HUT WULF IREN CIFR NBIS CRWV TEM HIMS RBLX U ROKU SQ PYPL ENPH FSLR RUN
MRNA BNTX ZM DOCU PTON NIO XPEV LI FUTU TTD ABNB UBER DASH SNAP PINS BILL OPEN AI PATH S DUOL ONON
CAVA TOST AXON CAR W CHWY BYND SPOT""".split()


@dataclass
class P:
    entry: str = "close"      # close | orh
    min_mom: float = 0.30
    min_adr: float = 3.0      # % ; Luk uses >5% on US names
    max_stop: float = 0.05
    adr_frac: float = 0.5
    min_stop: float = 0.004
    touch_tol: float = 0.003  # low within 0.3% above support counts as a touch
    risk_pct: float = 0.005
    max_pos: float = 0.25
    max_open: int = 8
    part_R: float = 0.0       # 0 = no partial
    part_frac: float = 0.33
    be_R: float = 0.0         # move stop to breakeven once close >= be_R (0 = off; partial also sets BE)
    trail_ema: int = 9        # 0 = no trail
    time_stop: int = 0        # exit at close after N days if < 1R (0 = off)
    cost_bps: float = 10


def load(tickers, suffix, start, cache):
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    import yfinance as yf
    syms = [t + suffix for t in tickers]
    d = yf.download(syms, start=start, auto_adjust=True, progress=False, group_by="ticker", threads=True)
    out = {}
    for s in syms:
        try:
            df = d[s][["Open", "High", "Low", "Close", "Volume"]].dropna()
        except KeyError:
            continue
        if len(df) > 300:
            out[s] = df
    pd.to_pickle(out, cache)
    return out


def indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for n in (9, 21, 50):
        df[f"e{n}"] = df.Close.ewm(span=n, adjust=False).mean()
    df["adr"] = ((df.High / df.Low - 1) * 100).rolling(20).mean()
    df["mom"] = df.Close / df.Close.shift(63) - 1
    # AVWAP anchored at the highest high of the last 60 bars (recomputed each day; no lookahead)
    tp = (df.High + df.Low + df.Close) / 3
    vol = df.Volume.replace(0, np.nan).fillna(1.0)
    pv, v, hi = (tp * vol).values, vol.values, df.High.values
    av = np.full(len(df), np.nan)
    cpv, cv = np.concatenate([[0], np.cumsum(pv)]), np.concatenate([[0], np.cumsum(v)])
    for i in range(60, len(df)):
        a = i - 59 + int(np.argmax(hi[i - 59:i + 1]))
        av[i] = (cpv[i + 1] - cpv[a]) / (cv[i + 1] - cv[a])
    df["avwap"] = av
    return df


def signals(df: pd.DataFrame, p: P) -> pd.Series:
    lead = (df.e9 > df.e21) & (df.e21 > df.e50) & (df.Close > df.e50)
    ok = lead & (df.mom >= p.min_mom) & (df.adr >= p.min_adr)
    touch = pd.Series(False, index=df.index)
    for s in ("e9", "e21", "avwap"):
        touch |= (df.Low <= df[s] * (1 + p.touch_tol)) & (df.Close > df[s])
    stop = (df.Close - df.Low) / df.Close
    ok &= touch & (stop >= p.min_stop) & (stop <= np.minimum(p.max_stop, p.adr_frac * df.adr / 100))
    ok &= df.Close > df.Open * 0.995  # reclaim: closes near/above open
    return ok.fillna(False)


def run(data: dict, p: P, start=None, end=None, capital=1_000_000.0):
    frames = {s: indicators(df) for s, df in data.items()} if "e9" not in next(iter(data.values())) else data
    sig = {s: signals(f, p) for s, f in frames.items()}
    dates = sorted(set().union(*[f.index for f in frames.values()]))
    dates = [d for d in dates if (start is None or d >= pd.Timestamp(start)) and (end is None or d <= pd.Timestamp(end))]
    c = p.cost_bps / 1e4
    cash, pos, pending, trades, curve = capital, {}, [], [], []

    def close_out(s, px, d, why, frac=1.0):
        nonlocal cash
        t = pos[s]
        q = t["qty"] * frac if frac < 1 else t["qty"]
        cash += q * px * (1 - c)
        t["realized"] += q * (px * (1 - c) - t["entry"] * (1 + c))
        t["qty"] -= q
        if frac >= 1 or t["qty"] <= 1e-9:
            trades.append(dict(sym=s, entry_date=t["d0"], exit_date=d, entry=t["entry"], stop0=t["stop0"],
                               exit=px, why=why, R=t["realized"] / t["risk_cash"], pnl=t["realized"],
                               days=t["days"]))
            del pos[s]

    for d in dates:
        # 1) process open positions
        for s in list(pos):
            f = frames[s]
            if d not in f.index:
                continue
            b = f.loc[d]
            t = pos[s]
            t["days"] += 1
            if t.get("exit_next"):
                close_out(s, b.Open, d, "trail"); continue
            if b.Open <= t["stop"]:
                close_out(s, b.Open, d, "stop_gap" if not t["be"] else "be"); continue
            if b.Low <= t["stop"]:
                close_out(s, t["stop"], d, "stop" if not t["be"] else "be"); continue
            R1 = t["entry"] - t["stop0"]
            if p.part_R and not t["parted"] and b.High >= t["entry"] + p.part_R * R1:
                tgt = max(b.Open, t["entry"] + p.part_R * R1)
                close_out(s, tgt, d, "partial", p.part_frac)
                t["parted"] = True; t["be"] = True; t["stop"] = max(t["stop"], t["entry"])
            if p.be_R and b.Close >= t["entry"] + p.be_R * R1 and not t["be"]:
                t["be"] = True; t["stop"] = max(t["stop"], t["entry"])
            if p.trail_ema and b.Close < f.loc[d, f"e{p.trail_ema}"] and t["days"] > 1:
                t["exit_next"] = True
            if p.time_stop and t["days"] >= p.time_stop and b.Close < t["entry"] + R1:
                close_out(s, b.Close, d, "time")
        # 2) ORH pending orders (from yesterday's signals)
        eq = cash + sum(t["qty"] * frames[s].loc[:d].Close.iloc[-1] for s, t in pos.items())
        todays = []
        for s, hi, lo in pending:
            f = frames[s]
            if d in f.index and s not in pos and f.loc[d, "High"] > hi:
                px = max(hi, f.loc[d, "Open"])
                todays.append((s, px, lo, d))
        pending = []
        # 3) new close-entry signals / queue ORH
        cands = []
        for s, sg in sig.items():
            if d in sg.index and sg.loc[d] and s not in pos:
                f = frames[s]
                if p.entry == "close":
                    cands.append((s, f.loc[d, "Close"], f.loc[d, "Low"], d, f.loc[d, "mom"]))
                else:
                    pending.append((s, f.loc[d, "High"], f.loc[d, "Low"]))
        cands.sort(key=lambda x: -x[4])
        entries = todays + [x[:4] for x in cands]
        for s, px, stop, d0 in entries:
            if len(pos) >= p.max_open or s in pos:
                continue
            risk_cash = eq * p.risk_pct
            qty = min(risk_cash / (px - stop), eq * p.max_pos / px, cash / (px * (1 + c)))
            if qty <= 0 or px <= stop:
                continue
            cash -= qty * px * (1 + c)
            pos[s] = dict(qty=qty, entry=px, stop=stop, stop0=stop, d0=d0, days=0, be=False, parted=False,
                          realized=0.0, risk_cash=qty * (px - stop))
        eq = cash + sum(t["qty"] * frames[s].loc[:d].Close.iloc[-1] for s, t in pos.items())
        curve.append((d, eq))
    eqc = pd.Series(dict(curve))
    return pd.DataFrame(trades), eqc


def stats(tr: pd.DataFrame, eq: pd.Series) -> dict:
    if tr.empty:
        return dict(trades=0)
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    dd = (eq / eq.cummax() - 1).min()
    w = tr[tr.R > 0]
    l = tr[tr.R <= 0]
    return dict(trades=len(tr), win_rate=round(len(w) / len(tr) * 100, 1),
                avg_win_R=round(w.R.mean(), 2) if len(w) else 0, avg_loss_R=round(l.R.mean(), 2) if len(l) else 0,
                expectancy_R=round(tr.R.mean(), 3), best_R=round(tr.R.max(), 1),
                total_ret_pct=round((eq.iloc[-1] / eq.iloc[0] - 1) * 100, 1),
                cagr_pct=round(((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1) * 100, 1) if yrs > 0 else None,
                max_dd_pct=round(dd * 100, 1), avg_days=round(tr.days.mean(), 1))


VARIANTS = {
    "A_trail9_only": P(),
    "B_trail21_only": P(trail_ema=21),
    "C_part3R_trail9": P(part_R=3, part_frac=0.33),
    "D_part3R_trail21": P(part_R=3, part_frac=0.33, trail_ema=21),
    "E_part5R_BE2R_trail21": P(part_R=5, part_frac=0.33, be_R=2, trail_ema=21),
    "F_part2R_half_trail21": P(part_R=2, part_frac=0.5, trail_ema=21),
    "G_ORH_part3R_trail21": P(entry="orh", part_R=3, part_frac=0.33, trail_ema=21),
    "H_fixed3R_noTrail": P(part_R=3, part_frac=1.0, trail_ema=0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["nse", "us"], default="nse")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--split", default="2021-01-01", help="train < split <= test")
    ap.add_argument("--risk", type=float, default=0.005)
    ap.add_argument("--out", default="pullback/results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tick, suf, adr, cost = (NSE, ".NS", 3.0, 15) if a.market == "nse" else (US, "", 5.0, 5)
    raw = load(tick, suf, a.start, f"{a.out}/{a.market}_data.pkl")
    frames = {s: indicators(df) for s, df in raw.items()}
    rows = []
    for name, p in VARIANTS.items():
        p = replace(p, min_adr=adr, cost_bps=cost, risk_pct=a.risk)
        for period, (s0, s1) in {"train": ("2016-01-01", a.split), "test": (a.split, None), "full": ("2016-01-01", None)}.items():
            tr, eq = run(frames, p, s0, s1)
            st = stats(tr, eq)
            rows.append(dict(variant=name, period=period, **st))
            if period == "full":
                tr.to_csv(f"{a.out}/{a.market}_{name}_trades.csv", index=False)
        print(name, "done", file=sys.stderr)
    res = pd.DataFrame(rows)
    res.to_csv(f"{a.out}/{a.market}_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(res.to_string(index=False))


if __name__ == "__main__":
    main()
