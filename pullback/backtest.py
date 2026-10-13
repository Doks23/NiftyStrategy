"""Martin Luk-style momentum pullback backtest (daily bars, portfolio level, vectorised panel).

Setup (signal day t):
  - Leader: EMA9 > EMA21 > EMA50, close > EMA50, return over 63d >= min_mom (default 30%)
  - Volatile enough: ADR(20) >= min_adr; liquid: 20d avg turnover >= min_turnover_cr (bhavcopy only)
  - Flush-and-reclaim: day's low dips to/through a support (EMA9, EMA21 or AVWAP anchored at the
    60d high) and close finishes back above that support.
Entry:
  - "close": buy at signal-day close (proxy for the intraday reclaim entry)
  - "orh":   buy next day only if price trades above signal-day high, at that high
Stop: signal-day low. Skip if stop distance > min(max_stop, adr_frac * ADR).
Sizing: risk_pct of equity per trade, capped at max_pos of equity, max_open positions.
Exits (variant):
  - hard stop intraday (gap-through fills at open)
  - optional partial: sell part_frac at part_R (limit), then stop -> breakeven
  - trail: close below EMA(trail_ema) -> exit next open
Costs: cost_bps per side (covers brokerage/STT/slippage).

Data sources:
  --market bhav  NSE bhavcopy panel, every EQ stock that traded on each day (no survivorship bias)
  --market nse   yfinance, ~150 current large/mid caps (survivorship-biased)
  --market us    yfinance, hand-picked US momentum names (hindsight-biased)
"""
from __future__ import annotations
import argparse, os, sys
from dataclasses import dataclass, replace
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

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
    min_turnover_cr: float = 10.0  # 20d avg traded value, Rs crore (bhavcopy only)
    min_price: float = 20.0
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
    trail_ema: int = 9        # 9 | 21 | 50 ; 0 = no trail
    time_stop: int = 0        # exit at close after N days if < 1R (0 = off)
    cost_bps: float = 10


def load_yf(tickers, suffix, start, cache) -> dict:
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


def dict_to_panel(data: dict) -> dict:
    return {k: pd.DataFrame({s: df[k] for s, df in data.items()}).sort_index()
            for k in ["Open", "High", "Low", "Close", "Volume"]}


def avwap_from_high(h: np.ndarray, tp: np.ndarray, v: np.ndarray, win: int = 60) -> np.ndarray:
    """AVWAP anchored at the highest high of the trailing `win` bars, per column. Arrays are (T, N)."""
    T, N = h.shape
    out = np.full((T, N), np.nan)
    if T < win:
        return out
    hf = np.where(np.isnan(h), -np.inf, h)
    arg = sliding_window_view(hf, win, axis=0).argmax(axis=-1)  # (T-win+1, N)
    anchor = arg + np.arange(T - win + 1)[:, None]                # absolute row of anchor
    pv = np.nan_to_num(tp * v)
    vv = np.nan_to_num(v)
    cpv = np.vstack([np.zeros((1, N)), np.cumsum(pv, axis=0)])
    cv = np.vstack([np.zeros((1, N)), np.cumsum(vv, axis=0)])
    rows = np.arange(win - 1, T)
    cols = np.arange(N)[None, :]
    num = cpv[rows + 1][:, :] - cpv[anchor, cols]
    den = cv[rows + 1][:, :] - cv[anchor, cols]
    with np.errstate(invalid="ignore", divide="ignore"):
        out[win - 1:] = np.where(den > 0, num / den, np.nan)
    return out


def indicators(pn: dict) -> dict:
    O, H, L, C = pn["Open"], pn["High"], pn["Low"], pn["Close"]
    V = pn["Volume"].where(pn["Volume"] > 0).fillna(1.0)
    ind = dict(pn)
    for n in (9, 21, 50):
        ind[f"e{n}"] = C.ewm(span=n, adjust=False, ignore_na=True).mean().where(C.notna())
    ind["adr"] = ((H / L - 1) * 100).rolling(20, min_periods=15).mean()
    ind["mom"] = C / C.shift(63) - 1
    tp = ((H + L + C) / 3).values
    ind["avwap"] = pd.DataFrame(avwap_from_high(H.values, tp, V.values), index=C.index, columns=C.columns)
    if "Turnover" in pn:
        ind["to_cr"] = (pn["Turnover"] / 1e7).rolling(20, min_periods=15).mean()
    return ind


def signals(ind: dict, p: P) -> pd.DataFrame:
    C, L, O = ind["Close"], ind["Low"], ind["Open"]
    ok = (ind["e9"] > ind["e21"]) & (ind["e21"] > ind["e50"]) & (C > ind["e50"])
    ok &= (ind["mom"] >= p.min_mom) & (ind["adr"] >= p.min_adr) & (C >= p.min_price)
    if "to_cr" in ind:
        ok &= ind["to_cr"] >= p.min_turnover_cr
    touch = False
    for s in ("e9", "e21", "avwap"):
        touch = touch | ((L <= ind[s] * (1 + p.touch_tol)) & (C > ind[s]))
    stop = (C - L) / C
    ok &= touch & (stop >= p.min_stop) & (stop <= np.minimum(p.max_stop, p.adr_frac * ind["adr"] / 100))
    ok &= C > O * 0.995  # reclaim: closes near/above open
    return ok.fillna(False)


def run(ind: dict, p: P, start=None, end=None, capital=1_000_000.0):
    sig = signals(ind, p)
    idx = ind["Close"].index
    mask = np.ones(len(idx), bool)
    if start:
        mask &= idx >= pd.Timestamp(start)
    if end:
        mask &= idx < pd.Timestamp(end)
    rows = np.where(mask)[0]
    syms = list(ind["Close"].columns)
    O, H, L, C = (ind[k].values for k in ("Open", "High", "Low", "Close"))
    TR = ind[f"e{p.trail_ema}"].values if p.trail_ema else None
    S, MOM = sig.values, ind["mom"].values
    c = p.cost_bps / 1e4
    cash, pos, pending, trades, curve = capital, {}, [], [], []

    def close_out(j, px, d, why, frac=1.0):
        nonlocal cash
        t = pos[j]
        q = t["qty"] if frac >= 1 else t["qty"] * frac
        cash += q * px * (1 - c)
        t["realized"] += q * (px * (1 - c) - t["entry"] * (1 + c))
        t["qty"] -= q
        if frac >= 1 or t["qty"] <= 1e-9:
            trades.append(dict(sym=syms[j], entry_date=t["d0"], exit_date=d, entry=t["entry"], stop0=t["stop0"],
                               exit=px, why=why, R=t["realized"] / t["risk_cash"], pnl=t["realized"], days=t["days"]))
            del pos[j]

    for i in rows:
        d = idx[i]
        for j in list(pos):
            if np.isnan(C[i, j]):
                continue  # not traded today (suspension/holiday for this name)
            t = pos[j]
            t["days"] += 1
            t["last"] = C[i, j]
            if t.get("exit_next"):
                close_out(j, O[i, j], d, "trail"); continue
            if O[i, j] <= t["stop"]:
                close_out(j, O[i, j], d, "be" if t["be"] else "stop_gap"); continue
            if L[i, j] <= t["stop"]:
                close_out(j, t["stop"], d, "be" if t["be"] else "stop"); continue
            R1 = t["entry"] - t["stop0"]
            if p.part_R and not t["parted"] and H[i, j] >= t["entry"] + p.part_R * R1:
                close_out(j, max(O[i, j], t["entry"] + p.part_R * R1), d, "partial", p.part_frac)
                if j not in pos:
                    continue
                t["parted"] = t["be"] = True
                t["stop"] = max(t["stop"], t["entry"])
            if p.be_R and not t["be"] and C[i, j] >= t["entry"] + p.be_R * R1:
                t["be"] = True
                t["stop"] = max(t["stop"], t["entry"])
            if TR is not None and t["days"] > 1 and C[i, j] < TR[i, j]:
                t["exit_next"] = True
            if p.time_stop and t["days"] >= p.time_stop and C[i, j] < t["entry"] + R1:
                close_out(j, C[i, j], d, "time")
        eq = cash + sum(t["qty"] * t["last"] for t in pos.values())
        entries = []
        for j, hi, lo in pending:  # ORH orders from yesterday
            if j not in pos and not np.isnan(H[i, j]) and H[i, j] > hi:
                entries.append((j, max(hi, O[i, j]), lo, d))
        pending = []
        cand = np.where(S[i])[0]
        if p.entry == "close":
            cand = cand[np.argsort(-MOM[i, cand])]
            entries += [(j, C[i, j], L[i, j], d) for j in cand]
        else:
            pending = [(j, H[i, j], L[i, j]) for j in cand]
        for j, px, stop, d0 in entries:
            if len(pos) >= p.max_open:
                break
            if j in pos or px <= stop:
                continue
            qty = min(eq * p.risk_pct / (px - stop), eq * p.max_pos / px, cash / (px * (1 + c)))
            if qty <= 0:
                continue
            cash -= qty * px * (1 + c)
            pos[j] = dict(qty=qty, entry=px, stop=stop, stop0=stop, d0=d0, days=0, be=False, parted=False,
                          realized=0.0, risk_cash=qty * (px - stop), last=C[i, j])
        curve.append((d, cash + sum(t["qty"] * t["last"] for t in pos.values())))
    for j in list(pos):  # mark open positions to the last close so they appear in trade stats
        close_out(j, pos[j]["last"], idx[rows[-1]], "open_at_end")
    return pd.DataFrame(trades), pd.Series(dict(curve))


def stats(tr: pd.DataFrame, eq: pd.Series) -> dict:
    if tr.empty:
        return dict(trades=0)
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    dd = (eq / eq.cummax() - 1).min()
    w, l = tr[tr.R > 0], tr[tr.R <= 0]
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
    ap.add_argument("--market", choices=["bhav", "nse", "us"], default="bhav")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--split", default="2021-01-01", help="train < split <= test")
    ap.add_argument("--risk", type=float, default=0.005)
    ap.add_argument("--min-adr", type=float, default=None)
    ap.add_argument("--min-turnover", type=float, default=10.0, help="Rs crore, bhavcopy only")
    ap.add_argument("--panel", default="data/bhav/panel.pkl")
    ap.add_argument("--out", default="pullback/results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.market == "bhav":
        pn, adr, cost = pd.read_pickle(a.panel), 3.0, 15
    elif a.market == "nse":
        pn, adr, cost = dict_to_panel(load_yf(NSE, ".NS", a.start, f"{a.out}/nse_data.pkl")), 3.0, 15
    else:
        pn, adr, cost = dict_to_panel(load_yf(US, "", a.start, f"{a.out}/us_data.pkl")), 5.0, 5
    ind = indicators(pn)
    first = str(ind["Close"].index[0] + pd.Timedelta(days=365))[:10]
    rows = []
    for name, p in VARIANTS.items():
        p = replace(p, min_adr=a.min_adr or adr, cost_bps=cost, risk_pct=a.risk, min_turnover_cr=a.min_turnover)
        for period, (s0, s1) in {"train": (first, a.split), "test": (a.split, None), "full": (first, None)}.items():
            tr, eq = run(ind, p, s0, s1)
            rows.append(dict(variant=name, period=period, **stats(tr, eq)))
            if period == "full":
                tr.to_csv(f"{a.out}/{a.market}_{name}_trades.csv", index=False)
                eq.to_csv(f"{a.out}/{a.market}_{name}_equity.csv")
        print(name, "done", file=sys.stderr)
    res = pd.DataFrame(rows)
    res.to_csv(f"{a.out}/{a.market}_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(res.to_string(index=False))


if __name__ == "__main__":
    main()
