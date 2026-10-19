"""10/21 EMA + Supertrend(10,3) swing strategy: indicators, signals, portfolio simulator.

Daily bars, long only. Close-based signals fill at the NEXT open; stop and 4R target fill intraday
at the level (or at the open when it gaps through). Stop is checked before target on the same bar.
Full position = Q shares (multiple of 4) sized by risk; every exit/add moves one tranche = Q/4.
"""
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

CR = 1e7


@dataclass
class Params:
    capital: float = 1_000_000
    risk_pct: float = 1.0            # % of equity risked per position (entry and after any add)
    max_pos_pct: float = 25.0        # notional cap per position, % of equity
    n_setup: int = 10                # bars after bullish crossover in which an entry is valid
    retest_tol: float = 1.0          # % above the EMA that still counts as a retest
    consol_max_atr: float = 1.5      # 3-5 bar range <= this x ATR14
    pivot_lb: int = 3                # pivot needs this many bars each side
    stop_buf_atr: float = 0.25
    target_r: float = 4.0
    be_after_t1: bool = True         # ASSUMPTION: stop -> entry after T1
    st_filter: bool = True           # Supertrend red blocks add-backs
    dema_filter: bool = True         # entries only above 50 DEMA
    min_turnover_cr: float = 30.0    # 30-day avg of close x volume
    turnover_days: int = 30
    min_mcap_cr: float = 1500.0      # close x current shares outstanding
    min_price: float = 20.0
    warmup: int = 60
    slippage_pct: float = 0.05       # market orders only (next-open fills, stops)
    charges_on: bool = True
    brokerage: float = 0.0           # per order (discount-broker delivery = 0)
    dp_charge: float = 15.34         # per sell order
    scale_rules: bool = True         # False = control: no T1/T2/T4/add-backs, T3 exits everything
    ignore_cash: bool = False        # signal-level mode: take every signal, size off starting capital
    start: Optional[date] = None     # first date signals are evaluated
    end: Optional[date] = None


def charges(value: float, side: str, p: Params) -> float:
    """NSE equity delivery: STT 0.1% both sides, stamp 0.015% buy, exchange 0.00297%, SEBI Rs10/cr,
    GST 18% on (brokerage + exchange + SEBI), DP charge on sell."""
    if not p.charges_on or value <= 0:
        return 0.0
    exch, sebi = value * 0.0000297, value * 1e-6
    c = value * 0.001 + exch + sebi + p.brokerage + 0.18 * (p.brokerage + exch + sebi)
    return c + (value * 0.00015 if side == "BUY" else p.dp_charge)


# ---------------------------------------------------------------- indicators
def ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False).mean()


def rma(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1 / n, adjust=False).mean()


def atr(h, l, c, n):
    pc = c.shift()
    return rma(pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1), n)


def supertrend(h, l, c, n=10, m=3.0) -> np.ndarray:
    """+1 green / -1 red, TradingView ta.supertrend logic (RMA ATR)."""
    a = atr(h, l, c, n).values
    hl2 = ((h + l) / 2).values
    up, dn, cl = hl2 - m * a, hl2 + m * a, c.values
    fu, fd, d = up.copy(), dn.copy(), np.ones(len(cl))
    for i in range(1, len(cl)):
        fu[i] = max(up[i], fu[i - 1]) if cl[i - 1] > fu[i - 1] else up[i]
        fd[i] = min(dn[i], fd[i - 1]) if cl[i - 1] < fd[i - 1] else dn[i]
        if d[i - 1] < 0 and cl[i] > fd[i - 1]:
            d[i] = 1
        elif d[i - 1] > 0 and cl[i] < fu[i - 1]:
            d[i] = -1
        else:
            d[i] = d[i - 1]
    return d


def pivot_track(is_piv: np.ndarray, val: np.ndarray, k: int):
    """Per bar t: last / previous confirmed pivot value, pivot bar of last, and 'confirmed at t'.
    A pivot at bar p is confirmed (usable) at bar p + k."""
    n = len(val)
    last, prev, pbar, new = np.full(n, np.nan), np.full(n, np.nan), np.full(n, -1), np.zeros(n, bool)
    L, P, B = np.nan, np.nan, -1
    for t in range(n):
        q = t - k
        if q >= 0 and is_piv[q]:
            P, L, B, new[t] = L, val[q], q, True
        last[t], prev[t], pbar[t] = L, P, B
    return last, prev, pbar, new


def last_true_index(mask: np.ndarray) -> np.ndarray:
    idx = np.where(mask, np.arange(len(mask)), -1)
    return np.maximum.accumulate(idx)


def prepare(df: pd.DataFrame, shares: Optional[float], p: Params) -> dict:
    """df: one symbol, sorted by date, columns open/high/low/close/volume. Returns numpy arrays."""
    o, h, l, c, v = (df[k].astype(float) for k in ("open", "high", "low", "close", "volume"))
    n, k = len(df), p.pivot_lb
    e10, e21, e50 = ema(c, 10), ema(c, 21), ema(c, 50)
    dema = 2 * e50 - ema(e50, 50)
    a14 = atr(h, l, c, 14)
    above = (e10 > e21).values
    xup = above & ~np.r_[True, above[:-1]]
    xdn = ~above & np.r_[False, above[:-1]]
    w = 2 * k + 1
    is_ph = (h.rolling(w, center=True).max() == h).values
    is_pl = (l.rolling(w, center=True).min() == l).values
    ph, ph_prev, ph_bar, ph_new = pivot_track(is_ph, h.values, k)
    pl, pl_prev, pl_bar, pl_new = pivot_track(is_pl, l.values, k)
    lxu, lxd = last_true_index(xup), last_true_index(xdn)
    t = np.arange(n)

    # ---- entry trigger (section 3), evaluated at close of bar t
    tol = 1 + p.retest_tol / 100
    retest = ((l <= e10 * tol) & (c > e10)) | ((l <= e21 * tol) & (c > e21))
    rt = retest.values
    rt_ok = (rt & (t >= lxu)) | (np.r_[False, rt[:-1]] & (t - 1 >= lxu))
    confirm = ((c > h.shift()) & ((c - l) >= 0.5 * (h - l))).values
    trig_a = rt_ok & confirm
    trig_b = np.zeros(n, bool)
    for m in (3, 4, 5):
        hh, ll = h.shift().rolling(m).max(), l.shift().rolling(m).min()
        trig_b |= (((hh - ll) <= p.consol_max_atr * a14.shift()) & (c > hh)).values
    setup = above & (lxu >= 0) & (t - lxu <= p.n_setup)
    regime = (c > dema).values if p.dema_filter else np.ones(n, bool)

    turnover = (c * v).rolling(p.turnover_days).mean()
    mcap = c * shares if shares else pd.Series(np.nan, index=c.index)
    liquid = ((turnover >= p.min_turnover_cr * CR) & (c > p.min_price) & (mcap >= p.min_mcap_cr * CR)).values
    entry = setup & regime & (trig_a | trig_b) & liquid & (t >= p.warmup)

    return dict(date=df["date"].dt.date.values, o=o.values, h=h.values, l=l.values, c=c.values,
                e10=e10.values, e21=e21.values, a14=a14.values, st=supertrend(h, l, c),
                above=above, lxu=lxu, lxd=lxd, ph=ph, ph_prev=ph_prev, ph_bar=ph_bar, ph_new=ph_new,
                pl=pl, pl_prev=pl_prev, pl_bar=pl_bar, pl_new=pl_new, piv_low=np.where(is_pl)[0],
                entry=entry, trig=np.where(trig_a, "RETEST", "BREAKOUT"), turnover=turnover.values,
                mcap=mcap.values)


def swing_stop(s: dict, i: int, k: int, buf: float) -> Optional[float]:
    """Most recent pivot low confirmed by bar i that sits below close[i], minus buffer x ATR."""
    for q in s["piv_low"][::-1]:
        if q + k > i:
            continue
        stop = s["l"][q] - buf * s["a14"][i]
        if s["l"][q] < s["c"][i] and stop > 0:
            return stop
        if i - q > 250:
            break
    return None


# ---------------------------------------------------------------- portfolio simulation
@dataclass
class Pos:
    sym: str
    entry_date: date
    entry_px: float
    stop0: float
    R: float
    Q: int
    trigger: str
    xup_bar: int
    target: float
    shares: int = 0
    stop: float = 0.0
    t1_done: bool = False
    exited: set = field(default_factory=set)       # exit tranches fired & not yet restored
    t4_bar: int = -1
    pending: list = field(default_factory=list)    # ("SELL", tag) / ("ADD", tag, stop)
    bought: float = 0.0
    sold: float = 0.0
    fees: float = 0.0
    adds: int = 0
    max_shares: int = 0
    events: list = field(default_factory=list)

    @property
    def unit(self):
        return self.Q // 4


def simulate(stocks: dict, p: Params):
    """stocks: {sym: prepare() dict}. Returns (trades, events, equity) DataFrames."""
    idx = {s: {d: i for i, d in enumerate(a["date"])} for s, a in stocks.items()}
    cal = sorted({d for a in stocks.values() for d in a["date"]
                  if (p.start is None or d >= p.start) and (p.end is None or d <= p.end)})
    slip = p.slippage_pct / 100
    cash, equity = p.capital, p.capital
    pos: dict = {}
    used_xup: dict = {}
    pending_entries: list = []
    trades, events, curve, last_close = [], [], [], {}
    skipped = {"gap_below_stop": 0, "size_lt_4": 0, "no_cash": 0}

    def buy(ps, d, px, qty, tag):
        nonlocal cash
        val = px * qty
        f = charges(val, "BUY", p)
        cash -= val + f
        ps.bought += val; ps.fees += f; ps.shares += qty
        ps.max_shares = max(ps.max_shares, ps.shares)
        events.append((d, ps.sym, "BUY", tag, qty, round(px, 2), round(f, 2)))

    def sell(ps, d, px, qty, tag):
        nonlocal cash
        qty = min(qty, ps.shares)
        if qty <= 0:
            return
        val = px * qty
        f = charges(val, "SELL", p)
        cash += val - f
        ps.sold += val; ps.fees += f; ps.shares -= qty
        events.append((d, ps.sym, "SELL", tag, qty, round(px, 2), round(f, 2)))

    def close_cycle(ps, d, reason):
        pnl = ps.sold - ps.bought - ps.fees
        trades.append(dict(symbol=ps.sym, entry_date=ps.entry_date, exit_date=d, trigger=ps.trigger,
                           entry_px=round(ps.entry_px, 2), stop0=round(ps.stop0, 2), R=round(ps.R, 2),
                           Q=ps.Q, adds=ps.adds, max_shares=ps.max_shares, exit_reason=reason,
                           bought=round(ps.bought), sold=round(ps.sold), fees=round(ps.fees),
                           pnl=round(pnl), r_mult=round(pnl / (ps.Q * ps.R), 2),
                           hold_days=(d - ps.entry_date).days))
        del pos[ps.sym]

    for d in cal:
        # ---------- 1. open: pending sells, then adds, then new entries
        for sym, ps in list(pos.items()):
            i = idx[sym].get(d)
            if i is None:
                continue
            o = stocks[sym]["o"][i]
            for order in [x for x in ps.pending if x[0] == "SELL"]:
                sell(ps, d, o * (1 - slip), ps.unit, order[1])
                if order[1] == "T4":
                    ps.t4_bar = i
            ps.pending = [x for x in ps.pending if x[0] == "ADD"]
            if ps.shares == 0:
                ps.pending = []
                close_cycle(ps, d, "TRANCHES")
        for sym, ps in list(pos.items()):
            i = idx[sym].get(d)
            if i is None:
                continue
            o = stocks[sym]["o"][i]
            for _, tag, stop in [x for x in ps.pending if x[0] == "ADD"]:
                px = o * (1 + slip)
                need = px * ps.unit + charges(px * ps.unit, "BUY", p)
                if o <= stop or (need > cash and not p.ignore_cash):
                    ps.events.append((d, f"ADD_SKIP_{tag}"))
                    continue
                buy(ps, d, px, ps.unit, f"A_{tag}")
                ps.stop = max(ps.stop, stop)
                ps.exited.discard(tag)
                ps.adds += 1
            ps.pending = []
        for sym, xbar in pending_entries:
            s = stocks[sym]
            i = idx[sym].get(d)
            if i is None or sym in pos:
                continue
            stop = swing_stop(s, i - 1, p.pivot_lb, p.stop_buf_atr)
            o = s["o"][i]
            px = o * (1 + slip)
            if stop is None or o <= stop:
                skipped["gap_below_stop"] += 1
                continue
            R = px - stop
            base = p.capital if p.ignore_cash else equity
            Q = int(min(base * p.risk_pct / 100 / R, base * p.max_pos_pct / 100 / px) // 4) * 4
            if Q < 4:
                skipped["size_lt_4"] += 1
                continue
            if Q * px * 1.0012 > cash and not p.ignore_cash:          # full size or nothing; no leftover-cash slivers
                skipped["no_cash"] += 1
                continue
            ps = Pos(sym, d, px, stop, R, Q, s["trig"][i - 1], xbar, px + p.target_r * R, stop=stop)
            pos[sym] = ps
            used_xup[sym] = xbar
            buy(ps, d, px, Q, "ENTRY")
        pending_entries = []

        # ---------- 2. intraday: stop first, then 4R target
        for sym, ps in list(pos.items()):
            s, i = stocks[sym], idx[sym].get(d)
            if i is None:
                continue
            o, h, l = s["o"][i], s["h"][i], s["l"][i]
            reason = "TRANCHES"
            if o <= ps.stop:
                sell(ps, d, o * (1 - slip), ps.shares, "STOP_GAP")
                reason = "STOP"
            elif p.scale_rules and not ps.t1_done and (o >= ps.target or (l > ps.stop and h >= ps.target)):
                sell(ps, d, max(o, ps.target), ps.unit, "T1")
                ps.t1_done = True
                ps.exited.add("T1")
                if p.be_after_t1:
                    ps.stop = max(ps.stop, ps.entry_px)
            elif l <= ps.stop:
                sell(ps, d, ps.stop * (1 - slip), ps.shares, "STOP")
                reason = "STOP"
            if ps.shares == 0:
                close_cycle(ps, d, reason)

        # ---------- 3. close: exit / add signals for open positions
        for sym, ps in pos.items():
            s, i = stocks[sym], idx[sym].get(d)
            if i is None or i < 1:
                continue
            c, e10, e21 = s["c"], s["e10"], s["e21"]
            fired = []
            if "T2" not in ps.exited and c[i] < e10[i] and c[i - 1] < e10[i - 1]:
                fired.append("T2")
            if "T3" not in ps.exited and c[i] < e21[i] and c[i - 1] < e21[i - 1]:
                fired.append("T3")
            if ("T4" not in ps.exited and not s["above"][i] and s["ph_new"][i]
                    and s["ph_bar"][i] > s["lxd"][i] and s["ph"][i] < s["ph_prev"][i]):
                fired.append("T4")
            if not p.scale_rules:
                if "T3" in fired:
                    ps.pending = [("SELL", "T3")] * (ps.shares // ps.unit)
                continue
            fired = fired[:ps.shares // ps.unit]
            for tag in fired:
                ps.exited.add(tag)
                ps.pending.append(("SELL", tag))
            if p.st_filter and s["st"][i] < 0:
                continue
            want = []
            if "T2" in ps.exited and "T2" not in fired and c[i] > e10[i]:
                want.append("T2")
            if "T3" in ps.exited and "T3" not in fired and c[i] > e21[i]:
                want.append("T3")
            if ("T4" in ps.exited and "T4" not in fired and ps.t4_bar >= 0 and s["above"][i]
                    and s["lxu"][i] > ps.t4_bar and s["pl_new"][i] and s["pl_bar"][i] > s["lxu"][i]
                    and s["pl"][i] > s["pl_prev"][i]):
                want.append("T4")
            if ("T1" in ps.exited and not np.isnan(s["ph"][i]) and c[i] > s["ph"][i]
                    and c[i - 1] <= s["ph"][i]):
                want.append("T1")
            if not want:
                continue
            add_stop = swing_stop(s, i, p.pivot_lb, p.stop_buf_atr)
            if add_stop is None:
                continue
            new_stop = max(ps.stop, add_stop)
            sh = ps.shares - len(fired) * ps.unit
            for tag in want:
                if (sh + ps.unit) * max(c[i] - new_stop, 0) > (p.capital if p.ignore_cash else equity) * p.risk_pct / 100:
                    ps.events.append((d, f"ADD_RISK_{tag}"))
                    continue
                sh += ps.unit
                ps.pending.append(("ADD", tag, new_stop))

        # ---------- 4. close: new entry candidates (fill next open), most liquid first
        cands = []
        for sym, s in stocks.items():
            i = idx[sym].get(d)
            if i is None or sym in pos or not s["entry"][i] or used_xup.get(sym) == s["lxu"][i]:
                continue
            cands.append((s["turnover"][i], sym, s["lxu"][i]))
        pending_entries = [(sym, xb) for _, sym, xb in sorted(cands, reverse=True)]

        # ---------- 5. mark to market
        for sym, a in stocks.items():
            i = idx[sym].get(d)
            if i is not None:
                last_close[sym] = a["c"][i]
        mv = sum(ps.shares * last_close[ps.sym] for ps in pos.values())
        equity = cash + mv
        curve.append(dict(date=d, cash=round(cash), invested=round(mv), equity=round(equity),
                          positions=len(pos)))

    for sym, ps in list(pos.items()):
        ps.sold += ps.shares * last_close[sym]
        ps.fees += charges(ps.shares * last_close[sym], "SELL", p)
        ps.shares = 0
        close_cycle(ps, cal[-1], "OPEN_MTM")
    ev = pd.DataFrame(events, columns=["date", "symbol", "side", "tag", "qty", "px", "fees"])
    return pd.DataFrame(trades), ev, pd.DataFrame(curve), skipped
