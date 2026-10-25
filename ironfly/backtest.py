"""Monthly NIFTY Iron Fly with breakeven adjustments (Dhan 'Working Professionals' video) - EOD backtest
on real NSE bhavcopy option closes.

Rules as implemented (gaps in the video summary closed here; change via CLI flags):
- Entry: first trading day after the monthly expiry, sell the NEXT monthly expiry. Priced at that day's
  close (video enters 11:00-12:00; EOD data has no intraday prices).
- Structure: sell ATM CE + ATM PE (strike nearest spot), buy CE at K+W and PE at K-W.
- Wing W ('3% loss rule'): widest W (step 50) whose max loss (W - net credit) x lot <= 3% of capital.
  High VIX -> fatter credit -> wider wings, which is the video's 'adjust strikes for VIX'.
- Adjustment: when spot CLOSES beyond a breakeven of the current position, the threatened short leg is
  moved to the strike of its long hedge, and a new hedge is bought `hedge` points further out
  (call side: short K -> short K+W, long K+W -> long K+W+hedge). Repeats on each new breach.
- Stop loss: exit everything when MTM P&L (net of costs) <= -4% of capital at a daily close.
- Exit: hold to expiry, settle at intrinsic on the expiry-day spot close (or --exit-days-before N).
- Pricing: EOD close (settlement price if untraded); ITM strikes re-priced from the OTM twin via
  put-call parity on the same-expiry future (deep-ITM closes are stale).
- Costs: Rs20/order, STT on sell premium 0.10% (0.15% from 2026-04-01), STT on ITM long exercise
  0.125% (0.15% from 2026-04-01), NSE txn 0.03503%, SEBI Rs10/cr, stamp 0.003% buy, GST 18%;
  slippage `slip` points per option bought/sold.

Usage: python -m ironfly.backtest [--capital 200000] [--from 2026-01-01] [--out data_if/run]
"""
import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("data_if")


def load():
    df = pd.read_csv(D / "nifty_fo.csv.gz", parse_dates=["TradDt", "XpryDt"])
    fut = df[df.FinInstrmTp == "IDF"]
    op = df[df.FinInstrmTp == "IDO"].copy()
    # untraded strikes: exchange settlement (theoretical) price is fresher than a stale last close
    op["px"] = np.where(op.TtlTradgVol > 0, op.ClsPric, op.SttlmPric)
    spot = df.groupby("TradDt").UndrlygPric.first()
    # futures exist only for monthly expiries; keep the last per month (Sep-2025 had Thu 25 + Tue 30)
    monthly = pd.Series(sorted(fut.XpryDt.unique()))
    monthly = monthly.groupby(monthly.dt.to_period("M")).max().tolist()
    # Deep-ITM closes are thin and often stale (a 350-wide fly showed a 450-pt value). Price every
    # strike that is ITM vs the same-expiry future from its liquid OTM twin by put-call parity:
    # C - P = (F - K) e^{-rT}.
    F = fut.set_index(["TradDt", "XpryDt"]).ClsPric
    op["F"] = [F.get(k, s) for k, s in zip(zip(op.TradDt, op.XpryDt), op.TradDt.map(spot))]
    op["dfac"] = np.exp(-0.065 * (op.XpryDt - op.TradDt).dt.days / 365)
    twin = op.set_index(["TradDt", "XpryDt", "StrkPric", "OptnTp"]).px
    other = op.OptnTp.map({"CE": "PE", "PE": "CE"})
    tw = twin.reindex(pd.MultiIndex.from_arrays([op.TradDt, op.XpryDt, op.StrkPric, other])).to_numpy()
    sign = np.where(op.OptnTp == "CE", 1, -1)
    itm = sign * (op.F - op.StrkPric) > 0
    parity = tw + sign * (op.F - op.StrkPric) * op.dfac
    op["px"] = np.where(itm & ~np.isnan(tw), parity, op.px)
    chain = {k: g.set_index(["StrkPric", "OptnTp"]).px for k, g in op.groupby(["TradDt", "XpryDt"])}
    lots = op.groupby(["TradDt", "XpryDt"]).NewBrdLotQty.max()   # lot as of trade date
    return spot, [pd.Timestamp(m) for m in monthly], chain, lots


def costs(px, qty, day, buy, slip):
    """Rs cost of trading |qty| units at premium px (slippage + statutory + brokerage)."""
    q = abs(qty)
    turn = px * q
    stt = 0.0 if buy else turn * (0.0015 if day >= pd.Timestamp("2026-04-01") else 0.001)
    txn, sebi = turn * 0.0003503, turn * 1e-6
    stamp = turn * 0.00003 if buy else 0.0
    brk = 20.0
    return slip * q + stt + txn + sebi + stamp + brk + 0.18 * (brk + txn + sebi)


class Pos:
    def __init__(s):
        s.legs = {}            # (strike, 'CE'/'PE') -> signed lots
        s.cash = 0.0           # premium received - paid (points * units)
        s.fees = 0.0

    def trade(s, k, t, n, px, day, units, slip):
        s.legs[(k, t)] = s.legs.get((k, t), 0) + n
        if s.legs[(k, t)] == 0:
            del s.legs[(k, t)]
        s.cash -= n * px * units
        s.fees += costs(px, n * units, day, n > 0, slip)

    def value(s, quote, units):
        return sum(n * quote(k, t) * units for (k, t), n in s.legs.items())

    def payoff(s, S, units):
        v = np.zeros_like(S, dtype=float)
        for (k, t), n in s.legs.items():
            v += n * units * (np.maximum(S - k, 0) if t == "CE" else np.maximum(k - S, 0))
        return v + s.cash

    def breached(s, spot, units):
        """Side ('CE'/'PE') whose breakeven the spot is beyond, judged on the expiry payoff; None if inside.
        If no strike is profitable any more (loss locked in), fall back to 'short leg on that side is ITM'."""
        S = np.arange(spot - 4000, spot + 4000, 1.0)
        p = s.payoff(S, units)
        ok = S[p >= 0]
        if len(ok):
            return "CE" if spot > ok.max() else "PE" if spot < ok.min() else None
        if any(t == "CE" and n < 0 and k < spot for (k, t), n in s.legs.items()):
            return "CE"
        if any(t == "PE" and n < 0 and k > spot for (k, t), n in s.legs.items()):
            return "PE"
        return None


def run_cycle(entry, expiry, days, spot, chain, units, a):
    cap = a.capital * a.lots
    ch0 = chain.get((entry, expiry))
    if ch0 is None:
        return None
    S0 = spot[entry]
    strikes = np.array(sorted({k for k, _ in ch0.index}))
    K = strikes[np.argmin(np.abs(strikes - S0))]
    q = lambda ch, k, t: ch.get((k, t), np.nan)
    # missing strike (thin far wing): fall back to intrinsic so MTM/close-out stay defined
    qq = lambda ch, k, t, S: v if not np.isnan(v := q(ch, k, t)) else max(S - k, 0) if t == "CE" else max(k - S, 0)

    # 3% rule: widest wing whose max loss fits the budget
    W, credit = None, None
    for w in range(100, 2050, 50):
        c = q(ch0, K, "CE") + q(ch0, K, "PE") - q(ch0, K + w, "CE") - q(ch0, K - w, "PE")
        if np.isnan(c):
            break
        if (w - c) * units * a.lots <= a.maxloss * cap:
            W, credit = w, c
    if W is None:
        return None

    p, log = Pos(), []
    for k, t, n in [(K, "CE", -a.lots), (K, "PE", -a.lots), (K + W, "CE", a.lots), (K - W, "PE", a.lots)]:
        p.trade(k, t, n, q(ch0, k, t), entry, units, a.slip)
    hedge = a.hedge if a.hedge > 0 else W * (-a.hedge or 1)   # <=0: multiple of the wing
    n_adj, exit_reason, last = 0, "expiry", entry
    for d in days:
        ch = chain.get((d, expiry))
        S = spot[d]
        last = d
        if d == expiry:
            break
        quote = lambda k, t: qq(ch, k, t, S)
        mtm = p.cash + p.value(quote, units) - p.fees
        if mtm <= -a.sl * cap:
            exit_reason = "stop"
            break
        if a.exit_days_before and days.index(d) >= len(days) - 1 - a.exit_days_before:
            exit_reason = "time"
            break
        side = p.breached(S, units)
        if side and n_adj < a.max_adj:
            sgn = 1 if side == "CE" else -1
            shorts = [k for (k, t), n in p.legs.items() if t == side and n < 0]
            longs = [k for (k, t), n in p.legs.items() if t == side and n > 0]
            ks = min(shorts, key=lambda k: abs(k - S)) if shorts else None
            kl = min(longs, key=lambda k: sgn * k) if longs else None   # nearest hedge
            if ks is None or kl is None or np.isnan(q(ch, kl + sgn * hedge, side)):
                continue
            new_short, new_long = kl, kl + sgn * hedge
            p.trade(ks, side, a.lots, q(ch, ks, side), d, units, a.slip)                # buy back ITM short
            p.trade(kl, side, -a.lots, q(ch, kl, side), d, units, a.slip)               # close old hedge
            p.trade(new_short, side, -a.lots, q(ch, new_short, side), d, units, a.slip)  # short at old hedge
            p.trade(new_long, side, a.lots, q(ch, new_long, side), d, units, a.slip)     # new deep-OTM hedge
            n_adj += 1
            log.append(f"{d:%m-%d} {side} S={S:.0f} {ks}->{new_short} hedge {new_long}")

    if exit_reason == "expiry" and last != expiry:
        exit_reason = "open"                    # data ends before expiry: mark to market, don't settle
    # close out
    S = spot[last]
    ch = chain.get((last, expiry))
    stt_ex = 0.0015 if last >= pd.Timestamp("2026-04-01") else 0.00125
    for (k, t), n in list(p.legs.items()):
        if exit_reason == "expiry":
            iv = max(S - k, 0) if t == "CE" else max(k - S, 0)
            p.cash += n * iv * units
            if n > 0 and iv > 0:
                p.fees += iv * n * units * stt_ex
        else:
            p.trade(k, t, -n, qq(ch, k, t, S), last, units, a.slip)
        p.legs.pop((k, t), None)
    gross = p.cash
    return dict(entry=entry.date(), expiry=expiry.date(), exit=last.date(), reason=exit_reason,
                spot_in=round(S0), spot_out=round(S), move_pct=round(100 * (S / S0 - 1), 2), atm=K, wing=W,
                credit=round(credit, 1), lot=units, adjustments=n_adj, gross=round(gross), costs=round(p.fees),
                net=round(gross - p.fees), ret_pct=round(100 * (gross - p.fees) / cap, 2), adj_log="; ".join(log))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=200000, help="capital per lot (Rs)")
    ap.add_argument("--lots", type=int, default=1)
    ap.add_argument("--maxloss", type=float, default=0.03, help="wing budget as fraction of capital")
    ap.add_argument("--sl", type=float, default=0.04, help="hard stop as fraction of capital")
    ap.add_argument("--hedge", type=int, default=0, help="new hedge distance after adjustment in points; 0 = one wing, -2 = two wings")
    ap.add_argument("--max-adj", type=int, default=10)
    ap.add_argument("--slip", type=float, default=1.0, help="points slippage per option traded")
    ap.add_argument("--exit-days-before", type=int, default=0)
    ap.add_argument("--from", dest="start", default="2024-01-01")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    spot, monthly, chain, lots = load()
    tdays = list(spot.index)
    rows = []
    for prev, exp in zip(monthly, monthly[1:]):
        if exp < pd.Timestamp(a.start) or prev < tdays[0]:
            continue
        after = [d for d in tdays if d > prev]
        if not after:
            continue
        entry = after[0]
        days = [d for d in tdays if entry < d <= exp]
        if not days:
            continue
        r = run_cycle(entry, exp, days, spot, chain, int(lots[(entry, exp)]), a)
        if r:
            rows.append(r)
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 70)
    print(t.drop(columns="adj_log").to_string(index=False))
    done = t[t.reason != "open"]
    print(f"\ncycles {len(done)}  wins {(done.net > 0).sum()}  net Rs {done.net.sum():,.0f}  "
          f"on Rs {a.capital * a.lots:,.0f} -> {done.ret_pct.sum():.1f}%  costs Rs {done.costs.sum():,.0f}  "
          f"worst {done.ret_pct.min():.2f}%  stops {(done.reason == 'stop').sum()}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        t.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
