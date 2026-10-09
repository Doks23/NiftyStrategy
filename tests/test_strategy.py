"""Logic tests on tiny hand-built days (NOT market data)."""
import sys
from datetime import date, datetime, timedelta
sys.path.insert(0, ".")
import pandas as pd
from niftyvwap.strategy import Params, run_day, session_vwap

D = date(2025, 1, 6)
T0 = datetime(2025, 1, 6, 9, 15)


def spot(closes):
    idx = [T0 + timedelta(minutes=5 * i) for i in range(len(closes))]
    return pd.DataFrame({"high": [c + 1 for c in closes], "low": [c - 1 for c in closes],
                         "close": closes, "volume": [100] * len(closes)}, index=pd.DatetimeIndex(idx))


def opts(minutes, ce, pe, strike=100):
    idx = pd.DatetimeIndex([T0 + timedelta(minutes=i) for i in range(minutes)])
    mk = lambda f: pd.DataFrame({"open": [f(i) for i in range(minutes)], "close": [f(i) for i in range(minutes)]}, index=idx)
    return {("CE", strike): mk(ce), ("PE", strike): mk(pe)}


P = Params(charges_on=False, roll_decay=0, skip_first_candle=False, slippage=0, lots=1, lot_size=1, capital=1000, loss_cap_pct=1.8, cooloff_min=3)
atm = lambda ts, typ: 100


def test_vwap_cumulative():
    v = session_vwap(spot([100, 102]), "volume")
    assert abs(v.iloc[0] - 100) < 1e-9 and abs(v.iloc[1] - 101) < 1e-9


def test_bull_sells_put_then_reversal_exits_and_flips():
    # first candle closes ABOVE its own vwap? with 1 bar vwap==tp==close -> flat. use rising then falling.
    s = spot([100, 105, 106, 90, 80])          # closes at 09:20.. ; 105>vwap bullish, 90<vwap bearish
    tr = run_day(D, s, opts(30, lambda i: 50, lambda i: 40 if i < 15 else 30), atm, P)
    assert tr[0].side == "SELL_PE"
    assert tr[0].reason == "REVERSAL"
    assert tr[0].pnl == tr[0].entry_px - tr[0].exit_px  # qty=1, no slippage
    assert tr[1].side == "SELL_CE"             # flips to bearish on next close after cool-off


def test_loss_cap_halts_day():
    # short PE entered ~40, option doubles -> loss > 18 cap => LOSS_CAP and no further trades
    s = spot([100, 105, 106, 107, 108, 109, 110, 111])
    tr = run_day(D, s, opts(40, lambda i: 50, lambda i: 40 if i < 12 else 70), atm, P)
    assert tr[-1].reason == "LOSS_CAP" and len(tr) == 1


def test_eod_squareoff():
    s = spot([100, 105] + [106] * 70)
    tr = run_day(D, s, opts(400, lambda i: 50, lambda i: 40), atm, P)
    assert tr[-1].reason == "EOD" and tr[-1].exit_ts.hour == 15 and tr[-1].exit_ts.minute == 15


def test_roll_down_and_first_candle_skip():
    s = spot([100, 105, 106, 107, 108, 109, 110, 111])
    ce = lambda i: 50
    pe = lambda i: 40 if i < 12 else 10      # premium decays 75% -> roll
    o = opts(40, ce, pe, 100); o.update(opts(40, ce, lambda i: 40 if i < 12 else 10, 150))
    q = Params(charges_on=False, roll_decay=0.6, skip_first_candle=True, slippage=0, lots=1, lot_size=1, capital=1000)
    tr = run_day(D, s, o, lambda ts, typ: 100 if ts < T0 + timedelta(minutes=12) else 150, q)
    assert tr[0].entry_ts == T0 + timedelta(minutes=10)      # 09:15 candle skipped (09:20 close ignored)
    assert tr[0].reason == "ROLL" and tr[1].strike == 150 and tr[1].side == "SELL_PE"


def test_charges_sanity():
    from niftyvwap.strategy import charges
    # 75 qty, sell 100 buy 60, after Apr-2026 STT 0.15%: sell turnover 7500
    f = charges(date(2026, 9, 1), 100, 60, 75)
    stt = 7500 * 0.0015; txn = (7500 + 4500) * 0.0003553; gst = 0.18 * (40 + txn + 12000 * 1e-6)
    assert abs(f - (40 + stt + txn + 12000 * 1e-6 + 4500 * 0.00003 + gst)) < 1e-6
    assert charges(date(2026, 1, 1), 100, 60, 75) < f          # old STT rate lower


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
