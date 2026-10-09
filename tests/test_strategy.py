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


P = Params(slippage=0, lots=1, lot_size=1, capital=1000, loss_cap_pct=1.8, cooloff_min=3)
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


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
