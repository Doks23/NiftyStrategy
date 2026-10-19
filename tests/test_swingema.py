"""Logic tests on hand-built daily series (NOT market data)."""
import sys
from dataclasses import replace
from datetime import date, timedelta
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from swingema.strategy import Params, charges, pivot_track, prepare, simulate, supertrend

P = Params(capital=1_000_000, min_turnover_cr=0, min_mcap_cr=0, min_price=0, warmup=30,
           charges_on=False, slippage_pct=0, dema_filter=False)


def frame(closes, spread=1.0, opens=None, lows=None, highs=None):
    c = np.asarray(closes, float)
    d0 = date(2025, 1, 1)
    df = pd.DataFrame({"date": pd.to_datetime([d0 + timedelta(days=i) for i in range(len(c))]),
                       "open": opens if opens is not None else np.r_[c[0], c[:-1]],
                       "high": highs if highs is not None else c + spread,
                       "low": lows if lows is not None else c - spread,
                       "close": c, "volume": 1e6})
    return df


def test_pivot_confirmed_only_after_k_bars():
    is_piv = np.array([0, 0, 1, 0, 0, 0, 0], bool)
    val = np.arange(7.0)
    last, prev, pbar, new = pivot_track(is_piv, val, 3)
    assert np.isnan(last[4]) and last[5] == 2 and pbar[5] == 2 and new[5] and not new[6]


def test_supertrend_flips_with_trend():
    c = pd.Series(np.r_[np.linspace(100, 150, 60), np.linspace(150, 90, 60)])
    st = supertrend(c + 1, c - 1, c)
    assert st[55] == 1 and st[-1] == -1


def test_charges_delivery():
    p = Params()
    assert abs(charges(100_000, "BUY", p) - (100 + 15 + 2.97 + 0.1 + 0.18 * 3.07)) < 0.01
    assert abs(charges(100_000, "SELL", p) - (100 + 2.97 + 0.1 + 0.18 * 3.07 + 15.34)) < 0.01


def pullback_series():
    # downtrend -> uptrend (bullish 10/21 cross) -> shallow pullback to 10 EMA -> bullish bar
    down = list(np.linspace(130, 100, 40))
    up = list(np.linspace(100, 120, 15))
    pb = [118.5, 117.8, 117.5]
    return down + up + pb + [121.0]


def test_entry_fills_next_open_with_stop_below_swing_low():
    c = pullback_series() + [122, 123, 124]
    s = prepare(frame(c), 1e9, P)
    sig = np.where(s["entry"])[0]
    assert len(sig) and sig[-1] <= len(pullback_series()) - 1
    tr, ev, eq, _ = simulate({"X": s}, P)
    e = ev[ev.tag == "ENTRY"].iloc[0]
    i = sig[0]
    assert e.date == s["date"][i + 1] and abs(e.px - s["o"][i + 1]) < 1e-9   # next-open fill
    assert e.qty % 4 == 0
    assert tr.iloc[0].stop0 < min(s["l"][i - 5:i + 1])                        # below swing low


def test_stop_exits_everything_intraday():
    base = pullback_series()
    n = len(base)
    c = base + [122, 90, 90]
    lows = np.array(c) - 1.0
    tr, ev, _, _ = simulate({"X": prepare(frame(c, lows=lows), 1e9, P)}, P)
    t = tr.iloc[0]
    assert t.exit_reason == "STOP" and abs(t.r_mult + 1) < 0.05


def test_t1_books_quarter_and_moves_stop_to_breakeven():
    base = pullback_series()
    c = base + [122] + [215] * 3 + [118] * 3          # R ~22.6 (swing low ~98) -> 4R ~211
    df = frame(c)
    tr, ev, _, _ = simulate({"X": prepare(df, 1e9, P)}, P)
    t1 = ev[ev.tag == "T1"]
    entry = ev[ev.tag == "ENTRY"].iloc[0]
    assert len(t1) == 1 and t1.iloc[0].qty == entry.qty // 4
    stops = ev[ev.tag.str.startswith("STOP")]
    assert len(stops) == 1 and abs(stops.iloc[0].px - entry.px) < 1e-9          # BE stop, not stop0


def test_two_closes_below_10ema_sells_one_tranche_then_close_above_adds_back():
    base = pullback_series()
    c = base + [123, 125, 127, 129, 131, 133, 135, 137, 139, 141, 143, 145, 147, 149, 151,
                141, 140, 150, 152, 154]                     # two closes under 10 EMA (~142), then back
    p = replace(P, risk_pct=50)                              # open-risk cap would otherwise block the add
    tr, ev, _, _ = simulate({"X": prepare(frame(c, spread=0.3), 1e9, p)}, p)
    tags = list(ev.tag)
    assert "T2" in tags and "A_T2" in tags and tags.index("T2") < tags.index("A_T2")
    assert ev[ev.tag == "T2"].iloc[0].qty == ev[ev.tag == "ENTRY"].iloc[0].qty // 4


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
