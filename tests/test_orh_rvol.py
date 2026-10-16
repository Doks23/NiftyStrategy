"""Run: python -I tests/test_orh_rvol.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from pullback.orh_rvol import Cfg, orh_entry, follow_daily, hod_trigger


def minutes(highs, lows, vols, opens=None, day="2026-06-01"):
    ts = pd.date_range(f"{day} 09:15", periods=len(highs), freq="1min")
    opens = opens or [l for l in lows]
    return pd.DataFrame(dict(ts=ts, open=opens, high=highs, low=lows, close=highs, volume=vols))


def test_rvol_gate():
    m = minutes([101] * 11 + [102] * 5, [97] * 11 + [100] * 5, [10] * 16)  # OR vol = 110; day low 97 < 2.5% cap
    assert orh_entry(m, vavg=10_000, contr_low=95, k=Cfg()) is None       # 1.1% < 8%
    e = orh_entry(m, vavg=1_000, contr_low=95, k=Cfg())                    # 11% >= 8%
    assert e and e["entry"] == 101 and round(e["stop"], 3) == round(101 * 0.975, 3) and not e["same_day_stop"]


def test_stop_uses_tightest_and_same_day_stop():
    m = minutes([101] * 11 + [102, 102, 102], [100] * 11 + [100.5, 99, 98], [100] * 14)
    e = orh_entry(m, vavg=1_000, contr_low=95, k=Cfg())
    assert e["stop"] == 100.0 and e["same_day_stop"]                        # day low 100 > 101*0.975


def test_no_break_no_trade():
    m = minutes([101] * 11 + [100.5] * 20, [99] * 31, [100] * 31)
    assert orh_entry(m, vavg=1_000, contr_low=95, k=Cfg()) is None


def test_follow_daily_be_part_trail():
    d = pd.bdate_range("2026-06-01", periods=6)
    df = pd.DataFrame(dict(Open=[100, 101, 104, 109, 110, 105], High=[101, 105, 110, 111, 111, 106],
                           Low=[99, 100.5, 103, 108, 104, 104], Close=[100.5, 104, 109, 110, 105, 105]), index=d)
    e21 = pd.Series([99, 100, 102, 104, 106, 106.0], index=d)
    pnl, why, xd = follow_daily(df, 0, 100.0, 98.0, e21, Cfg())            # R=2: BE@104, 25% @108
    assert why == "21EMA" and xd == d[5] and abs(pnl - (0.25 * 8 + 0.75 * 5)) < 1e-9, (pnl, why)


def bars(rows, day="2026-06-01"):
    ts = pd.date_range(f"{day} 09:15", periods=len(rows), freq="1min")
    o, h, l, c, v = zip(*rows)
    return pd.DataFrame(dict(ts=ts, open=o, high=h, low=l, close=c, volume=v))


def test_hod_trigger_fires_and_caps_stop():
    # gap up 2% over prev close 100, RVOL builds, HOD 102.5 set early, broken at 09:20
    rows = [(102, 102.5, 101.5, 102.2, 500)] + [(102.2, 102.4, 101.8, 102.3, 100)] * 4 + \
           [(102.3, 103.0, 102.2, 102.9, 300)] + [(102.9, 103.1, 102.7, 103.0, 50)] * 10
    e = hod_trigger(bars(rows), avg_vol=8_000, prev_close=100, prev_low=99, k=Cfg())
    assert "entry" in e, e
    assert abs(e["entry"] - 102.6) < 1e-6          # ceil_tick(102.5 * 1.0005) = 102.6
    assert abs(e["sl_pct"] - (102.6 - 101.05) / 102.6 * 100) < 1e-6   # LOD 101.5 -> min 1.5% widens to 101.05
    assert not e["same_day_stop"]


def test_hod_trigger_blocks_low_rvol_and_below_prev_close():
    rows = [(102, 102.5, 101.5, 102.2, 10)] * 16
    assert hod_trigger(bars(rows), 100_000, 100, 99, Cfg()) == {"blocked": "RVOL_LOW"}
    rows = [(99, 99.5, 98.8, 99.2, 1000)] * 16
    assert hod_trigger(bars(rows), 8_000, 100, 99, Cfg())["blocked"] in ("NO_BREAK", "RVOL_LOW", "LTP_BLW_PRV_CLS")


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
