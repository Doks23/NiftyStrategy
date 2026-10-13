"""Run: python -I tests/test_pullback.py  (from repo root, with repo on sys.path)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from pullback.bhavcopy import adj_factor
from pullback.backtest import P, avwap_from_high, indicators, run


def test_adj_factor_bonus():
    d = pd.bdate_range("2024-01-01", periods=5)
    c = pd.DataFrame({"X": [100, 102, 51, 52, 53.0]}, index=d)
    pc = pd.DataFrame({"X": [99, 100, 51, 51, 52.0]}, index=d)  # 1:1 bonus ex on day 3, NSE adjusts PREVCLOSE
    adj = (c * adj_factor(c, pc))["X"].round(2).tolist()
    assert adj == [50.0, 51.0, 51.0, 52.0, 53.0], adj


def test_avwap_anchor():
    h = np.array([[1.0], [5.0], [2.0], [2.0]])
    tp = np.array([[1.0], [4.0], [2.0], [3.0]])
    v = np.ones((4, 1))
    out = avwap_from_high(h, tp, v, win=3)
    assert np.isnan(out[1, 0]) and out[2, 0] == 3.0 and out[3, 0] == 3.0  # anchored at bar 1


def test_stop_and_trail():
    # uptrend, pullback-reclaim signal, then a stop-out
    n = 120
    d = pd.bdate_range("2020-01-01", periods=n)
    c = np.linspace(100, 200, n)
    o, h, l = c * 0.995, c * 1.02, c * 0.98
    l[100] = c[100] * 0.96  # flush into the 9 EMA area
    pn = {k: pd.DataFrame({"S": v}, index=d) for k, v in dict(Open=o, High=h, Low=l, Close=c, Volume=np.ones(n)).items()}
    ind = indicators(pn)
    tr, eq = run(ind, P(min_adr=0, min_mom=0.0, max_stop=0.1, adr_frac=10, touch_tol=0.05))
    assert len(tr) > 0 and np.isfinite(tr.R).all()


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
