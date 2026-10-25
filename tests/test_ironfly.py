"""Logic tests on a hand-built option chain priced by intrinsic + flat time value (NOT market data)."""
import sys
from types import SimpleNamespace
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from ironfly.backtest import Pos, run_cycle

A = SimpleNamespace(capital=200000, lots=1, maxloss=0.03, sl=0.04, hedge=0, max_adj=10, slip=0.0,
                    exit_days_before=0, adj_mode="roll")
EXP = pd.Timestamp("2026-01-27")


def chain(S, tv):
    """Strikes every 50 around 24000; price = intrinsic + tv * exp(-|K-S|/400)."""
    idx, px = [], []
    for k in np.arange(22000, 26050, 50.0):
        for t in ("CE", "PE"):
            iv = max(S - k, 0) if t == "CE" else max(k - S, 0)
            idx.append((k, t)); px.append(iv + tv * np.exp(-abs(k - S) / 400))
    return pd.Series(px, index=pd.MultiIndex.from_tuples(idx))


def setup(path, tv0=300):
    days = list(pd.bdate_range("2026-01-01", EXP))
    spot = pd.Series(path, index=days[:len(path)])
    ch = {(d, EXP): chain(s, tv0 * (1 - i / len(path))) for i, (d, s) in enumerate(spot.items())}
    return days[0], days[1:len(path)], spot, ch


def test_breakevens_of_iron_fly():
    p = Pos()
    for k, t, n, px in [(24000, "CE", -1, 300), (24000, "PE", -1, 300), (24500, "CE", 1, 150), (23500, "PE", 1, 150)]:
        p.trade(k, t, n, px, pd.Timestamp("2026-01-01"), 1, 0)
    # credit 300 -> breakevens 23700 / 24300
    assert p.breached(24000, 1) is None and p.breached(24300, 1) is None and p.breached(23700, 1) is None
    assert p.breached(24302, 1) == "CE" and p.breached(23698, 1) == "PE"


def test_wing_respects_3pct_budget_and_flat_market_profits():
    entry, days, spot, ch = setup([24000.0] * 18)
    r = run_cycle(entry, EXP, days, spot, ch, 65, A)
    assert (r["wing"] - r["credit"]) * 65 <= 0.03 * 200000
    assert r["adjustments"] == 0 and r["net"] > 0


def test_breach_moves_short_call_to_old_hedge():
    path = [24000.0] * 5 + [24000.0 + 60 * i for i in range(1, 14)]
    entry, days, spot, ch = setup(path)
    r = run_cycle(entry, EXP, days, spot, ch, 65, A.__class__(**{**vars(A), "sl": 1.0}))
    assert r["adjustments"] >= 1
    first = r["adj_log"].split("; ")[0]
    w = r["wing"]
    assert f"24000.0->{24000.0 + w}" in first and f"hedge {24000.0 + 2 * w}" in first


def test_recenter_opens_new_fly_at_spot():
    path = [24000.0] * 5 + [24000.0 + 60 * i for i in range(1, 14)]
    entry, days, spot, ch = setup(path)
    r = run_cycle(entry, EXP, days, spot, ch, 65, A.__class__(**{**vars(A), "sl": 1.0, "adj_mode": "recenter"}))
    first = r["adj_log"].split("; ")[0]
    S = float(first.split("S=")[1].split()[0])
    K2 = float(first.split("recenter ")[1].split()[0])
    assert abs(K2 - S) <= 25


def test_untested_moves_put_side_up():
    path = [24000.0] * 5 + [24000.0 + 60 * i for i in range(1, 14)]
    entry, days, spot, ch = setup(path)
    r = run_cycle(entry, EXP, days, spot, ch, 65, A.__class__(**{**vars(A), "sl": 1.0, "adj_mode": "untested"}))
    first = r["adj_log"].split("; ")[0]
    assert first.split()[1] == "CE" and "PE -> " in first
    K2, KL = map(float, first.split("PE -> ")[1].split("/"))
    assert K2 > 24000 and K2 - KL == r["wing"]


def test_stop_loss_exits_early():
    path = [24000.0, 24000.0] + [24000.0 + 300 * i for i in range(1, 16)]
    entry, days, spot, ch = setup(path)
    r = run_cycle(entry, EXP, days, spot, ch, 65, A)
    assert r["reason"] == "stop" and r["net"] < 0


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f(); print("ok", name)
