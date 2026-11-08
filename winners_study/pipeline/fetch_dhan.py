"""Step 1: daily OHLCV for every winner + the benchmark index from Dhan (split-adjusted).

Window: 1 Jan of the year before -> 31 Mar of the year after, so the 200-day average
and the post-move bars are available.
"""
import time
import pandas as pd, requests
from .common import ROOT, dhan_creds

SCRIP_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
HIST_URL = "https://api.dhan.co/v2/charts/historical"


def _get(headers, sid, seg, inst, frm, to):
    j = requests.post(HIST_URL, headers=headers, timeout=30, json={
        "securityId": str(sid), "exchangeSegment": seg, "instrument": inst,
        "expiryCode": 0, "oi": False, "fromDate": frm, "toDate": to}).json()
    if "timestamp" not in j:
        return None, str(j)[:200]
    ts = pd.to_datetime(j["timestamp"], unit="s", utc=True).tz_convert("Asia/Kolkata").normalize().tz_localize(None)
    return pd.DataFrame({"date": ts, "open": j["open"], "high": j["high"], "low": j["low"],
                         "close": j["close"], "volume": j["volume"]}), None


def run(cfg):
    y = cfg["year"]; frm, to = f"{y-1}-01-01", f"{y+1}-03-31"
    cid, tok = dhan_creds()
    H = {"access-token": tok, "client-id": cid, "Content-Type": "application/json", "Accept": "application/json"}
    scrip = cfg["cache"].parent / "scrip_master.csv"
    if not scrip.exists():
        scrip.write_bytes(requests.get(SCRIP_URL, timeout=120).content)
    sm = pd.read_csv(scrip, low_memory=False)
    eq = sm[(sm.SEM_EXM_EXCH_ID == "NSE") & (sm.SEM_SEGMENT == "E") & (sm.SEM_INSTRUMENT_NAME == "EQUITY")]
    out = cfg["cache"] / "dhan"
    w = pd.read_csv(ROOT / cfg["input_csv"])
    for s in w["Stock Name"]:
        m = eq[eq.SEM_TRADING_SYMBOL == s]
        if m.empty:
            print(f"  {s}: no NSE security id - skipped"); continue
        df, err = _get(H, m.SEM_SMST_SECURITY_ID.iloc[0], "NSE_EQ", "EQUITY", frm, to)
        if df is None:
            print(f"  {s}: {err}"); continue
        df.to_csv(out / f"{s}.csv", index=False); print(f"  {s}: {len(df)} bars")
        time.sleep(0.25)
    df, err = _get(H, cfg["index_security_id"], "IDX_I", "INDEX", frm, to)
    if df is None:
        raise SystemExit(f"index fetch failed: {err}")
    df.to_csv(out / "INDEX.csv", index=False); print(f"  INDEX: {len(df)} bars")
