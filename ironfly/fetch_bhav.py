"""NSE F&O UDiFF bhavcopy (public archive, EOD) -> NIFTY index options + futures only.
Usage: python -m ironfly.fetch_bhav 2024-01-01 2026-10-24   -> data_if/nifty_fo.csv.gz"""
import io, sys, zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
import pandas as pd, requests

OUT = Path("data_if")
URL = "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{}_F_0000.csv.zip"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}
KEEP = ["TradDt", "FinInstrmTp", "XpryDt", "StrkPric", "OptnTp", "OpnPric", "HghPric", "LwPric", "ClsPric",
        "SttlmPric", "UndrlygPric", "OpnIntrst", "TtlTradgVol", "NewBrdLotQty"]


def one(d):
    f = OUT / "days" / f"{d:%Y%m%d}.csv"
    if f.exists():
        return d, "cached"
    for _ in range(3):
        try:
            r = requests.get(URL.format(f"{d:%Y%m%d}"), headers=UA, timeout=60)
        except requests.RequestException:
            continue
        if r.status_code == 404:
            return d, "holiday"
        if r.status_code == 200:
            z = zipfile.ZipFile(io.BytesIO(r.content))
            df = pd.read_csv(z.open(z.namelist()[0]), low_memory=False)
            df = df[(df.TckrSymb == "NIFTY") & df.FinInstrmTp.isin(["IDO", "IDF"])][KEEP]
            df.to_csv(f, index=False)
            return d, len(df)
    return d, f"fail {r.status_code}"


def main(a, b):
    (OUT / "days").mkdir(parents=True, exist_ok=True)
    days = [a + timedelta(i) for i in range((b - a).days + 1)]
    days = [d for d in days if d.weekday() < 5]
    with ThreadPoolExecutor(4) as ex:
        for d, s in ex.map(one, days):
            if isinstance(s, str) and s.startswith("fail"):
                print(d, s, flush=True)
    df = pd.concat([pd.read_csv(f) for f in sorted((OUT / "days").glob("*.csv"))])
    df.to_csv(OUT / "nifty_fo.csv.gz", index=False)
    print(len(df), df.TradDt.min(), df.TradDt.max(), df.TradDt.nunique(), "days")


if __name__ == "__main__":
    main(date.fromisoformat(sys.argv[1]), date.fromisoformat(sys.argv[2]))
