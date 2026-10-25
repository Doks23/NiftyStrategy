"""India VIX + NIFTY 50 daily closes from Yahoo (2023 -> today) -> data_if/vix_nifty.csv
Usage: python -m ironfly.fetch_vix"""
import pandas as pd, requests

URL = "https://query2.finance.yahoo.com/v8/finance/chart/{}?period1=1672531200&period2=1893456000&interval=1d"


def series(sym):
    j = requests.get(URL.format(sym), headers={"User-Agent": "Mozilla/5.0"}, timeout=60).json()["chart"]["result"][0]
    d = pd.to_datetime(j["timestamp"], unit="s", utc=True).tz_convert("Asia/Kolkata").tz_localize(None).normalize()
    return pd.Series(j["indicators"]["quote"][0]["close"], index=d).dropna().groupby(level=0).last()


if __name__ == "__main__":
    df = pd.DataFrame({"vix": series("%5EINDIAVIX"), "nifty": series("%5ENSEI")}).dropna()
    df.index.name = "date"
    df.to_csv("data_if/vix_nifty.csv")
    print(len(df), df.index.min().date(), df.index.max().date(), df.tail(2).to_dict("index"))
