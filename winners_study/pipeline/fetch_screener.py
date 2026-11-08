"""Step 2: quarterly results (sales, net profit, EPS) from Screener.in.

Consolidated page first; falls back to standalone when consolidated is empty or stale.
"""
import json, re, time
import pandas as pd, requests
from bs4 import BeautifulSoup
from .common import read_inputs

H = {"User-Agent": "Mozilla/5.0 (research script)"}


def _num(x):
    x = x.replace(",", "").replace("%", "").strip()
    try: return float(x)
    except ValueError: return None


def _parse(html):
    soup = BeautifulSoup(html, "lxml")
    sec = soup.find("section", id="quarters")
    if not sec or not sec.find("table"):
        return None
    t = sec.find("table")
    hdr = [th.get_text(strip=True) for th in t.find("thead").find_all("th")][1:]
    rows = {}
    for tr in t.find("tbody").find_all("tr"):
        c = [td.get_text(strip=True) for td in tr.find_all("td")]
        if c: rows[re.sub(r"[^A-Za-z ]", "", c[0]).strip()] = [_num(v) for v in c[1:]]
    sales = rows.get("Sales") or rows.get("Revenue") or rows.get("Total Income")
    if not sales or all(v in (None, 0) for v in sales[-6:]):
        return None
    about = soup.find("div", class_="company-profile")
    return {"q": hdr, "sales": sales, "np": rows.get("Net Profit"), "eps": rows.get("EPS in Rs"),
            "about": about.get_text(" ", strip=True)[:500] if about else ""}


def run(cfg):
    y = cfg["year"]
    store = cfg["cache"] / "screener.json"
    out = json.loads(store.read_text()) if store.exists() and not cfg.get("refresh") else {}
    w = read_inputs(cfg)
    for s in w["Stock Name"]:
        if s in out:
            continue                                    # already fetched (use --refresh to redo)
        best = None
        for path in (f"/company/{s}/consolidated/", f"/company/{s}/"):
            r = requests.get("https://www.screener.in" + path, headers=H, timeout=30)
            if r.status_code != 200: continue
            d = _parse(r.text)
            # stale consolidated statements (latest quarter before the study year) -> try standalone
            if d and int(d["q"][-1].split()[-1]) >= y:
                best = {"src": path, **d}; break
        out[s] = best
        print(f"  {s}: {best['src'] if best else 'NONE'}")
        time.sleep(1.2)
    store.write_text(json.dumps(out))
