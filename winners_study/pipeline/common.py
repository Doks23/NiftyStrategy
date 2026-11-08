"""Shared paths, config and Dhan credential loading for the winners study."""
import json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # winners_study/
CACHE = ROOT / "cache"                                  # gitignored: raw downloads


def load_config(year):
    cfg = json.loads((ROOT / "config" / f"{year}.json").read_text())
    cfg["cache"] = CACHE / str(year)
    cfg["cache"].mkdir(parents=True, exist_ok=True)
    (cfg["cache"] / "dhan").mkdir(exist_ok=True)
    return cfg


def dhan_creds(env_path=".env.dhan"):
    """DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN from env, else from .env.dhan (same as niftyvwap/data.py)."""
    p = Path(env_path)
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    cid, tok = os.environ.get("DHAN_CLIENT_ID"), os.environ.get("DHAN_ACCESS_TOKEN")
    if not cid or not tok:
        raise SystemExit("Set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN (env or .env.dhan)")
    return cid, tok
