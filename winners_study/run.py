"""Yearly winners deep-dive: fetch -> analyse -> HTML report.

Usage (from the repo root):
    python -m winners_study.run --year 2025                 # all steps
    python -m winners_study.run --year 2025 --steps analyze,build
"""
import argparse
from .pipeline import analyze, build_page, fetch_dhan, fetch_screener
from .pipeline.common import load_config

STEPS = {"dhan": fetch_dhan.run, "screener": fetch_screener.run, "analyze": analyze.run, "build": build_page.run}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--steps", default="dhan,screener,analyze,build")
    a = ap.parse_args()
    cfg = load_config(a.year)
    for s in a.steps.split(","):
        print(f"[{s}]"); STEPS[s](cfg)


if __name__ == "__main__":
    main()
