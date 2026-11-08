# Yearly winners study

A deep dive into NSE stocks that gained 200%+ in a calendar year. For each stock, the study:

- finds its low, breakout, 52-week-high reclaim and top;
- scores the setup on a 9-point checklist;
- compares every day with the Nifty Smallcap 250;
- builds an interactive report with candlestick charts and bar replay.

The output is one self-contained HTML file: `reports/winners_<year>.html`.

## Run

```bash
# credentials: env vars, or DHAN_CLIENT_ID=... / DHAN_ACCESS_TOKEN=... in ./.env.dhan (gitignored)
export DHAN_CLIENT_ID=...
export DHAN_ACCESS_TOKEN=...      # Dhan tokens are valid for 24h

python -m winners_study.run --year 2025                          # fetch + analyse + build
python -m winners_study.run --year 2025 --steps analyze,build    # rebuild from cache only
```

Run it from the repo root. Requires `pandas`, `numpy`, `requests`, `beautifulsoup4` and `lxml`.

| Step | What it does | Output |
|---|---|---|
| `dhan` | Daily OHLCV for each stock and the benchmark index (Dhan, split-adjusted). Covers 1 Jan of the year before to 31 Mar of the year after. | `cache/<year>/dhan/*.csv` |
| `screener` | Quarterly sales, net profit and EPS from Screener.in. Uses the consolidated page, or standalone if consolidated is stale. | `cache/<year>/screener.json` |
| `analyze` | Finds the move and its dates, computes setup metrics and earnings YoY, scores the checklist and builds the chart rows. | `cache/<year>/analysis.json`, `charts.json` |
| `build` | Fills `pipeline/template.html` with the data. | `reports/winners_<year>.html` |

`cache/` is gitignored. Use Dhan rather than Yahoo for prices: Yahoo's data is not split-adjusted for some stocks (JSLL and SIGMA in 2025).

## Adding a new year

1. Put the winners file at `inputs/winners_<year>.csv`. It needs the same columns as the 2025 file: `Stock Name`, `Maximum Potential Return(%) in <year>`, `Basic Industry`, and so on.
2. Copy `config/2025.json` to `config/<year>.json`. Set:
   - `return_col`: the return column's name in the new CSV;
   - `index_low_window`: the date range in which the benchmark made its main low that year;
   - `prior_low_window` and `retest_window`: the main low and the later retest.
3. Run all steps with a fresh Dhan token.
4. Optional: add a one-line read per stock in `notes/<year>.json`. The build lists any stocks without one.
5. The narrative text in `pipeline/template.html` is written for 2025: the thesis, the scan-rule figures and the red-flag examples. Review and rewrite it for the new year. Everything else on the page is computed from the data.

## Definitions

- **Move**: the largest rise from a low in the year to a later high in the same year.
- **Breakout**: the first close after the low above the prior 50-session high, on at least 1.5× the 50-day average volume.
- **Reclaim**: the first close above the prior 250-session high.
- **RS**: stock close ÷ benchmark close. It is used in the checklist and the relative-strength story; the chart does not draw it.
- **Excess**: the stock's daily % change minus the benchmark's daily % change.
- **Result date**: an estimate, quarter end + 45 days (+60 for March), the SEBI deadline.

Winners only: there is no comparison group, so the hit rates describe what winners looked like, not how often the setup works.
