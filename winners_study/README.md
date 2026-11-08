# Yearly winners study

A deep dive into NSE stocks that gained 200%+ in a calendar year. For each stock, the study:

- finds its low, breakout, 52-week-high reclaim and top;
- scores the setup on a 9-point checklist;
- compares every day with the Nifty Smallcap 250;
- builds an interactive report with candlestick charts and bar replay.

On each chart you can show or hide EMA 10, EMA 20, the 50 DMA and the 200 DMA. The **Measure %** tool (shortcut: M) works like TradingView's ruler: click two points and it shows the % move, the price change, and the bars and days between them.

The output is one self-contained HTML file: `reports/winners_<year>.html`.

## Run

```bash
# credentials: env vars, or DHAN_CLIENT_ID=... / DHAN_ACCESS_TOKEN=... in ./.env.dhan (gitignored)
export DHAN_CLIENT_ID=...
export DHAN_ACCESS_TOKEN=...      # Dhan tokens are valid for 24h

python -m winners_study.run --year 2025                          # fetch + analyse + build
python -m winners_study.run --year 2025 --steps analyze,build    # rebuild from cache only
python -m winners_study.run --year 2025 --refresh                # re-download everything
```

Downloads are incremental: Dhan and Screener data already in `cache/` is reused, so adding names only fetches the new ones.

Run it from the repo root. Requires `pandas`, `numpy`, `requests`, `beautifulsoup4` and `lxml`.

| Step | What it does | Output |
|---|---|---|
| `dhan` | Daily OHLCV for each stock and the benchmark index (Dhan, split-adjusted). Covers 1 Jan of the year before to 31 Mar of the year after. | `cache/<year>/dhan/*.csv` |
| `screener` | Quarterly sales, net profit and EPS from Screener.in. Uses the consolidated page, or standalone if consolidated is stale. | `cache/<year>/screener.json` |
| `analyze` | Finds the move and its dates, computes setup metrics and earnings YoY, scores the checklist and builds the chart rows. | `cache/<year>/analysis.json`, `charts.json` |
| `build` | Fills `pipeline/template.html` with the data. | `reports/winners_<year>.html` |

`cache/` is gitignored. Use Dhan rather than Yahoo for prices: Yahoo's data is not split-adjusted for some stocks (JSLL and SIGMA in 2025).

## Inputs, screening and tiers

`input_csv` in the config can list several files. They are merged, and the first occurrence of a symbol wins. Only `Stock Name` and `Basic Industry` are required.

Each stock's tier comes from its actual low-to-high move in the year, measured on Dhan prices. Tier 1 is 200% or more (`tier1_min` in the config), Tier 2 is 100–200%. The report's group switch shows the summaries for Tier 1, Tier 2 or all stocks.

2025 has two input files:
- `inputs/winners_2025.csv`: the original 32.
- `inputs/winners_2025_additions.csv`: 50 names screened from `inputs/winnerdb_2025_raw.csv`.

`inputs/winnerdb_2025_screening.csv` gives the keep or remove decision and the reason for every row of the raw file. 35 were removed: 10 duplicates of names already in the study (SMLISUZU is SMLMAH), 4 that could not be verified (a rights entitlement, a missing symbol, and two whose file details don't match the NSE symbol), and 21 that moved less than 100% in 2025. That raw file's "gain from 52-week low" is measured up to the export date, not over 2025, and some figures include unadjusted splits.

## Adding a new year

1. Put the winners file at `inputs/winners_<year>.csv`. It needs the same columns as the 2025 file: `Stock Name`, `Maximum Potential Return(%) in <year>`, `Basic Industry`, and so on.
2. Copy `config/2025.json` to `config/<year>.json`. Set:
   - `return_col`: the return column's name in the new CSV;
   - `index_low_window`: the date range in which the benchmark made its main low that year;
   - `prior_low_window` and `retest_window`: the main low and the later retest.
3. Run all steps with a fresh Dhan token.
4. Optional: add a one-line read per stock in `notes/<year>.json`. The build lists any stocks without one, and the page shows a summary generated from the data for those stocks.
5. The narrative text in `pipeline/template.html` is written for 2025: the thesis, the scan-rule figures and the red-flag examples. Review and rewrite it for the new year. Everything else on the page is computed from the data.

## Definitions

- **Move**: the largest rise from a low in the year to a later high in the same year.
- **Breakout**: the first close after the low above the prior 50-session high, on at least 1.5× the 50-day average volume.
- **Reclaim**: the first close above the prior 250-session high.
- **RS**: stock close ÷ benchmark close. It is used in the checklist and the relative-strength story; the chart does not draw it.
- **Excess**: the stock's daily % change minus the benchmark's daily % change.
- **Result date**: an estimate, quarter end + 45 days (+60 for March), the SEBI deadline.

Winners only: there is no comparison group, so the hit rates describe what winners looked like, not how often the setup works.
