"""
main.py  --  Pattern Recognition Module, Day 1
FYP: An Explainable Intelligent Agent for Price Action Trading using VLMs

Data source : ES futures 5-minute bars (2021-2026), resampled to daily.
Ticker      : SPY (ES futures used as proxy; same underlying index)

Full pipeline
-------------
1. Read 14 raw CSV files from data/raw/
       data_loader.load_raw_csvs()
2. Resample 5-minute -> daily OHLCV
       data_loader.resample_to_daily()
3. Convert rows to KlineBar objects
       pa_agent_adapter.KlineBar  (adapted from pa_agent/data/base.py)
4. Sliding window scan (30-day window, step=1 day)
       pattern_detection.scan()
       uses classify_bar(), get_overlap_ratio(), get_micro_double(),
       get_breakout_prev_range() from pa_agent/ai/kline_features.py
5. Merge duplicate hits from overlapping windows
       pattern_detection.merge_consecutive_hits()
6. Save results.csv + one annotated PNG chart per event
       visualize.plot_event()
"""
import os
import sys
import pandas as pd

import data_loader
import pattern_detection as det
import visualize

WINDOW_SIZE = 30   # sliding window size in trading days


def main():
    for d in ("data/raw", "data", "figures", "output"):
        os.makedirs(d, exist_ok=True)

    # ── Step 1: Build daily CSV from raw 5-minute files (first run only) ──────
    if os.path.exists(data_loader.CACHE_CSV):
        print(f"Found cached daily CSV: {data_loader.CACHE_CSV} -- skipping rebuild.")
    else:
        print("First run: building daily CSV from raw 5-minute data...")
        try:
            data_loader.build_daily_csv()
        except FileNotFoundError as e:
            print(f"\nERROR: {e}")
            sys.exit(1)

    # ── Step 2: Load data and convert to KlineBar list ────────────────────────
    print("\nLoading data...")
    df, bars = data_loader.csv_to_bars()

    # ── Step 3: Sliding window pattern scan ───────────────────────────────────
    print(f"\nRunning {WINDOW_SIZE}-day sliding window scan...")
    raw_hits = det.scan(df, bars, window_size=WINDOW_SIZE, step=1)
    events   = det.merge_consecutive_hits(raw_hits, max_gap=2)
    print(f"  {len(raw_hits)} raw window hits  ->  {len(events)} merged events\n")

    # ── Step 4: Save results and generate charts ──────────────────────────────
    rows = []
    for ev in events:
        date  = df["Date"].iloc[ev["end_idx"]]
        price = float(df["Close"].iloc[ev["end_idx"]])
        print(f"  {ev['pattern']:<16}  {date.date()}  "
              f"close={price:.2f}  {ev['detail']}")

        fig_path = visualize.plot_event(df, ev)
        rows.append({
            "Date":    date.date(),
            "Pattern": ev["pattern"],
            "Price":   round(price, 2),
            **ev["detail"],
            "Figure":  fig_path,
        })

    out_csv = "output/results.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nDone.")
    print(f"  {len(rows)} events saved  ->  {out_csv}")
    print(f"  {len(rows)} charts saved  ->  figures/")


if __name__ == "__main__":
    main()