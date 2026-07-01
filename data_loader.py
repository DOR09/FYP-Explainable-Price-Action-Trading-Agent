"""
data_loader.py

Data source: 14 CSV files of ES futures 5-minute bars (2021-2026),
uploaded by Kevin. No volume column in source data.

Adapted from pa_agent/data/yfinance_source.py (YFinanceSource.latest_snapshot).
Key differences from PA_Agent's live version:
  - Batch historical processing instead of real-time polling
  - Resample 5-minute bars to daily OHLCV
  - Volume synthesized from bar count per day (no raw volume available)

Pipeline:
  1. Read all CSV files from DATA_DIR
  2. Parse Unix timestamps -> UTC datetime
  3. Merge, sort, deduplicate
  4. Resample 5min -> daily OHLCV
  5. Convert rows to KlineBar objects (PA_Agent native structure)
"""
import os
import glob
import pandas as pd

from pa_agent_adapter import KlineBar, normalize_kline_bar

DATA_DIR  = "data/raw"       # folder containing 1.csv ... 14.csv
CACHE_CSV = "data/SPY_daily.csv"


def load_raw_csvs(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    Read all *.csv files in data_dir, merge into one DataFrame,
    sort by datetime, and drop duplicate timestamps.
    Each file has columns: time (Unix seconds), open, high, low, close, [ema]
    """
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in '{data_dir}'.\n"
            f"Please copy all 14 CSV files (1.csv ... 14.csv) into "
            f"PatternRecognition/data/raw/"
        )

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df.columns = [c.lower() for c in df.columns]
        df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
        dfs.append(df[["datetime", "open", "high", "low", "close"]])
        print(f"  {os.path.basename(f)}: {len(df)} rows  "
              f"({df['datetime'].min().date()} -> {df['datetime'].max().date()})")

    combined = (pd.concat(dfs)
                  .sort_values("datetime")
                  .drop_duplicates("datetime")
                  .reset_index(drop=True))
    print(f"  Total: {len(combined)} 5-minute bars after merge")
    return combined


def resample_to_daily(df5m: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 5-minute bars to daily OHLCV.

    Volume: source data contains no volume column.
    We use the count of 5-minute bars per day as a volume proxy.
    A full US trading session produces ~78 bars at 5-minute frequency.
    This preserves day-to-day relative volume variation so that the
    Breakout volume-confirmation logic in pattern_detection.py still works.
    Counts are scaled to realistic SPY-like magnitudes (~50M shares/day).
    """
    df5m = df5m.set_index("datetime")

    daily_ohlc = df5m[["open", "high", "low", "close"]].resample("1D").agg({
        "open":  "first",
        "high":  "max",
        "low":   "min",
        "close": "last",
    })

    daily_vol = df5m["close"].resample("1D").count().rename("volume")

    daily = pd.concat([daily_ohlc, daily_vol], axis=1).dropna()

    baseline_bars   = 78
    daily["volume"] = (daily["volume"] / baseline_bars * 50_000_000).astype(int)

    daily.index.name = "Date"
    daily.columns    = ["Open", "High", "Low", "Close", "Volume"]
    daily            = daily.reset_index()

    # Keep trading days only (Mon-Fri); resample('1D') may include weekends
    daily = daily[daily["Date"].dt.dayofweek < 5].reset_index(drop=True)

    print(f"  Resampled to {len(daily)} daily bars: "
          f"{daily['Date'].iloc[0].date()} -> {daily['Date'].iloc[-1].date()}")
    return daily


def build_daily_csv(data_dir: str = DATA_DIR,
                    out_path: str = CACHE_CSV) -> str:
    """
    Full build pipeline: raw CSVs -> merged 5min -> daily OHLCV -> saved CSV.
    Called once on first run; subsequent runs read the cached file directly.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"Reading raw 5-minute CSVs from {data_dir}/ ...")
    df5m  = load_raw_csvs(data_dir)
    daily = resample_to_daily(df5m)
    daily.to_csv(out_path, index=False)
    print(f"  Daily CSV saved -> {out_path}")
    return out_path


def csv_to_bars(csv_path: str = CACHE_CSV) -> tuple[pd.DataFrame, list[KlineBar]]:
    """
    Load the daily CSV and return:
      1. Clean pandas DataFrame        (used by visualize.py)
      2. List of KlineBar oldest-first (used by pattern_detection.py)

    KlineBar is PA_Agent's native bar structure (pa_agent/data/base.py).
    normalize_kline_bar() ensures OHLC values are internally consistent.
    """
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    bars: list[KlineBar] = []
    for i, row in df.iterrows():
        bar = normalize_kline_bar(KlineBar(
            seq=i + 1,
            ts_open=float(row["Date"].timestamp() * 1000),
            open=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(row["Close"]),
            volume=float(row["Volume"]),
            closed=True,
        ))
        bars.append(bar)

    print(f"Loaded {len(bars)} daily bars: "
          f"{df['Date'].iloc[0].date()} -> {df['Date'].iloc[-1].date()}")
    return df, bars