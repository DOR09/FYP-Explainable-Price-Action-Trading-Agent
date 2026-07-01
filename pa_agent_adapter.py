"""
pa_agent_adapter.py

Core structures and functions extracted directly from PA_Agent source code:
  - KlineBar                  <- pa_agent/data/base.py
  - normalize_kline_bar()     <- pa_agent/data/base.py
  - classify_bar()            <- pa_agent/ai/kline_features.py
  - get_overlap_ratio()       <- pa_agent/ai/kline_features.py
  - get_micro_double()        <- pa_agent/ai/kline_features.py
  - get_breakout_prev_range() <- pa_agent/ai/kline_features.py
  - compute_atr()             <- based on PA_Agent ATR logic
"""
from __future__ import annotations
import math
from dataclasses import dataclass


# ── KlineBar ──────────────────────────────────────────────────────────────────
# Source: pa_agent/data/base.py
@dataclass(frozen=True)
class KlineBar:
    seq: int          # bar sequence number; 1 = newest closed bar
    ts_open: float    # bar open timestamp in milliseconds (UTC)
    open: float
    high: float
    low: float
    close: float
    volume: float
    closed: bool = True


def normalize_kline_bar(bar: KlineBar) -> KlineBar:
    """
    Source: pa_agent/data/base.py  normalize_kline_bar()
    Ensures high >= low and low <= close <= high.
    """
    high  = max(bar.high, bar.low)
    low   = min(bar.high, bar.low)
    close = max(low, min(high, bar.close))
    if high == bar.high and low == bar.low and close == bar.close:
        return bar
    return KlineBar(
        seq=bar.seq, ts_open=bar.ts_open,
        open=bar.open, high=high, low=low, close=close,
        volume=bar.volume, closed=bar.closed,
    )


# ── Single-bar geometry features ─────────────────────────────────────────────
# Source: pa_agent/ai/kline_features.py

def classify_bar(bar: KlineBar, prev: KlineBar | None = None) -> str:
    """
    Source: pa_agent/ai/kline_features.py  _classify_bar()
    Returns: inside | outside_bull | outside_bear | doji |
             trend_bull | trend_bear | other | flat

    Used in pattern_detection.py for Bull Flag detection:
    counts inside/doji bars in the flag zone as consolidation signals.
    """
    high       = max(bar.high, bar.low)
    low        = min(bar.high, bar.low)
    full_range = high - low
    body       = abs(bar.close - bar.open)

    body_ratio     = body / full_range if full_range > 0 else None
    close_position = (
        max(0.0, min(1.0, (bar.close - low) / full_range))
        if full_range > 0 else None
    )

    if prev is not None:
        if bar.high <= prev.high and bar.low >= prev.low:
            return "inside"
        if bar.high >= prev.high and bar.low <= prev.low:
            return "outside_bull" if bar.close >= bar.open else "outside_bear"

    if body_ratio is None or close_position is None:
        return "flat"
    if body_ratio <= 0.25:
        return "doji"
    if bar.close > bar.open and close_position >= 0.65:
        return "trend_bull"
    if bar.close < bar.open and close_position <= 0.35:
        return "trend_bear"
    return "other"


def get_overlap_ratio(bar: KlineBar, prev: KlineBar | None) -> float | None:
    """
    Source: pa_agent/ai/kline_features.py  _overlap_ratio()
    Price overlap between two adjacent bars as a fraction of their combined range.
    High overlap = consolidation / flag zone.
    Low overlap  = trending / directional move.
    Used in pattern_detection.py to measure flag zone tightness.
    """
    if prev is None:
        return None
    high        = min(bar.high, prev.high)
    low         = max(bar.low,  prev.low)
    overlap     = max(0.0, high - low)
    denominator = max(bar.high, prev.high) - min(bar.low, prev.low)
    if denominator <= 0:
        return None
    return overlap / denominator


def get_micro_double(bar: KlineBar, prev: KlineBar | None, atr: float) -> str:
    """
    Source: pa_agent/ai/kline_features.py  _micro_double()
    Returns 'MDB' (Micro Double Bottom), 'MDT' (Micro Double Top), or 'none'.
    Two adjacent bars with nearly identical lows => MDB; highs => MDT.
    Used in pattern_detection.py as supplementary confirmation for
    Double Bottom / Double Top patterns.
    """
    if prev is None:
        return "none"
    tolerance = atr * 0.02 if (not math.isnan(atr) and atr > 0) else 0.0
    if abs(bar.low  - prev.low)  <= tolerance:
        return "MDB"
    if abs(bar.high - prev.high) <= tolerance:
        return "MDT"
    return "none"


def get_breakout_prev_range(bars: list[KlineBar], idx: int,
                             lookback: int = 5) -> str:
    """
    Source: pa_agent/ai/kline_features.py  _breakout_prev_range()
    Returns 'up' | 'down' | 'both' | 'none'.
    Checks whether bars[idx] breaks above/below the high/low of the
    previous `lookback` bars.
    Used directly as the core Breakout detector in pattern_detection.py.
    """
    prev_bars = bars[idx + 1: idx + 1 + lookback]
    if not prev_bars:
        return "none"
    broke_high = bars[idx].high > max(b.high for b in prev_bars)
    broke_low  = bars[idx].low  < min(b.low  for b in prev_bars)
    if broke_high and broke_low:
        return "both"
    if broke_high:
        return "up"
    if broke_low:
        return "down"
    return "none"


def compute_atr(bars: list[KlineBar], period: int = 14) -> list[float]:
    """
    Average True Range over `period` bars.
    bars must be sorted oldest-first.
    Required as input to get_micro_double().
    """
    n   = len(bars)
    trs = []
    for i in range(n):
        high       = bars[i].high
        low        = bars[i].low
        prev_close = bars[i - 1].close if i > 0 else bars[i].close
        tr = max(high - low,
                 abs(high - prev_close),
                 abs(low  - prev_close))
        trs.append(tr)

    atrs = [math.nan] * n
    if n >= period:
        atrs[period - 1] = sum(trs[:period]) / period
        for i in range(period, n):
            atrs[i] = (atrs[i - 1] * (period - 1) + trs[i]) / period
    return atrs