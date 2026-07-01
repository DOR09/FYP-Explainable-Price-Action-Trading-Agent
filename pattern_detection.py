"""
pattern_detection.py

Rule-based pattern detectors applied to a 30-day sliding window.
Reuses PA_Agent geometry functions (via pa_agent_adapter.py) as building blocks:

  classify_bar()            <- pa_agent/ai/kline_features.py  _classify_bar()
      Used in: detect_bull_flag() to count inside/doji bars in the flag zone

  get_overlap_ratio()       <- pa_agent/ai/kline_features.py  _overlap_ratio()
      Used in: detect_bull_flag() to measure consolidation tightness

  get_micro_double()        <- pa_agent/ai/kline_features.py  _micro_double()
      Used in: detect_double_bottom/top() as supplementary micro-pattern check

  get_breakout_prev_range() <- pa_agent/ai/kline_features.py  _breakout_prev_range()
      Used in: detect_breakout() as the core structural breakout signal

Pattern naming follows PA_Agent's pattern_routing.py taxonomy:
  double_top_bottom, breakout, bull_flag
"""
import numpy as np
import pandas as pd

from pa_agent_adapter import (
    KlineBar,
    classify_bar,
    get_overlap_ratio,
    get_micro_double,
    get_breakout_prev_range,
    compute_atr,
)


# ── Pattern 1: Double Bottom ──────────────────────────────────────────────────

def detect_double_bottom(window_df: pd.DataFrame,
                          window_bars: list[KlineBar],
                          atrs: list[float],
                          min_sep: int    = 5,
                          price_tol: float = 0.03,
                          min_bounce: float = 0.05):
    """
    Two distinct lows of similar depth separated by a meaningful bounce.

    Fix over naive argmin approach:
    Split the window in half and find each half's own minimum independently.
    The original single-argmin approach points to the SAME bar when there is
    only one global minimum, making it unable to find two separate lows.

    Enhancement using PA_Agent:
    Also calls get_micro_double() to catch 2-bar MDB patterns at window end.
    """
    lows  = window_df["Low"].values
    highs = window_df["High"].values
    n     = len(lows)
    if n < min_sep * 2:
        return False, {}

    mid  = n // 2
    idx1 = int(np.argmin(lows[:mid]))
    idx2 = mid + int(np.argmin(lows[mid:]))

    if idx2 - idx1 < min_sep:
        return False, {}

    low1, low2 = lows[idx1], lows[idx2]
    if abs(low1 - low2) / low1 > price_tol:
        return False, {}

    bounce_high = highs[idx1:idx2 + 1].max()
    bounce_pct  = (bounce_high - min(low1, low2)) / min(low1, low2)
    if bounce_pct < min_bounce:
        return False, {}

    # Supplementary micro double bottom check (PA_Agent: get_micro_double)
    micro = "none"
    if len(window_bars) >= 2 and atrs:
        atr_val = atrs[-1] if not np.isnan(atrs[-1]) else 0.0
        micro   = get_micro_double(window_bars[-1], window_bars[-2], atr_val)

    return True, {
        "low1":         round(float(low1), 2),
        "low2":         round(float(low2), 2),
        "bounce_pct":   round(float(bounce_pct * 100), 2),
        "micro_double": micro,
    }


# ── Pattern 2: Double Top ─────────────────────────────────────────────────────

def detect_double_top(window_df: pd.DataFrame,
                       window_bars: list[KlineBar],
                       atrs: list[float],
                       min_sep: int    = 5,
                       price_tol: float = 0.03,
                       min_drop: float  = 0.05):
    """
    Two similar-height peaks with a meaningful dip between them.
    Mirror logic of detect_double_bottom.
    Also calls get_micro_double() to catch 2-bar MDT patterns at window end.
    """
    highs = window_df["High"].values
    lows  = window_df["Low"].values
    n     = len(highs)
    if n < min_sep * 2:
        return False, {}

    mid  = n // 2
    idx1 = int(np.argmax(highs[:mid]))
    idx2 = mid + int(np.argmax(highs[mid:]))

    if idx2 - idx1 < min_sep:
        return False, {}

    high1, high2 = highs[idx1], highs[idx2]
    if abs(high1 - high2) / high1 > price_tol:
        return False, {}

    dip_low  = lows[idx1:idx2 + 1].min()
    drop_pct = (max(high1, high2) - dip_low) / max(high1, high2)
    if drop_pct < min_drop:
        return False, {}

    micro = "none"
    if len(window_bars) >= 2 and atrs:
        atr_val = atrs[-1] if not np.isnan(atrs[-1]) else 0.0
        micro   = get_micro_double(window_bars[-1], window_bars[-2], atr_val)

    return True, {
        "high1":        round(float(high1), 2),
        "high2":        round(float(high2), 2),
        "drop_pct":     round(float(drop_pct * 100), 2),
        "micro_double": micro,
    }


# ── Pattern 3: Breakout ───────────────────────────────────────────────────────

def detect_breakout(window_df: pd.DataFrame,
                     window_bars: list[KlineBar],
                     volume_mult: float = 1.0):
    """
    Wraps PA_Agent's get_breakout_prev_range() (kline_features._breakout_prev_range)
    and adds volume confirmation on top of the structural breakout signal.

    In the full PA_Agent system this feeds into LLM prompt context for
    further reasoning. Here we use it directly as a rule-based signal.
    """
    if len(window_bars) < 6:
        return False, {}

    # PA_Agent function expects bars in newest-first order
    bars_newest_first = list(reversed(window_bars))
    direction = get_breakout_prev_range(bars_newest_first, idx=0, lookback=5)

    if direction not in ("up", "down", "both"):
        return False, {}

    # Volume confirmation
    last_vol = float(window_df["Volume"].iloc[-1])
    avg_vol  = float(window_df["Volume"].iloc[:-1].mean())
    if avg_vol > 0 and last_vol < volume_mult * avg_vol:
        return False, {}

    resistance = window_df["High"].iloc[:-1].max()
    return True, {
        "direction":      direction,
        "breakout_close": round(float(window_df["Close"].iloc[-1]), 2),
        "resistance":     round(float(resistance), 2),
        "volume_ratio":   round(float(last_vol / avg_vol) if avg_vol > 0 else 1.0, 2),
    }


# ── Pattern 4: Bull Flag ──────────────────────────────────────────────────────

def detect_bull_flag(window_df: pd.DataFrame,
                      window_bars: list[KlineBar],
                      pole_thresh: float      = 0.10,
                      flag_max_overlap: float  = 0.70):
    """
    Pole (strong rally) followed by Flag (tight consolidation).

    Enhanced using PA_Agent geometry functions:
    - classify_bar(): counts inside/doji bars in flag zone
      (PA_Agent uses this to identify 'ii' consolidation patterns)
    - get_overlap_ratio(): measures price overlap between adjacent bars
      (PA_Agent uses this for barbwire/overlap detection)

    Flag is confirmed when avg overlap is high OR inside/doji ratio is high.
    """
    closes   = window_df["Close"].values
    n        = len(closes)
    pole_len = n // 2

    pole_move = (closes[pole_len - 1] - closes[0]) / closes[0]
    if pole_move < pole_thresh:
        return False, {}

    flag_bars = window_bars[pole_len:]
    if len(flag_bars) < 2:
        return False, {}

    overlaps     = []
    inside_count = 0
    for i in range(1, len(flag_bars)):
        bar_type = classify_bar(flag_bars[i], flag_bars[i - 1])
        if bar_type in ("inside", "doji"):
            inside_count += 1
        ov = get_overlap_ratio(flag_bars[i], flag_bars[i - 1])
        if ov is not None:
            overlaps.append(ov)

    avg_overlap  = float(np.mean(overlaps)) if overlaps else 0.0
    inside_ratio = inside_count / max(1, len(flag_bars) - 1)

    if avg_overlap < flag_max_overlap and inside_ratio < 0.4:
        return False, {}

    return True, {
        "pole_move_pct":     round(float(pole_move * 100), 2),
        "flag_avg_overlap":  round(avg_overlap, 3),
        "flag_inside_ratio": round(inside_ratio, 3),
    }


# ── Sliding window scan ───────────────────────────────────────────────────────

def scan(df: pd.DataFrame,
         bars: list[KlineBar],
         window_size: int = 30,
         step: int        = 1) -> list[dict]:
    """
    Slide a window of `window_size` days across the full history one day at a time.
    Run all four detectors on every window.
    Returns raw hits — the same real-world pattern may appear in many
    consecutive windows and will be merged by merge_consecutive_hits().
    """
    atrs = compute_atr(bars, period=14)
    hits = []

    for i in range(0, len(df) - window_size + 1, step):
        w_df   = df.iloc[i:i + window_size]
        w_bars = bars[i:i + window_size]
        w_atrs = atrs[i:i + window_size]

        for name, fn, needs_atr in [
            ("Double Bottom", detect_double_bottom, True),
            ("Double Top",    detect_double_top,    True),
            ("Breakout",      detect_breakout,      False),
            ("Bull Flag",     detect_bull_flag,     False),
        ]:
            ok, detail = fn(w_df, w_bars, w_atrs) if needs_atr else fn(w_df, w_bars)
            if ok:
                hits.append({
                    "pattern":   name,
                    "start_idx": i,
                    "end_idx":   i + window_size - 1,
                    "detail":    detail,
                })

    return hits


def merge_consecutive_hits(hits: list[dict], max_gap: int = 2) -> list[dict]:
    """
    A single real-world pattern triggers many consecutive overlapping windows.
    Merge same-pattern hits whose start indices are within max_gap of each other,
    keeping only the first (earliest, cleanest) detection per group.
    """
    by_pattern: dict[str, list[dict]] = {}
    for h in hits:
        by_pattern.setdefault(h["pattern"], []).append(h)

    merged = []
    for pattern, plist in by_pattern.items():
        plist = sorted(plist, key=lambda h: h["start_idx"])
        group = [plist[0]]
        for h in plist[1:]:
            if h["start_idx"] - group[-1]["start_idx"] <= max_gap:
                group.append(h)
            else:
                merged.append(group[0])
                group = [h]
        merged.append(group[0])

    return sorted(merged, key=lambda h: h["start_idx"])