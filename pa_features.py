from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class KlineBar:

    seq: int

    ts_open: float

    open: float

    high: float

    low: float

    close: float

    volume: float

    closed: bool = True


def normalize_kline_bar(
    bar: KlineBar,
) -> KlineBar:

    high = max(
        bar.high,
        bar.low,
    )

    low = min(
        bar.high,
        bar.low,
    )

    close = max(
        low,
        min(
            high,
            bar.close,
        ),
    )

    return KlineBar(
        seq=bar.seq,
        ts_open=bar.ts_open,
        open=bar.open,
        high=high,
        low=low,
        close=close,
        volume=bar.volume,
        closed=bar.closed,
    )


def classify_bar(
    bar: KlineBar,
    prev: KlineBar | None = None,
) -> str:

    high = max(
        bar.high,
        bar.low,
    )

    low = min(
        bar.high,
        bar.low,
    )

    full_range = high - low

    body = abs(
        bar.close - bar.open
    )

    if full_range <= 0:
        return "flat"

    body_ratio = (
        body / full_range
    )

    close_position = (
        bar.close - low
    ) / full_range

    if prev is not None:

        if (
            bar.high <= prev.high
            and bar.low >= prev.low
        ):
            return "inside"

        if (
            bar.high >= prev.high
            and bar.low <= prev.low
        ):

            if bar.close >= bar.open:
                return "outside_bull"

            return "outside_bear"

    if body_ratio <= 0.25:
        return "doji"

    if (
        bar.close > bar.open
        and close_position >= 0.65
    ):
        return "trend_bull"

    if (
        bar.close < bar.open
        and close_position <= 0.35
    ):
        return "trend_bear"

    return "other"


def get_overlap_ratio(
    bar: KlineBar,
    prev: KlineBar | None,
):

    if prev is None:
        return None

    overlap_high = min(
        bar.high,
        prev.high,
    )

    overlap_low = max(
        bar.low,
        prev.low,
    )

    overlap = max(
        0.0,
        overlap_high - overlap_low,
    )

    denominator = (
        max(bar.high, prev.high)
        -
        min(bar.low, prev.low)
    )

    if denominator <= 0:
        return None

    return overlap / denominator


def get_micro_double(
    bar: KlineBar,
    prev: KlineBar | None,
    atr: float,
):

    if prev is None:
        return "none"

    tolerance = 0.0

    if (
        not math.isnan(atr)
        and atr > 0
    ):
        tolerance = atr * 0.02

    if (
        abs(
            bar.low - prev.low
        )
        <= tolerance
    ):
        return "MDB"

    if (
        abs(
            bar.high - prev.high
        )
        <= tolerance
    ):
        return "MDT"

    return "none"


def get_breakout_prev_range(
    bars: list[KlineBar],
    idx: int,
    lookback: int = 5,
):

    prev_bars = bars[
        idx + 1:
        idx + 1 + lookback
    ]

    if not prev_bars:
        return "none"

    broke_high = (
        bars[idx].high
        >
        max(
            b.high
            for b in prev_bars
        )
    )

    broke_low = (
        bars[idx].low
        <
        min(
            b.low
            for b in prev_bars
        )
    )

    if broke_high and broke_low:
        return "both"

    if broke_high:
        return "up"

    if broke_low:
        return "down"

    return "none"


def compute_atr(
    bars: list[KlineBar],
    period: int = 14,
):

    trs = []

    for i, bar in enumerate(bars):

        previous_close = (
            bars[i - 1].close
            if i > 0
            else bar.close
        )

        true_range = max(
            bar.high - bar.low,
            abs(
                bar.high
                - previous_close
            ),
            abs(
                bar.low
                - previous_close
            ),
        )

        trs.append(
            true_range
        )

    atrs = [
        math.nan
    ] * len(bars)

    if len(bars) < period:
        return atrs

    atrs[period - 1] = (
        sum(trs[:period])
        / period
    )

    for i in range(
        period,
        len(bars),
    ):

        atrs[i] = (
            atrs[i - 1]
            * (period - 1)
            + trs[i]
        ) / period

    return atrs
