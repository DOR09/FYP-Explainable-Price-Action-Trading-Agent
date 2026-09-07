import numpy as np

from pa_agent_adapter import (
    KlineBar,
    compute_atr,
)


def compute_ema(
    values,
    period=20,
):

    if len(values) == 0:
        return []

    alpha = 2 / (
        period + 1
    )

    ema = values[0]

    result = [ema]

    for value in values[1:]:

        ema = (
            alpha * value
            +
            (1 - alpha) * ema
        )

        result.append(ema)

    return result


def classify_trend(
    closes,
    ema,
    lookback=5,
):

    if len(closes) < lookback:
        return "unknown"

    recent = closes[
        -lookback:
    ]

    slope = (
        recent[-1]
        -
        recent[0]
    )

    last_close = closes[-1]
    last_ema = ema[-1]

    if (
        slope > 0
        and last_close > last_ema
    ):
        return "bullish"

    if (
        slope < 0
        and last_close < last_ema
    ):
        return "bearish"

    return "sideways"


def swing_points(
    highs,
    lows,
    strength=2,
):

    result = []

    n = len(highs)

    for i in range(
        strength,
        n - strength,
    ):

        high = highs[i]
        low = lows[i]

        left_highs = highs[
            i - strength:i
        ]

        right_highs = highs[
            i + 1:
            i + strength + 1
        ]

        left_lows = lows[
            i - strength:i
        ]

        right_lows = lows[
            i + 1:
            i + strength + 1
        ]

        if (
            high >= max(
                left_highs
            )
            and
            high >= max(
                right_highs
            )
        ):

            result.append(
                {
                    "index": i,
                    "type": "swing_high",
                    "price": float(high),
                }
            )

        elif (
            low <= min(
                left_lows
            )
            and
            low <= min(
                right_lows
            )
        ):

            result.append(
                {
                    "index": i,
                    "type": "swing_low",
                    "price": float(low),
                }
            )

    return result


def label_structure(
    swings,
):

    highs = [
        s
        for s in swings
        if s["type"]
        == "swing_high"
    ]

    lows = [
        s
        for s in swings
        if s["type"]
        == "swing_low"
    ]

    if len(highs) < 2:
        high_label = "unknown"
    else:

        high_label = (
            "HH"
            if highs[-1]["price"]
            >
            highs[-2]["price"]
            else "LH"
        )

    if len(lows) < 2:
        low_label = "unknown"
    else:

        low_label = (
            "HL"
            if lows[-1]["price"]
            >
            lows[-2]["price"]
            else "LL"
        )

    if (
        high_label == "HH"
        and
        low_label == "HL"
    ):
        structure = "bullish"

    elif (
        high_label == "LH"
        and
        low_label == "LL"
    ):
        structure = "bearish"

    else:
        structure = "mixed"

    return (
        high_label,
        low_label,
        structure,
    )


def compute_market_structure(
    df,
    bars: list[KlineBar],
):

    closes = (
        df["Close"]
        .astype(float)
        .tolist()
    )

    highs = (
        df["High"]
        .astype(float)
        .tolist()
    )

    lows = (
        df["Low"]
        .astype(float)
        .tolist()
    )

    ema20 = compute_ema(
        closes,
        20,
    )

    atr14 = compute_atr(
        bars,
        14,
    )

    results = []

    for i in range(
        len(df)
    ):

        start = max(
            0,
            i - 20,
        )

        local_highs = highs[
            start:i + 1
        ]

        local_lows = lows[
            start:i + 1
        ]

        local_swings = swing_points(
            local_highs,
            local_lows,
            strength=2,
        )

        # Convert local index back to global index
        for swing in local_swings:
            swing["index"] += start

        hh, hl, structure = (
            label_structure(
                local_swings
            )
        )

        trend = classify_trend(
            closes[:i + 1],
            ema20[:i + 1],
        )

        recent_range = (
            max(
                local_highs
            )
            -
            min(
                local_lows
            )
        )

        momentum = 0.0

        if len(closes) >= 6:
            momentum = (
                closes[i]
                -
                closes[i - 5]
            )

        results.append(
            {
                "trend": trend,
                "structure": structure,
                "last_high_structure": hh,
                "last_low_structure": hl,
                "ema20": round(
                    ema20[i],
                    4,
                ),
                "atr14": (
                    round(
                        atr14[i],
                        4,
                    )
                    if not np.isnan(
                        atr14[i]
                    )
                    else None
                ),
                "recent_range": round(
                    recent_range,
                    4,
                ),
                "momentum_5d": round(
                    momentum,
                    4,
                ),
            }
        )

    return results
