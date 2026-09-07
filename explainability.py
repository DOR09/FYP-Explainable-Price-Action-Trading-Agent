def build_explanation(
    pattern,
    pattern_detail,
    feature,
    market,
    decision,
):

    parts = []

    parts.append(
        f"Detected pattern: {pattern}."
    )

    trend = market.get(
        "trend",
        "unknown",
    )

    structure = market.get(
        "structure",
        "mixed",
    )

    parts.append(
        f"Market trend is {trend}."
    )

    parts.append(
        f"Market structure is {structure}."
    )

    ema_relation = feature.get(
        "ema_relation",
        "unknown",
    )

    parts.append(
        f"Price is {ema_relation} EMA20."
    )

    body_ratio = feature.get(
        "body_ratio"
    )

    if body_ratio is not None:

        parts.append(
            f"Current candle body ratio "
            f"is {body_ratio:.3f}."
        )

    range_atr = feature.get(
        "range_atr_ratio"
    )

    if range_atr is not None:

        parts.append(
            f"Range/ATR ratio is "
            f"{range_atr:.3f}."
        )

    micro = feature.get(
        "micro_double",
        "none",
    )

    if micro != "none":

        parts.append(
            f"Micro pattern detected: "
            f"{micro}."
        )

    action = decision[
        "action"
    ]

    confidence = decision[
        "confidence"
    ]

    parts.append(
        f"Final decision: {action} "
        f"with {confidence}% confidence."
    )

    return " ".join(parts)
