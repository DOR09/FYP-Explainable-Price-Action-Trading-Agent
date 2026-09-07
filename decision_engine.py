def make_decision(
    pattern,
    pattern_detail,
    feature,
    market,
):

    score = 0

    reasons = []

    trend = market.get(
        "trend",
        "unknown",
    )

    structure = market.get(
        "structure",
        "mixed",
    )

    ema_relation = feature.get(
        "ema_relation",
        "unknown",
    )

    breakout = feature.get(
        "breakout_prev",
        "none",
    )

    follow = feature.get(
        "follow_through",
        "pending",
    )

    # ============================================================
    # Pattern direction
    # ============================================================

    if pattern in (
        "Double Bottom",
        "Bull Flag",
    ):

        score += 2

        reasons.append(
            "Pattern has bullish structure."
        )

    elif pattern == "Double Top":

        score -= 2

        reasons.append(
            "Pattern has bearish structure."
        )

    elif pattern == "Breakout":

        direction = (
            pattern_detail.get(
                "direction",
                "none",
            )
        )

        if direction == "up":

            score += 2

            reasons.append(
                "Price broke upward from "
                "the previous range."
            )

        elif direction == "down":

            score -= 2

            reasons.append(
                "Price broke downward from "
                "the previous range."
            )

    # ============================================================
    # Market trend
    # ============================================================

    if trend == "bullish":

        score += 2

        reasons.append(
            "Market trend is bullish."
        )

    elif trend == "bearish":

        score -= 2

        reasons.append(
            "Market trend is bearish."
        )

    # ============================================================
    # Market structure
    # ============================================================

    if structure == "bullish":

        score += 1

        reasons.append(
            "Market structure supports "
            "higher highs / higher lows."
        )

    elif structure == "bearish":

        score -= 1

        reasons.append(
            "Market structure supports "
            "lower highs / lower lows."
        )

    # ============================================================
    # EMA context
    # ============================================================

    if ema_relation == "above":

        score += 1

        reasons.append(
            "Price is above EMA20."
        )

    elif ema_relation == "below":

        score -= 1

        reasons.append(
            "Price is below EMA20."
        )

    # ============================================================
    # Follow-through
    # ============================================================

    if follow == "yes":

        if score >= 0:
            score += 1
        else:
            score -= 1

        reasons.append(
            "Follow-through confirms "
            "the previous directional move."
        )

    elif follow == "failed":

        if score > 0:
            score -= 1
        elif score < 0:
            score += 1

        reasons.append(
            "Follow-through failed."
        )

    # ============================================================
    # Convert score into decision
    # ============================================================

    if score >= 4:

        action = "BUY"
        direction = "bullish"

    elif score <= -4:

        action = "SELL"
        direction = "bearish"

    else:

        action = "WAIT"
        direction = (
            "bullish"
            if score > 0
            else
            "bearish"
            if score < 0
            else
            "neutral"
        )

    confidence = min(
        95,
        max(
            50,
            50 + abs(score) * 8,
        ),
    )

    return {
        "action": action,
        "direction": direction,
        "score": score,
        "confidence": confidence,
        "reasons": reasons,
    }
