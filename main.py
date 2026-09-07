import os
import sys
import pandas as pd

import data_loader
import pattern_detection as det
import pa_features
import market_structure
import decision_engine
import explainability
import visualize


WINDOW_SIZE = 30


def main():

    # ============================================================
    # 1. Create required directories
    # ============================================================

    for directory in (
        "data",
        "data/raw",
        "output",
        "figures",
    ):
        os.makedirs(directory, exist_ok=True)

    # ============================================================
    # 2. Build daily data
    # ============================================================

    cache_path = data_loader.CACHE_CSV

    if os.path.exists(cache_path):
        print(
            f"Found cached daily data: {cache_path}"
        )
    else:
        print(
            "Daily data not found. "
            "Building from raw 5-minute CSV files..."
        )

        try:
            data_loader.build_daily_csv()
        except FileNotFoundError as exc:
            print(f"\nERROR: {exc}")
            sys.exit(1)

    # ============================================================
    # 3. Load daily data and KlineBar objects
    # ============================================================

    print("\nLoading daily data...")

    df, bars = data_loader.csv_to_bars()

    if len(df) < WINDOW_SIZE:
        print(
            f"ERROR: At least {WINDOW_SIZE} daily bars "
            f"are required."
        )
        sys.exit(1)

    print(
        f"Loaded {len(df)} daily bars."
    )

    # ============================================================
    # 4. Compute PA_Agent-style K-line features
    # ============================================================

    print("\nComputing Price Action features...")

    features = pa_features.compute_features(
        bars,
        ema_period=20,
        atr_period=14,
    )

    features_df = pd.DataFrame(features)

    features_path = "output/features.csv"

    features_df.to_csv(
        features_path,
        index=False,
    )

    print(
        f"Price Action features saved -> {features_path}"
    )

    # ============================================================
    # 5. Compute market structure
    # ============================================================

    print("\nComputing market structure...")

    market = market_structure.compute_market_structure(
        df,
        bars,
    )

    market_df = pd.DataFrame(market)

    market_path = "output/market_structure.csv"

    market_df.to_csv(
        market_path,
        index=False,
    )

    print(
        f"Market structure saved -> {market_path}"
    )

    # ============================================================
    # 6. Detect Price Action patterns
    # ============================================================

    print(
        f"\nRunning {WINDOW_SIZE}-day "
        "sliding-window pattern detection..."
    )

    raw_hits = det.scan(
        df,
        bars,
        window_size=WINDOW_SIZE,
        step=1,
    )

    events = det.merge_consecutive_hits(
        raw_hits,
        max_gap=2,
    )

    print(
        f"Detected {len(raw_hits)} raw hits "
        f"-> {len(events)} events"
    )

    # ============================================================
    # 7. Decision engine
    # ============================================================

    decisions = []

    for event in events:

        end_idx = event["end_idx"]

        price = float(
            df["Close"].iloc[end_idx]
        )

        feature = (
            features[end_idx]
            if end_idx < len(features)
            else {}
        )

        structure = (
            market[end_idx]
            if end_idx < len(market)
            else {}
        )

        decision = decision_engine.make_decision(
            pattern=event["pattern"],
            pattern_detail=event["detail"],
            feature=feature,
            market=structure,
        )

        explanation = (
            explainability.build_explanation(
                pattern=event["pattern"],
                pattern_detail=event["detail"],
                feature=feature,
                market=structure,
                decision=decision,
            )
        )

        event["decision"] = decision
        event["explanation"] = explanation

        # ========================================================
        # 8. Generate chart
        # ========================================================

        figure_path = visualize.plot_event(
            df,
            event,
        )

        decisions.append(
            {
                "Date": df["Date"].iloc[end_idx],
                "Pattern": event["pattern"],
                "Price": round(price, 2),
                "Action": decision["action"],
                "Direction": decision["direction"],
                "Confidence": decision["confidence"],
                "Score": decision["score"],
                "Explanation": explanation,
                "Figure": figure_path,
                **event["detail"],
            }
        )

    # ============================================================
    # 9. Save final results
    # ============================================================

    results_path = "output/results.csv"

    pd.DataFrame(decisions).to_csv(
        results_path,
        index=False,
    )

    print(
        f"\nFinal results saved -> {results_path}"
    )

    print(
        f"Charts saved -> figures/"
    )

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
