"""
visualize.py
Generate an annotated candlestick chart for each detected pattern event
and save it as a PNG file to the figures/ directory.
"""
import os
import matplotlib.pyplot as plt
import mplfinance as mpf


def plot_event(df, event, pad: int = 5, out_dir: str = "figures") -> str:
    """
    Plot one pattern event as a candlestick chart with a red detection box.

    Parameters
    ----------
    df      : full daily DataFrame
    event   : dict with keys pattern, start_idx, end_idx, detail
    pad     : extra bars to show on each side of the detection window
    out_dir : folder to save the PNG
    """
    os.makedirs(out_dir, exist_ok=True)

    # Slice with padding on both sides
    s = max(0, event["start_idx"] - pad)
    e = min(len(df) - 1, event["end_idx"] + pad)
    window = df.iloc[s:e + 1].set_index("Date")[
        ["Open", "High", "Low", "Close", "Volume"]
    ]

    slug     = event["pattern"].lower().replace(" ", "_")
    out_path = f"{out_dir}/{slug}_{event['start_idx']:05d}.png"

    start_date = df["Date"].iloc[event["start_idx"]].date()
    end_date   = df["Date"].iloc[event["end_idx"]].date()

    fig, axes = mpf.plot(
        window,
        type="candle",
        style="yahoo",
        volume=True,
        title=f"{event['pattern']}  ({start_date} -> {end_date})",
        returnfig=True,
        figsize=(10, 6),
    )
    ax = axes[0]

    # Red shaded box marking the detection window (inside the padding)
    box_s = event["start_idx"] - s
    box_e = event["end_idx"]   - s
    ax.axvspan(box_s, box_e, color="red", alpha=0.08)
    ax.axvline(box_s, color="red", linestyle="--", linewidth=1)
    ax.axvline(box_e, color="red", linestyle="--", linewidth=1)

    # Show pattern detail values below the chart
    detail_str = "  |  ".join(f"{k}={v}" for k, v in event["detail"].items())
    ax.set_xlabel(detail_str, fontsize=8)

    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path