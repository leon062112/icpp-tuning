"""
Experiment 2 – Plotting

Figure 3: Parameter → Hardware Behavior (Controlled Experiments)
  (a) Tile Size:   X=BLOCK_M, left=L2 Hit Rate, right=Occupancy, bars=Latency
  (b) num_warps:   X=warps,   left=Occupancy,    right=Stall Not Selected, bars=Latency
  (c) num_stages:  X=stages,  left=Stall Mem Dep, right=DRAM Throughput, bars=Latency
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_results(path=None):
    if path is None:
        path = os.path.join(SCRIPT_DIR, "ncu_results.json")
    with open(path) as f:
        return json.load(f)


def _extract(results, metric_key):
    """Extract a metric value list from experiment results."""
    vals = []
    for r in results:
        v = r["metrics"].get(metric_key)
        vals.append(v if v is not None else 0.0)
    return vals


def plot_subplot(ax, x_labels, latencies, metric_left, metric_right,
                 label_left, label_right, color_left="#1565C0", color_right="#C62828",
                 title="", xlabel=""):
    """Draw one subplot: gray latency bars + two metric lines on dual Y axes."""
    x = np.arange(len(x_labels))

    # ── Latency bars (primary axis) ──
    bars = ax.bar(x, latencies, width=0.45, color="#e0f2d7",
                  edgecolor="gray", alpha=0.7, zorder=1, label="Latency")
    ax.set_ylabel("Latency (μs)", color="gray", fontsize=9)
    ax.tick_params(axis="y", colors="gray")
    # Annotate bar values
    for i, v in enumerate(latencies):
        ax.text(x[i], v, f"{v:.1f}", ha="center", va="bottom",
                fontsize=8, color="gray")

    # ── Left metric line (twin axis on the left side) ──
    ax2 = ax.twinx()
    ax2.spines["left"].set_position(("outward", 50))
    ax2.yaxis.set_label_position("left")
    ax2.yaxis.tick_left()
    line_l, = ax2.plot(x, metric_left, "o-", color=color_left, linewidth=2,
                       markersize=8, zorder=3, label=label_left)
    ax2.set_ylabel(label_left, color=color_left, fontsize=9)
    ax2.tick_params(axis="y", colors=color_left)

    # ── Right metric line (twin axis on the right side) ──
    ax3 = ax.twinx()
    line_r, = ax3.plot(x, metric_right, "s--", color=color_right, linewidth=2,
                       markersize=8, zorder=3, label=label_right)
    ax3.set_ylabel(label_right, color=color_right, fontsize=9)
    ax3.tick_params(axis="y", colors=color_right)

    # ── Formatting ──
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_xlim(-0.5, len(x_labels) - 0.5)

    # Combined legend – just above the plot area
    lines = [bars, line_l, line_r]
    labels = ["Latency (μs)", label_left, label_right]
    ax.legend(lines, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              fontsize=7, ncol=3, framealpha=0.8)
    # Title above the legend
    ax.set_title(title, fontsize=11, fontweight="bold", pad=28)


def main():
    data = load_results()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5),
                             gridspec_kw={"wspace": 0.75, "width_ratios": [1, 1, 1.15]})

    # ── (a) Tile Size effect ──
    res_a = data["exp2a"]
    x_a = [r["label"] for r in res_a]
    lat_a = _extract(res_a, "latency_us")
    l2_a = _extract(res_a, "l2_hit_rate_pct")
    occ_a = _extract(res_a, "occupancy_pct")

    plot_subplot(
        axes[0], x_a, lat_a,
        metric_left=l2_a, metric_right=occ_a,
        label_left="L2 Hit Rate (%)", label_right="Occupancy (%)",
        title="(a) Tile Size Effect",
        xlabel="BLOCK_M",
    )

    # ── (b) num_warps effect ──
    res_b = data["exp2b"]
    x_b = [r["label"] for r in res_b]
    lat_b = _extract(res_b, "latency_us")
    occ_b = _extract(res_b, "occupancy_pct")
    stall_ns = _extract(res_b, "stall_not_selected")

    plot_subplot(
        axes[1], x_b, lat_b,
        metric_left=occ_b, metric_right=stall_ns,
        label_left="Occupancy (%)", label_right="Stall Not Selected\n(inst/warp)",
        title="(b) num_warps Effect",
        xlabel="num_warps",
    )

    # ── (c) num_stages effect ──
    res_c = data["exp2c"]
    x_c = [r["label"] for r in res_c]
    lat_c = _extract(res_c, "latency_us")
    stall_mem = _extract(res_c, "stall_long_scoreboard")
    dram_tp = _extract(res_c, "dram_throughput_pct")

    plot_subplot(
        axes[2], x_c, lat_c,
        metric_left=stall_mem, metric_right=dram_tp,
        label_left="Stall Mem Dep\n(inst/warp)", label_right="DRAM Throughput (%)",
        title="(c) num_stages Effect",
        xlabel="num_stages",
    )

    fig.suptitle("Figure 3: Tuning Parameter → Hardware Behavior (M=256, N=2304, K=768)",
                 fontsize=13, y=0.98)
    fig.subplots_adjust(left=0.06, right=0.94, top=0.78, bottom=0.12)

    path = os.path.join(SCRIPT_DIR, "figure3_parameter_hw_behavior.pdf")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
