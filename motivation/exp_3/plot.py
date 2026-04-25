"""
Experiment 3 – Plotting

Figure 4: Hardware Bottleneck Migration across Dynamic Shapes
  1×4 layout: Occupancy, L2 Hit Rate, DRAM Throughput, Stall Mem Dep
  Each subplot: Config-A solid, Config-B dashed
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

COLOR_A = "#1565C0"
COLOR_B = "#C62828"


def load_results(path=None):
    if path is None:
        path = os.path.join(SCRIPT_DIR, "ncu_results.json")
    with open(path) as f:
        return json.load(f)


def _extract(results_list, metric_key):
    return [r["metrics"].get(metric_key, 0) for r in results_list]


def _get_M_values(results_list):
    return [r["M"] for r in results_list]


def plot_bottleneck_migration(data):
    config_names = list(data.keys())
    assert len(config_names) == 2
    name_a, name_b = config_names
    res_a, res_b = data[name_a], data[name_b]
    M_vals = _get_M_values(res_a)

    label_a = "Config-A  (BM128×BN128×BK32, w4 s3)"
    label_b = "Config-B  (BM32×BN32×BK64, w4 s5)"

    metrics_info = [
        ("occupancy_pct",         "Occupancy (%)",             "(a)"),
        ("l2_hit_rate_pct",       "L2 Hit Rate (%)",           "(b)"),
        ("dram_throughput_pct",   "DRAM Throughput (%)",        "(c)"),
        ("stall_long_scoreboard", "Stall Mem Dep (inst/warp)",  "(d)"),
    ]

    x = np.arange(len(M_vals))
    x_labels = [str(m) for m in M_vals]

    fig, axes = plt.subplots(1, 4, figsize=(22, 4.8))

    for idx, (metric_key, ylabel, tag) in enumerate(metrics_info):
        ax = axes[idx]
        vals_a = _extract(res_a, metric_key)
        vals_b = _extract(res_b, metric_key)

        ax.plot(x, vals_a, "o-", color=COLOR_A, linewidth=2.2, markersize=6,
                label=label_a, zorder=3)
        ax.plot(x, vals_b, "s--", color=COLOR_B, linewidth=2.2, markersize=6,
                label=label_b, zorder=3)

        ax.fill_between(x, vals_a, vals_b, alpha=0.08, color="gray")

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_xlabel("M", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"{tag} {ylabel}", fontsize=11, fontweight="bold")
        ax.grid(alpha=0.25)

        if idx == 0:
            ax.legend(fontsize=7, loc="best")

        for i in [0, len(M_vals) - 1]:
            va_a, va_b = vals_a[i], vals_b[i]
            off_a = (6, 5) if va_a >= va_b else (6, -12)
            off_b = (6, -12) if va_a >= va_b else (6, 5)
            ax.annotate(f"{va_a:.1f}", xy=(x[i], va_a), xytext=off_a,
                        textcoords="offset points", fontsize=7,
                        color=COLOR_A, fontweight="bold")
            ax.annotate(f"{va_b:.1f}", xy=(x[i], va_b), xytext=off_b,
                        textcoords="offset points", fontsize=7,
                        color=COLOR_B, fontweight="bold")

    fig.suptitle(
        "Figure 4: Hardware Bottleneck Migration across Dynamic Shapes"
        "   (GEMM, N=2304, K=768)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    path = os.path.join(SCRIPT_DIR, "figure4_bottleneck_migration.pdf")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def print_summary_table(data):
    """Print a formatted table for paper inclusion."""
    config_names = list(data.keys())
    name_a, name_b = config_names

    print("\n" + "=" * 90)
    print("Table: Hardware Metrics Summary")
    print("=" * 90)
    header = (f"{'M':>6s} | {'Lat-A(us)':>10s} {'Lat-B(us)':>10s} {'A/B':>6s} | "
              f"{'Occ-A':>6s} {'Occ-B':>6s} | {'L2-A':>6s} {'L2-B':>6s} | "
              f"{'DRAM-A':>7s} {'DRAM-B':>7s} | {'Stall-A':>8s} {'Stall-B':>8s}")
    print(header)
    print("-" * 95)

    for i in range(len(data[name_a])):
        ra = data[name_a][i]["metrics"]
        rb = data[name_b][i]["metrics"]
        M = data[name_a][i]["M"]
        la, lb = ra.get("latency_us", 0), rb.get("latency_us", 0)
        ratio = la / lb if lb > 0 else 0
        print(f"{M:>6d} | {la:>10.2f} {lb:>10.2f} {ratio:>6.2f} | "
              f"{ra.get('occupancy_pct',0):>6.2f} {rb.get('occupancy_pct',0):>6.2f} | "
              f"{ra.get('l2_hit_rate_pct',0):>6.2f} {rb.get('l2_hit_rate_pct',0):>6.2f} | "
              f"{ra.get('dram_throughput_pct',0):>7.2f} {rb.get('dram_throughput_pct',0):>7.2f} | "
              f"{ra.get('stall_long_scoreboard',0):>8.1f} {rb.get('stall_long_scoreboard',0):>8.1f}")


def main():
    data = load_results()
    plot_bottleneck_migration(data)
    print_summary_table(data)


if __name__ == "__main__":
    main()
