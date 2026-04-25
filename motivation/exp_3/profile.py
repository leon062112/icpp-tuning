"""
Experiment 3 – NCU Profiling Orchestrator

Profiles 2 configs × 8 shapes = 16 runs to show hardware bottleneck
migration across dynamic shapes.

Config-A (large tile): BM128_BN128_BK32_w4_s3
Config-B (small tile): BM32_BN32_BK64_w4_s5
"""

import csv
import io
import json
import os
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NCU = "/opt/nvidia/nsight-compute/2024.3.2/ncu"
DRIVER = os.path.join(SCRIPT_DIR, "ncu_driver.py")

# ─────────────────── Configs (from benchmark Step 1) ───────────────────

CONFIG_A = {
    "name": "Config-A (BM128)",
    "bm": 128, "bn": 128, "bk": 32, "warps": 4, "stages": 3,
}
CONFIG_B = {
    "name": "Config-B (BM32)",
    "bm": 32, "bn": 32, "bk": 64, "warps": 4, "stages": 5,
}

CONFIGS = [CONFIG_A, CONFIG_B]
M_VALUES = [24, 64, 128, 256, 512, 1024, 2048, 4096]

# NCU sections & raw metrics (same as exp_2)
SECTIONS = [
    "SpeedOfLight",
    "MemoryWorkloadAnalysis",
    "Occupancy",
    "LaunchStats",
    "WarpStateStats",
]

RAW_METRICS = [
    "smsp__average_warp_latency_issue_stalled_not_selected",
    "smsp__average_warp_latency_issue_stalled_long_scoreboard",
    "sm__pipe_tensor_op_hmma_cycles_active",
    "sm__cycles_elapsed",
]


def build_ncu_cmd(cfg, M):
    section_flags = " ".join(f"--section {s}" for s in SECTIONS)
    metrics_flag = "--metrics " + ",".join(RAW_METRICS)
    return (
        f"CUDA_VISIBLE_DEVICES=1 {NCU} "
        f"--kernel-name regex:matmul_kernel "
        f"--launch-skip 1 --launch-count 1 "
        f"{section_flags} {metrics_flag} "
        f"--csv "
        f"python {DRIVER} "
        f"--bm {cfg['bm']} --bn {cfg['bn']} --bk {cfg['bk']} "
        f"--warps {cfg['warps']} --stages {cfg['stages']} --M {M}"
    )


def parse_ncu_csv(stdout):
    """Parse NCU 2024 CSV output into {metric_name: value} dict."""
    lines = stdout.strip().split("\n")
    csv_lines = [l for l in lines if l.strip() and not l.startswith("==")]
    if not csv_lines:
        return {}

    header_idx = None
    for i, line in enumerate(csv_lines):
        if '"Metric Name"' in line:
            header_idx = i
            break
    if header_idx is None:
        print("  WARNING: Could not find CSV header")
        return {}

    csv_text = "\n".join(csv_lines[header_idx:])
    reader = csv.DictReader(io.StringIO(csv_text))

    metrics = {}
    for row in reader:
        name = row.get("Metric Name", "").strip().strip('"')
        value_str = row.get("Metric Value", "").strip().strip('"').replace(",", "")
        if not name or not value_str:
            continue
        try:
            metrics[name] = float(value_str)
        except (ValueError, TypeError):
            metrics[name] = value_str
    return metrics


def derive_metrics(raw):
    """Build a clean metrics dict from raw NCU output."""
    m = {}
    m["occupancy_pct"] = raw.get("Achieved Occupancy", 0)
    m["theoretical_occupancy_pct"] = raw.get("Theoretical Occupancy", 0)
    m["l2_hit_rate_pct"] = raw.get("L2 Hit Rate", 0)
    m["dram_throughput_pct"] = raw.get("DRAM Throughput", 0)
    m["latency_us"] = raw.get("Duration", 0) / 1000.0  # ns → μs
    m["registers_per_thread"] = raw.get("Registers Per Thread", 0)
    m["smem_dynamic_bytes"] = raw.get("Dynamic Shared Memory Per Block", 0)
    m["mem_pipes_busy_pct"] = raw.get("Mem Pipes Busy", 0)
    m["stall_not_selected"] = raw.get(
        "smsp__average_warp_latency_issue_stalled_not_selected.ratio", 0)
    m["stall_long_scoreboard"] = raw.get(
        "smsp__average_warp_latency_issue_stalled_long_scoreboard.ratio", 0)
    hmma_avg = raw.get("sm__pipe_tensor_op_hmma_cycles_active.avg", 0)
    elapsed_avg = raw.get("sm__cycles_elapsed.avg", 0)
    m["tensor_pipe_util_pct"] = (hmma_avg / elapsed_avg * 100) if elapsed_avg > 0 else 0
    return m


def run_ncu(cfg, M):
    """Run NCU for one (config, M) combination and return parsed metrics."""
    cmd = build_ncu_cmd(cfg, M)
    print(f"  CMD: ...{cmd[-120:]}")
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        print(f"  ERROR: NCU exited with code {result.returncode}")
        print(f"  STDERR (last 300): {result.stderr[-300:]}")
        return None

    raw = parse_ncu_csv(result.stdout)
    if not raw:
        print(f"  WARNING: No metrics parsed")
        print(f"  STDOUT preview: {result.stdout[:300]}")
        return None

    print(f"  Parsed {len(raw)} raw metrics")
    for dk in ["Achieved Occupancy", "L2 Hit Rate", "DRAM Throughput", "Duration"]:
        if dk in raw:
            print(f"    {dk} = {raw[dk]}")
        else:
            print(f"    {dk} = MISSING")

    return derive_metrics(raw)


def main():
    print("=" * 60)
    print("Experiment 3: Hardware Bottleneck Migration – NCU Profiling")
    print("=" * 60)
    total_runs = len(CONFIGS) * len(M_VALUES)
    print(f"Total: {len(CONFIGS)} configs × {len(M_VALUES)} shapes = {total_runs} NCU runs\n")

    all_results = {}
    run_idx = 0

    for cfg in CONFIGS:
        cfg_name = cfg["name"]
        print(f"\n{'='*50}")
        print(f"Config: {cfg_name}")
        print(f"  BM={cfg['bm']}, BN={cfg['bn']}, BK={cfg['bk']}, "
              f"warps={cfg['warps']}, stages={cfg['stages']}")
        print(f"{'='*50}")

        all_results[cfg_name] = []

        for M in M_VALUES:
            run_idx += 1
            print(f"\n[{run_idx}/{total_runs}] {cfg_name}, M={M}")
            metrics = run_ncu(cfg, M)
            if metrics is None:
                metrics = {}
            all_results[cfg_name].append({
                "M": M,
                "metrics": metrics,
            })

    # Save
    out_path = os.path.join(SCRIPT_DIR, "ncu_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for cfg_name, results in all_results.items():
        print(f"\n{cfg_name}:")
        print(f"  {'M':>6s}  {'Lat(us)':>8s}  {'Occ%':>6s}  {'L2Hit%':>7s}  "
              f"{'DRAM%':>6s}  {'Tensor%':>8s}")
        for r in results:
            m = r["metrics"]
            print(f"  {r['M']:>6d}  {m.get('latency_us',0):>8.2f}  "
                  f"{m.get('occupancy_pct',0):>6.2f}  {m.get('l2_hit_rate_pct',0):>7.2f}  "
                  f"{m.get('dram_throughput_pct',0):>6.2f}  "
                  f"{m.get('tensor_pipe_util_pct',0):>8.2f}")


if __name__ == "__main__":
    main()
