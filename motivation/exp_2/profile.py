"""
Experiment 2: NCU Profiling Orchestrator

Runs NCU on 8 unique configs (10 logical configs, deduplicated),
parses CSV output, and saves structured results to ncu_results.json.
"""

import csv
import io
import json
import os
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NCU = "/opt/nvidia/nsight-compute/2024.3.2/ncu"
DRIVER = os.path.join(SCRIPT_DIR, "ncu_driver.py")

# ─────────────────── Experiment Configs ───────────────────

CONFIGS = {
    "exp2a": [
        {"label": "BM=32",  "bm": 32,  "bn": 128, "bk": 32, "warps": 4, "stages": 3},
        {"label": "BM=64",  "bm": 64,  "bn": 128, "bk": 32, "warps": 4, "stages": 3},
        {"label": "BM=128", "bm": 128, "bn": 128, "bk": 32, "warps": 4, "stages": 3},
    ],
    "exp2b": [
        {"label": "warps=2", "bm": 64, "bn": 128, "bk": 32, "warps": 2, "stages": 3},
        {"label": "warps=4", "bm": 64, "bn": 128, "bk": 32, "warps": 4, "stages": 3},
        {"label": "warps=8", "bm": 64, "bn": 128, "bk": 32, "warps": 8, "stages": 3},
    ],
    "exp2c": [
        {"label": "stages=2", "bm": 64, "bn": 128, "bk": 32, "warps": 4, "stages": 2},
        {"label": "stages=3", "bm": 64, "bn": 128, "bk": 32, "warps": 4, "stages": 3},
        {"label": "stages=4", "bm": 64, "bn": 128, "bk": 32, "warps": 4, "stages": 4},
        {"label": "stages=5", "bm": 64, "bn": 128, "bk": 32, "warps": 4, "stages": 5},
    ],
}

SECTIONS = [
    "SpeedOfLight",
    "MemoryWorkloadAnalysis",
    "Occupancy",
    "LaunchStats",
    "WarpStateStats",
]

# Additional raw metrics not available via section display names
RAW_METRICS = [
    "smsp__average_warp_latency_issue_stalled_not_selected",
    "smsp__average_warp_latency_issue_stalled_long_scoreboard",
    "sm__pipe_tensor_op_hmma_cycles_active",
    "sm__cycles_elapsed",
]

# Display-name keys we extract from section CSV output
# (NCU 2024 CSV uses display names like "Achieved Occupancy", not raw metric IDs)
DISPLAY_KEYS = [
    "Achieved Occupancy",       # % (from Occupancy section)
    "Theoretical Occupancy",    # % (from Occupancy section)
    "L2 Hit Rate",              # % (from MemoryWorkloadAnalysis)
    "DRAM Throughput",          # % (from SpeedOfLight)
    "Duration",                 # ns (from SpeedOfLight)
    "Registers Per Thread",     # register/thread (from LaunchStats)
    "Driver Shared Memory Per Block",  # byte/block (from LaunchStats)
    "Dynamic Shared Memory Per Block", # byte/block (from LaunchStats)
    "Block Limit Registers",    # block (from Occupancy)
    "Block Limit Shared Mem",   # block (from Occupancy)
    "Waves Per SM",             # (from LaunchStats)
    "Mem Pipes Busy",           # % (from MemoryWorkloadAnalysis)
]


def config_key(cfg):
    return (cfg["bm"], cfg["bn"], cfg["bk"], cfg["warps"], cfg["stages"])


def build_ncu_cmd(cfg):
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
        f"--warps {cfg['warps']} --stages {cfg['stages']}"
    )


def parse_ncu_csv(stdout):
    """Parse NCU 2024 CSV output into {metric_name: value} dict.

    NCU 2024 CSV format (without --page raw):
      Each row has "Section Name", "Metric Name", "Metric Unit", "Metric Value".
      Section metrics use display names (e.g. "Achieved Occupancy").
      --metrics counters appear under "Command line profiler metrics" with raw names
      and sub-metrics like .avg, .max, .min, .sum, .ratio, .pct.
    """
    lines = stdout.strip().split("\n")
    csv_lines = [l for l in lines if l.strip() and not l.startswith("==")]
    if not csv_lines:
        return {}

    # Find header
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

    # From section display names
    m["occupancy_pct"] = raw.get("Achieved Occupancy", 0)
    m["theoretical_occupancy_pct"] = raw.get("Theoretical Occupancy", 0)
    m["l2_hit_rate_pct"] = raw.get("L2 Hit Rate", 0)
    m["dram_throughput_pct"] = raw.get("DRAM Throughput", 0)
    m["latency_us"] = raw.get("Duration", 0) / 1000.0  # ns → μs
    m["registers_per_thread"] = raw.get("Registers Per Thread", 0)
    m["smem_dynamic_bytes"] = raw.get("Dynamic Shared Memory Per Block", 0)
    m["mem_pipes_busy_pct"] = raw.get("Mem Pipes Busy", 0)

    # From raw metrics (--metrics flag)
    # Stall: Not Selected (inst/warp ratio)
    m["stall_not_selected"] = raw.get(
        "smsp__average_warp_latency_issue_stalled_not_selected.ratio", 0)
    # Stall: Long Scoreboard / Memory Dependency (inst/warp ratio)
    m["stall_long_scoreboard"] = raw.get(
        "smsp__average_warp_latency_issue_stalled_long_scoreboard.ratio", 0)

    # Tensor pipe utilization (%)
    hmma_avg = raw.get("sm__pipe_tensor_op_hmma_cycles_active.avg", 0)
    elapsed_avg = raw.get("sm__cycles_elapsed.avg", 0)
    if elapsed_avg > 0:
        m["tensor_pipe_util_pct"] = hmma_avg / elapsed_avg * 100
    else:
        m["tensor_pipe_util_pct"] = 0

    return m


def run_ncu(cfg):
    """Run NCU for one config and return parsed metrics dict."""
    cmd = build_ncu_cmd(cfg)
    print(f"  CMD: ...{cmd[-100:]}")
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

    # Validate key display names
    for dk in ["Achieved Occupancy", "L2 Hit Rate", "DRAM Throughput", "Duration"]:
        if dk in raw:
            print(f"    {dk} = {raw[dk]}")
        else:
            print(f"    {dk} = MISSING")

    return derive_metrics(raw)


def main():
    print("=" * 60)
    print("Experiment 2: NCU Profiling")
    print("=" * 60)

    # Deduplicate configs
    seen = {}
    all_results = {}

    unique_keys = set()
    for configs in CONFIGS.values():
        for cfg in configs:
            unique_keys.add(config_key(cfg))
    print(f"Total: 10 logical configs, {len(unique_keys)} unique NCU runs\n")

    run_idx = 0
    for exp_name, configs in CONFIGS.items():
        print(f"\n--- {exp_name} ---")
        all_results[exp_name] = []

        for cfg in configs:
            key = config_key(cfg)
            if key not in seen:
                run_idx += 1
                print(f"\n[{run_idx}/{len(unique_keys)}] Profiling {cfg['label']} "
                      f"(BM={cfg['bm']}, BN={cfg['bn']}, BK={cfg['bk']}, "
                      f"w={cfg['warps']}, s={cfg['stages']})")
                seen[key] = run_ncu(cfg)
                if seen[key] is None:
                    seen[key] = {}
            else:
                print(f"  Reusing cached result for {cfg['label']}")

            all_results[exp_name].append({
                "label": cfg["label"],
                "config": cfg,
                "metrics": seen[key],
            })

    # Save
    out_path = os.path.join(SCRIPT_DIR, "ncu_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary table
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for exp_name, results in all_results.items():
        print(f"\n{exp_name}:")
        print(f"  {'Label':>12s}  {'Lat(us)':>8s}  {'Occ%':>6s}  {'L2Hit%':>7s}  "
              f"{'DRAM%':>6s}  {'Tensor%':>8s}  {'StallNS':>8s}  {'StallMem':>9s}")
        for r in results:
            m = r["metrics"]
            print(f"  {r['label']:>12s}  {m.get('latency_us',0):>8.2f}  "
                  f"{m.get('occupancy_pct',0):>6.2f}  {m.get('l2_hit_rate_pct',0):>7.2f}  "
                  f"{m.get('dram_throughput_pct',0):>6.2f}  "
                  f"{m.get('tensor_pipe_util_pct',0):>8.2f}  "
                  f"{m.get('stall_not_selected',0):>8.2f}  "
                  f"{m.get('stall_long_scoreboard',0):>9.2f}")


if __name__ == "__main__":
    main()
