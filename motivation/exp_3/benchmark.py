"""
Experiment 3 – Step 1: Extended-M Benchmark

Goal: Benchmark all valid configs across an expanded M range (24..4096)
      to find per-shape best configs and select Config-A / Config-B
      for NCU bottleneck-migration profiling.

Setup:
  - Kernel: GEMM (A @ B, fp16 → fp16, accumulate fp32)
  - Fixed:  N = 2304, K = 768
  - Dynamic: M ∈ {24, 64, 128, 256, 512, 1024, 2048, 4096}
  - Config space: reuse exp_1 (BLOCK_M × BLOCK_N × BLOCK_K × num_warps × num_stages)
"""

import json
import os
import sys

# Import kernel & utilities from exp_1
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exp_1"))
from benchmark import (
    generate_configs, verify_correctness, benchmark_config,
    config_to_str, N, K, DEVICE,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Extended M range
M_VALUES = [24, 64, 128, 256, 512, 1024, 2048, 4096]


def select_configs(results, M_values, config_names):
    """Recommend Config-A (large tile) and Config-B (small tile) from results."""
    common = set(config_names)
    for m in M_values:
        common &= set(results[str(m)].keys())
    common = sorted(common)

    if not common:
        print("ERROR: No common configs across all shapes!")
        return None, None

    largest_M = max(M_values)
    smallest_M = min(M_values)

    # Config-A: best on largest M, prefer large BLOCK_M
    lats_large = {c: results[str(largest_M)][c] for c in common}
    # Among top-5 on largest M, pick the one with largest BLOCK_M
    top5_large = sorted(lats_large, key=lats_large.get)[:5]
    config_a = max(top5_large, key=lambda c: int(c.split("_")[0].replace("BM", "")))

    # Config-B: best on smallest M, prefer small BLOCK_M
    lats_small = {c: results[str(smallest_M)][c] for c in common}
    top5_small = sorted(lats_small, key=lats_small.get)[:5]
    config_b = min(top5_small, key=lambda c: int(c.split("_")[0].replace("BM", "")))

    # Ensure they are different; if same, pick next best
    if config_a == config_b:
        for c in sorted(lats_large, key=lats_large.get):
            bm = int(c.split("_")[0].replace("BM", ""))
            if bm >= 128 and c != config_b:
                config_a = c
                break

    return config_a, config_b


def main():
    print("=" * 70)
    print("Experiment 3 – Step 1: Extended-M Benchmark")
    print("=" * 70)
    print(f"Shapes: M ∈ {M_VALUES},  N={N}, K={K}")

    # Generate & validate configs (reuse exp_1 logic)
    configs = generate_configs()
    print(f"Total configs in search space: {len(configs)}")

    print("\nVerifying correctness of configs...")
    valid_configs = []
    for i, cfg in enumerate(configs):
        try:
            if verify_correctness(cfg):
                valid_configs.append(cfg)
        except Exception:
            pass
        if (i + 1) % 50 == 0:
            print(f"  Verified {i+1}/{len(configs)}, {len(valid_configs)} valid")
    print(f"Valid configs: {len(valid_configs)}")

    # Benchmark
    total = len(valid_configs) * len(M_VALUES)
    print(f"\nBenchmarking {len(valid_configs)} configs × {len(M_VALUES)} shapes = {total} runs ...")

    results = {}
    done = 0
    for M_val in M_VALUES:
        results[M_val] = {}
        print(f"\n--- M = {M_val} ---")
        for cfg in valid_configs:
            cstr = config_to_str(cfg)
            lat = benchmark_config(cfg, M_val)
            if lat is not None:
                results[M_val][cstr] = lat
            done += 1
            if done % 50 == 0:
                print(f"  Progress: {done}/{total} ({100*done/total:.1f}%)")

    # Save
    config_names = [config_to_str(c) for c in valid_configs]
    output = {
        "N": N, "K": K,
        "M_values": M_VALUES,
        "configs": config_names,
        "config_details": valid_configs,
        "results": {str(m): results[m] for m in M_VALUES},
    }
    out_path = os.path.join(SCRIPT_DIR, "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    print("\n" + "=" * 70)
    print("Per-shape Best Configs")
    print("=" * 70)
    for M_val in M_VALUES:
        if not results[M_val]:
            continue
        best_cfg = min(results[M_val], key=results[M_val].get)
        best_lat = results[M_val][best_cfg]
        print(f"  M={M_val:>5d}: best = {best_cfg}, latency = {best_lat:.4f} ms")

    # Recommend Config-A / Config-B
    print("\n" + "=" * 70)
    print("Config Selection for NCU Profiling")
    print("=" * 70)
    str_results = {str(m): results[m] for m in M_VALUES}
    config_a, config_b = select_configs(str_results, M_VALUES, config_names)
    if config_a and config_b:
        print(f"  Config-A (large tile): {config_a}")
        print(f"  Config-B (small tile): {config_b}")
        # Cross performance
        print("\n  Cross-shape performance comparison:")
        print(f"  {'M':>6s}  {'Config-A (ms)':>14s}  {'Config-B (ms)':>14s}  {'A/B ratio':>10s}")
        for M_val in M_VALUES:
            la = results[M_val].get(config_a, float("inf"))
            lb = results[M_val].get(config_b, float("inf"))
            ratio = la / lb if lb > 0 else float("inf")
            print(f"  {M_val:>6d}  {la:>14.4f}  {lb:>14.4f}  {ratio:>10.2f}")


if __name__ == "__main__":
    main()
