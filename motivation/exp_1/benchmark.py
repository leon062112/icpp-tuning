"""
Experiment 1: Static Tuning Performance Loss Quantification

Goal: Show that the optimal autotuning config is shape-dependent,
      and static tuning (picking one config for all shapes) incurs
      significant performance loss on certain shapes.

Setup:
  - Kernel: GEMM  (A @ B,  fp16 → fp16, accumulate fp32)
  - Fixed:  N = 2304, K = 768  (LLM hidden → intermediate projection)
  - Dynamic: M ∈ {24, 43, 62, 81, 100, 128, 192, 256}
  - Config space: BLOCK_M × BLOCK_N × BLOCK_K × num_warps × num_stages
"""

import itertools
import json
import os
import sys
import time

import torch
import triton
import triton.language as tl

# Use a GPU with free memory (configurable via env CUDA_VISIBLE_DEVICES)
DEVICE = triton.runtime.driver.active.get_active_torch_device()
print(f"Using device: {DEVICE}")

# ─────────────────── Config space ───────────────────

BLOCK_MS = [32, 64, 128, 256]
BLOCK_NS = [32, 64, 128, 256]
BLOCK_KS = [32, 64]
NUM_WARPS_LIST = [2, 4, 8]
NUM_STAGES_LIST = [2, 3, 4, 5]
GROUP_SIZE_M = 8

# Shapes: fixed N, K; variable M
N, K = 2304, 768
M_VALUES = [24, 43, 62, 81, 100, 128, 192, 256]

# Benchmark parameters
WARMUP = 25
REP = 100


def generate_configs():
    """Generate all valid (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) configs."""
    configs = []
    for bm, bn, bk, nw, ns in itertools.product(
        BLOCK_MS, BLOCK_NS, BLOCK_KS, NUM_WARPS_LIST, NUM_STAGES_LIST
    ):
        # Filter out obviously bad configs
        # 1. Shared memory constraint: (bm*bk + bk*bn) * 2 bytes (fp16) * ns <= 48KB
        smem_bytes = (bm * bk + bk * bn) * 2 * ns
        if smem_bytes > 49152:
            continue
        # 2. Minimum threads: num_warps * 32 should be able to cover the tile
        threads = nw * 32
        # For Triton dot, need at least bm * bn / (threads) iterations per thread
        # Generally fine, but skip very large tiles with few warps
        if bm * bn > threads * 256:  # heuristic upper bound
            continue
        # 3. BLOCK_K should not exceed K
        if bk > K:
            continue
        configs.append({
            "BLOCK_SIZE_M": bm,
            "BLOCK_SIZE_N": bn,
            "BLOCK_SIZE_K": bk,
            "num_warps": nw,
            "num_stages": ns,
        })
    return configs


# ─────────────────── Triton kernel (no autotune) ───────────────────

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul_with_config(a, b, config):
    """Run GEMM with a specific config (no autotune)."""
    M, K_ = a.shape
    K_, N_ = b.shape
    c = torch.empty((M, N_), device=a.device, dtype=torch.float16)
    bm = config["BLOCK_SIZE_M"]
    bn = config["BLOCK_SIZE_N"]
    grid = (triton.cdiv(M, bm) * triton.cdiv(N_, bn),)
    matmul_kernel[grid](
        a, b, c,
        M, N_, K_,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=bm,
        BLOCK_SIZE_N=bn,
        BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return c


def verify_correctness(config, M_val=128):
    """Quick correctness check for a config."""
    a = torch.randn((M_val, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    c_triton = matmul_with_config(a, b, config)
    c_torch = torch.matmul(a, b)
    return torch.allclose(c_triton, c_torch, atol=1e-1, rtol=1e-2)


def benchmark_config(config, M_val):
    """Benchmark a single config on a single shape. Returns latency in ms."""
    a = torch.randn((M_val, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    try:
        ms = triton.testing.do_bench(
            lambda: matmul_with_config(a, b, config),
            warmup=WARMUP, rep=REP,
        )
        return ms
    except Exception as e:
        print(f"  [SKIP] config {config_to_str(config)} on M={M_val}: {e}")
        return None


def config_to_str(cfg):
    return (
        f"BM{cfg['BLOCK_SIZE_M']}_BN{cfg['BLOCK_SIZE_N']}_BK{cfg['BLOCK_SIZE_K']}"
        f"_w{cfg['num_warps']}_s{cfg['num_stages']}"
    )


def main():
    print("=" * 70)
    print("Experiment 1: Static Tuning Performance Loss Quantification")
    print("=" * 70)
    print(f"Shapes: M ∈ {M_VALUES},  N={N}, K={K}")

    # Generate config space
    configs = generate_configs()
    print(f"Total configs in search space: {len(configs)}")

    # Filter configs that are valid (pass correctness check)
    print("\nVerifying correctness of configs...")
    valid_configs = []
    for i, cfg in enumerate(configs):
        try:
            if verify_correctness(cfg):
                valid_configs.append(cfg)
            else:
                print(f"  [FAIL] config {config_to_str(cfg)} correctness check failed")
        except Exception as e:
            print(f"  [SKIP] config {config_to_str(cfg)}: {e}")
        if (i + 1) % 50 == 0:
            print(f"  Verified {i+1}/{len(configs)} configs, {len(valid_configs)} valid so far")
    print(f"Valid configs: {len(valid_configs)}")

    # Benchmark all valid configs x all shapes
    print(f"\nBenchmarking {len(valid_configs)} configs × {len(M_VALUES)} shapes "
          f"= {len(valid_configs) * len(M_VALUES)} runs ...")

    # results[M][config_str] = latency_ms
    results = {}
    total = len(valid_configs) * len(M_VALUES)
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
            if done % 20 == 0:
                print(f"  Progress: {done}/{total} ({100*done/total:.1f}%)")

    # Save raw results
    output = {
        "N": N,
        "K": K,
        "M_values": M_VALUES,
        "configs": [config_to_str(c) for c in valid_configs],
        "config_details": valid_configs,
        "results": {str(m): results[m] for m in M_VALUES},
    }
    out_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Quick summary
    print("\n" + "=" * 70)
    print("Quick Summary")
    print("=" * 70)
    for M_val in M_VALUES:
        if not results[M_val]:
            continue
        best_cfg = min(results[M_val], key=results[M_val].get)
        best_lat = results[M_val][best_cfg]
        print(f"  M={M_val:>4d}: best config = {best_cfg}, latency = {best_lat:.4f} ms")

    # Find static baseline: best config on largest shape (M=256)
    largest_M = max(M_VALUES)
    if results[largest_M]:
        static_cfg = min(results[largest_M], key=results[largest_M].get)
        print(f"\nStatic baseline (best on M={largest_M}): {static_cfg}")
        for M_val in M_VALUES:
            if static_cfg in results[M_val]:
                per_shape_best = min(results[M_val].values())
                static_lat = results[M_val][static_cfg]
                loss = (static_lat - per_shape_best) / per_shape_best * 100
                print(f"  M={M_val:>4d}: per-shape best = {per_shape_best:.4f} ms, "
                      f"static = {static_lat:.4f} ms, loss = {loss:.1f}%")


if __name__ == "__main__":
    main()
