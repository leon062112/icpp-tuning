"""Evaluate formula-selected TileLang GEMM+Bias+LayerNorm config against autotune oracle."""

import argparse

import torch
import triton

import sys
from pathlib import Path

# Add project root to path so 'kernel' package is importable
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from kernel.autotuner.gpu_spec import get_gpu_spec
from kernel.autotuner.interface import configure_autotuner_cache, rank_all
from kernel.tilelang.gemm.gemm_bias_layernorm import (
    DESCRIPTOR,
    build_fused_kernel,
    get_gemm_configs,
    torch_baseline,
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Compare formula-selected TileLang GEMM+Bias+LayerNorm config against autotune oracle."
    )
    parser.add_argument("--N", type=int, default=2304)
    parser.add_argument("--K", type=int, default=768)
    parser.add_argument("--M-values", type=int, nargs="+", default=None)
    parser.add_argument("--M-start", type=int, default=None)
    parser.add_argument("--M-end", type=int, default=None)
    parser.add_argument("--M-step", type=int, default=1)
    parser.add_argument("--cache-dir", type=str, default=".tilelang_cache")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--compare-oracle", action="store_true")
    args = parser.parse_args()

    configure_autotuner_cache(args.cache_dir)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda_device)

    hw = get_gpu_spec()
    print(f"Using HW spec: {hw.name}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print()

    if args.M_values:
        m_values = args.M_values
    elif args.M_start is not None and args.M_end is not None:
        m_values = list(range(args.M_start, args.M_end + 1, args.M_step))
    else:
        m_values = [64, 128, 256]

    for M in m_values:
        ranked = rank_all(DESCRIPTOR, hw=hw, M=M, N=args.N, K=args.K)
        best_entry = ranked[0]
        formula_cfg = best_entry["raw_config"]
        print(f"M={M}, N={args.N}, K={args.K}")
        print(f"  Formula top-1: {best_entry['formatted']} score={best_entry['score']:.6f}")

        if not torch.cuda.is_available():
            print("  CUDA unavailable: skip runtime benchmark")
            print()
            continue

        a = torch.randn(M, args.K, device="cuda", dtype=torch.float16)
        b = torch.randn(args.K, args.N, device="cuda", dtype=torch.float16)
        bias = torch.randn(args.N, device="cuda", dtype=torch.float16)
        gamma = torch.randn(args.N, device="cuda", dtype=torch.float16)
        beta = torch.randn(args.N, device="cuda", dtype=torch.float16)

        formula_kernel = build_fused_kernel(M, args.N, args.K, gemm_config=formula_cfg)

        formula_latency = triton.testing.do_bench(
            lambda: formula_kernel(a, b, bias, gamma, beta), warmup=100, rep=200,
        )
        torch_latency = triton.testing.do_bench(
            lambda: torch_baseline(a, b, bias, gamma, beta), warmup=100, rep=200,
        )
        torch_speedup = torch_latency / formula_latency if formula_latency > 0 else 0.0

        print(f"  Formula kernel: {best_entry['formatted']} latency={formula_latency:.4f} ms")
        print(f"  Torch ref:      latency={torch_latency:.4f} ms")
        print(f"  Speedup vs torch: {torch_speedup:.4f}x")

        if args.compare_oracle:
            best_oracle_cfg = None
            best_oracle_latency = float("inf")
            all_cfgs = get_gemm_configs(args.K)
            for oracle_cfg in all_cfgs:
                oracle_kernel = build_fused_kernel(M, args.N, args.K, gemm_config=oracle_cfg)
                oracle_latency = triton.testing.do_bench(
                    lambda: oracle_kernel(a, b, bias, gamma, beta), warmup=50, rep=100,
                )
                if oracle_latency < best_oracle_latency:
                    best_oracle_latency = oracle_latency
                    best_oracle_cfg = oracle_cfg

            perf_ratio = best_oracle_latency / formula_latency if formula_latency > 0 else 0.0
            print(f"  Oracle kernel:  {DESCRIPTOR.format_config(best_oracle_cfg)} latency={best_oracle_latency:.4f} ms")
            print(f"  Perf ratio (oracle/formula): {perf_ratio:.4f}")
        print()


if __name__ == "__main__":
    main()
