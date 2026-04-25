"""Evaluate formula-selected TileLang GEMM (split-k) config against autotune oracle."""

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
from kernel.tilelang.gemm.gemm import DESCRIPTOR, get_configs, ref_program, build_kernel


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Compare formula-selected TileLang GEMM config against autotune oracle."
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
        m_values = [24, 64, 128, 256]

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

        kernel = build_kernel(M, args.N, args.K, config=formula_cfg)

        a = torch.randn(M, args.K, device="cuda", dtype=torch.float16)
        b = torch.randn(args.K, args.N, device="cuda", dtype=torch.float16)

        formula_latency = triton.testing.do_bench(
            lambda: kernel(a, b), warmup=100, rep=200,
        )
        torch_latency = triton.testing.do_bench(
            lambda: torch.mm(a.float(), b.float()), warmup=100, rep=200,
        )
        torch_speedup = torch_latency / formula_latency if formula_latency > 0 else 0.0

        print(f"  Formula kernel: {DESCRIPTOR.format_config(formula_cfg)} latency={formula_latency:.4f} ms")
        print(f"  Torch ref:      latency={torch_latency:.4f} ms")
        print(f"  Speedup vs torch: {torch_speedup:.4f}x")

        if args.compare_oracle:
            all_cfgs = get_configs(M, args.K)
            best_oracle_cfg = None
            best_oracle_latency = float("inf")
            for cfg in all_cfgs:
                k = build_kernel(M, args.N, args.K, config=cfg)
                lat = triton.testing.do_bench(lambda: k(a, b), warmup=50, rep=100)
                if lat < best_oracle_latency:
                    best_oracle_latency = lat
                    best_oracle_cfg = cfg
            oracle_latency = best_oracle_latency
            oracle_cfg = best_oracle_cfg
            perf_ratio = oracle_latency / formula_latency if formula_latency > 0 else 0.0
            print(f"  Oracle kernel:  {DESCRIPTOR.format_config(oracle_cfg)} latency={oracle_latency:.4f} ms")
            print(f"  Perf ratio (oracle/formula): {perf_ratio:.4f}")
        print()


if __name__ == "__main__":
    main()
