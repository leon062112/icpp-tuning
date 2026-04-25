"""Evaluate formula-selected TileLang GEMM+Bias+ReLU config against autotune oracle."""

import argparse
import csv
from pathlib import Path
from typing import List

import torch
import triton

import sys

# Add project root to path so 'kernel' package is importable
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from kernel.autotuner.gpu_spec import get_gpu_spec
from kernel.autotuner.interface import configure_autotuner_cache, rank_all
from kernel.tilelang.gemm.gemm_bias_act import DESCRIPTOR, get_configs, ref_program, build_kernel


# ---------------------------------------------------------------------------
# Detail row / CSV helpers
# ---------------------------------------------------------------------------


def _detail_row(
    M: int, N: int, K: int,
    label: str, rank, entry: dict,
    latency_ms=None,
) -> dict:
    detail = entry.get("detail", {})
    row = {
        "M": M, "N": N, "K": K,
        "label": label,
        "rank": rank if rank is not None else "",
        "config": entry["formatted"],
        "score": entry["score"],
        "latency_ms": latency_ms if latency_ms is not None else "",
        "eff_mainloop": detail.get("eff_mainloop", ""),
        "eff_memory": detail.get("eff_memory", ""),
        "eff_parallel": detail.get("eff_parallel", ""),
        "eff_pipeline": detail.get("eff_pipeline", ""),
        "eff_epilogue": detail.get("eff_epilogue", ""),
        "eff_reduction": detail.get("eff_reduction", ""),
    }
    for key, value in detail.get("projection", {}).items():
        row[f"proj_{key}"] = value
    return row


def _write_detail_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_detail_comparison(rows: List[dict]) -> None:
    print("  Detailed score comparison:")
    print(
        "    "
        f"{'label':<12} {'rank':>4} {'cfg':<24} {'score':>9} "
        f"{'emain':>7} {'emem':>7} {'epar':>7} {'epipe':>7} {'eepi':>7} {'lat(ms)':>8}"
    )
    for row in rows:
        print(
            "    "
            f"{row['label']:<12} {str(row['rank']):>4} {row['config']:<24} "
            f"{row['score']:>9.6f} {row['eff_mainloop']:>7.4f} {row['eff_memory']:>7.4f} "
            f"{row['eff_parallel']:>7.4f} {row['eff_pipeline']:>7.4f} {row['eff_epilogue']:>7.4f} "
            f"{row['latency_ms'] if row['latency_ms'] != '' else '':>8}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Compare formula-selected TileLang GEMM+Bias+ReLU config against autotune oracle."
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
    parser.add_argument("--export-detailed-csv", type=str, default=None)
    parser.add_argument("--detail-top-k", type=int, default=0)
    parser.add_argument("--print-detailed-table", action="store_true")
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

    need_detail = args.export_detailed_csv or args.print_detailed_table
    detail_rows: List[dict] = []

    for M in m_values:
        ranked = rank_all(DESCRIPTOR, hw=hw, detailed=need_detail, M=M, N=args.N, K=args.K)
        best_entry = ranked[0]
        formula_cfg = best_entry["raw_config"]
        per_m_rows: List[dict] = []
        print(f"M={M}, N={args.N}, K={args.K}")
        print(f"  Formula top-1: {best_entry['formatted']} score={best_entry['score']:.6f}")

        if need_detail:
            top_k = max(1, min(args.detail_top_k, len(ranked))) if args.detail_top_k > 0 else 1
            for rank_idx, entry in enumerate(ranked[:top_k], start=1):
                row = _detail_row(M, args.N, args.K, f"ranked_{rank_idx}", rank_idx, entry)
                detail_rows.append(row)
                per_m_rows.append(row)

        if not torch.cuda.is_available():
            print("  CUDA unavailable: skip runtime benchmark")
            if args.print_detailed_table and per_m_rows:
                _print_detail_comparison(per_m_rows)
            print()
            continue

        kernel = build_kernel(M, args.N, args.K, config=formula_cfg)

        a = torch.randn(M, args.K, device="cuda", dtype=torch.float16)
        b = torch.randn(args.K, args.N, device="cuda", dtype=torch.float16)
        bias = torch.randn(args.N, device="cuda", dtype=torch.float16)

        formula_latency = triton.testing.do_bench(
            lambda: kernel(a, b, bias), warmup=100, rep=200,
        )
        torch_latency = triton.testing.do_bench(
            lambda: ref_program(a, b, bias), warmup=100, rep=200,
        )
        torch_speedup = torch_latency / formula_latency if formula_latency > 0 else 0.0

        print(f"  Formula kernel: {DESCRIPTOR.format_config(formula_cfg)} latency={formula_latency:.4f} ms")
        print(f"  Torch ref:      latency={torch_latency:.4f} ms")
        print(f"  Speedup vs torch: {torch_speedup:.4f}x")

        if need_detail:
            formula_row = _detail_row(
                M, args.N, args.K, "formula_runtime", 1,
                best_entry, latency_ms=formula_latency,
            )
            detail_rows.append(formula_row)
            per_m_rows.append(formula_row)

        if args.compare_oracle:
            all_cfgs = get_configs(M, args.K)
            best_oracle_cfg = None
            best_oracle_latency = float("inf")
            for oracle_cfg in all_cfgs:
                oracle_kernel = build_kernel(M, args.N, args.K, config=oracle_cfg)
                lat = triton.testing.do_bench(lambda: oracle_kernel(a, b, bias), warmup=50, rep=100)
                if lat < best_oracle_latency:
                    best_oracle_latency = lat
                    best_oracle_cfg = oracle_cfg
            oracle_latency = best_oracle_latency
            oracle_cfg = best_oracle_cfg
            perf_ratio = oracle_latency / formula_latency if formula_latency > 0 else 0.0
            print(f"  Oracle kernel:  {DESCRIPTOR.format_config(oracle_cfg)} latency={oracle_latency:.4f} ms")
            print(f"  Perf ratio (oracle/formula): {perf_ratio:.4f}")
            if need_detail:
                oracle_rank = next(
                    (idx for idx, e in enumerate(ranked, start=1)
                     if e["raw_config"] == oracle_cfg),
                    None,
                )
                oracle_entry = next(
                    (e for e in ranked if e["raw_config"] == oracle_cfg),
                    best_entry,
                )
                oracle_row = _detail_row(
                    M, args.N, args.K, "oracle_runtime", oracle_rank,
                    oracle_entry, latency_ms=oracle_latency,
                )
                detail_rows.append(oracle_row)
                per_m_rows.append(oracle_row)

        if args.print_detailed_table and per_m_rows:
            _print_detail_comparison(per_m_rows)
        print()

    if args.export_detailed_csv:
        _write_detail_csv(args.export_detailed_csv, detail_rows)
        print(f"Detailed rows written to {Path(args.export_detailed_csv).expanduser().resolve()}")


if __name__ == "__main__":
    main()
