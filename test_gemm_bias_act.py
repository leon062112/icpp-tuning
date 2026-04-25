"""Test gemm_bias_act.py on a specific CUDA device."""
import os
import sys

# Set device before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import tilelang
import tilelang.language as T
import triton

from kernel.autotuner.interface import select_best
from kernel.tilelang.gemm.gemm_bias_act import DESCRIPTOR, build_kernel, ref_program

torch.backends.cuda.matmul.allow_tf32 = False

def test_gemm_bias_act():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Device ID (local): 0 (global: {os.environ['CUDA_VISIBLE_DEVICES']})")
    print()

    # Test a few representative shapes
    test_shapes = [
        (1, 2304, 768),
        (16, 2304, 768),
        (64, 2304, 768),
        (128, 4096, 4096),
        (256, 1024, 1024),
    ]

    print(f"{'M':>5} | {'N':>5} | {'K':>5} | {'Config':>30} | {'Latency_ms':>10} | {'TFLOPS':>8} | {'Correct':>8}")
    print("-" * 95)

    results = []
    for M, N, K in test_shapes:
        total_flops = 2 * M * N * K

        try:
            best_config, score = select_best(DESCRIPTOR, M=M, N=N, K=K)
            kernel = build_kernel(M, N, K, config=best_config)

            a = torch.randn(M, K, device="cuda", dtype=torch.float16)
            b = torch.randn(K, N, device="cuda", dtype=torch.float16)
            bias = torch.randn(N, device="cuda", dtype=torch.float16)

            tl_lat = triton.testing.do_bench(
                lambda: kernel(a, b, bias),
                warmup=50, rep=100,
            )

            c_tl = kernel(a, b, bias)
            c_ref = ref_program(a, b, bias)
            ok = torch.allclose(c_tl.float(), c_ref.float(), rtol=1e-2, atol=1e-2)

            tl_tflops = total_flops / tl_lat * 1e-9
            cfg_str = DESCRIPTOR.format_config(best_config)
            status = "PASS" if ok else "FAIL"

            print(f"{M:>5} | {N:>5} | {K:>5} | {cfg_str:>30} | {tl_lat:>10.4f} | {tl_tflops:>7.2f} | {status:>8}")

            results.append({
                "M": M, "N": N, "K": K,
                "config": cfg_str,
                "latency_ms": tl_lat,
                "tflops": tl_tflops,
                "correct": ok,
                "error": None,
            })
        except Exception as e:
            print(f"{M:>5} | {N:>5} | {K:>5} | {'ERROR':>30} | {'N/A':>10} | {'N/A':>7} | {'ERROR':>8}")
            results.append({
                "M": M, "N": N, "K": K,
                "config": None,
                "latency_ms": None,
                "tflops": None,
                "correct": False,
                "error": str(e),
            })

    print()
    return results


if __name__ == "__main__":
    results = test_gemm_bias_act()
    # Print summary
    passed = sum(1 for r in results if r["correct"])
    failed = sum(1 for r in results if not r["correct"] and r["error"] is None)
    errors = sum(1 for r in results if r["error"] is not None)
    print(f"Summary: {passed} passed, {failed} failed, {errors} errors out of {len(results)} tests")
