"""Test conv2d.py on a specific CUDA device."""
import os
import sys

# Set device before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import tilelang
import tilelang.language as T

from kernel.autotuner.interface import select_best
from kernel.tilelang.conv.conv2d import (
    DESCRIPTOR, build_conv2d_kernel, get_ref_program, calc_oh_ow, calc_conv_flops,
    verify_correctness
)

torch.backends.cuda.matmul.allow_tf32 = False


def do_bench_ms(fn, X, W_weight, Y, warmup=50, rep=100):
    for _ in range(warmup):
        fn(X, W_weight, Y)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn(X, W_weight, Y)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep


def test_conv2d():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Device ID (local): 0 (global: {os.environ['CUDA_VISIBLE_DEVICES']})")
    print()

    # Test a few representative shapes
    test_shapes = [
        # (N, C, H, W, F_out, KH, KW, S, P)
        (1, 3, 224, 224, 64, 7, 7, 2, 3),    # ResNet stem
        (1, 64, 56, 56, 64, 3, 3, 1, 1),     # ResNet bottleneck
        (1, 128, 28, 28, 256, 1, 1, 1, 0),   # 1x1 conv
        (4, 256, 14, 14, 512, 3, 3, 1, 1),   # Batch > 1
        (8, 512, 7, 7, 512, 3, 3, 1, 1),     # Deep layer
    ]

    print(f"{'Shape':>40} | {'Config':>30} | {'Latency_ms':>10} | {'TFLOPS':>8} | {'Correct':>8}")
    print("-" * 115)

    results = []
    for N, C, H, W, F_out, KH, KW, S, P in test_shapes:
        OH, OW = calc_oh_ow(H, W, KH, KW, S, P, 1)
        total_flops = calc_conv_flops(N, C, OH, OW, F_out, KH, KW)
        shape_str = f"N={N},C={C},H={H},W={W},F={F_out},K={KH}x{KW}"

        try:
            best_config, score = select_best(
                DESCRIPTOR, N=N, C=C, H=H, W=W, OC=F_out,
                KH=KH, KW=KW, stride=S, padding=P
            )
            kernel = build_conv2d_kernel(N, C, H, W, F_out, KH, KW, S, P, config=best_config)

            X = torch.randn(N, H, W, C, dtype=torch.float16, device="cuda")
            W_weight = torch.randn(KH, KW, C, F_out, dtype=torch.float16, device="cuda")
            Y = torch.zeros(N, OH, OW, F_out, dtype=torch.float16, device="cuda")

            lat_ms = do_bench_ms(kernel, X, W_weight, Y, warmup=50, rep=100)

            ref_prog = get_ref_program(S, P, 1, use_compile=False)
            is_correct = verify_correctness(kernel, ref_prog, N, C, H, W, F_out, KH, KW, S, P, 1)

            tflops = total_flops / lat_ms * 1e-9
            cfg_str = DESCRIPTOR.format_config(best_config)
            status = "PASS" if is_correct else "FAIL"

            print(f"{shape_str:>40} | {cfg_str:>30} | {lat_ms:>10.4f} | {tflops:>7.2f} | {status:>8}")

            results.append({
                "shape": shape_str,
                "config": cfg_str,
                "latency_ms": lat_ms,
                "tflops": tflops,
                "correct": is_correct,
                "error": None,
            })
        except Exception as e:
            print(f"{shape_str:>40} | {'ERROR':>30} | {'N/A':>10} | {'N/A':>7} | {'ERROR':>8}")
            results.append({
                "shape": shape_str,
                "config": None,
                "latency_ms": None,
                "tflops": None,
                "correct": False,
                "error": str(e),
            })

    print()
    return results


if __name__ == "__main__":
    results = test_conv2d()
    passed = sum(1 for r in results if r["correct"])
    failed = sum(1 for r in results if not r["correct"] and r["error"] is None)
    errors = sum(1 for r in results if r["error"] is not None)
    print(f"Summary: {passed} passed, {failed} failed, {errors} errors out of {len(results)} tests")
