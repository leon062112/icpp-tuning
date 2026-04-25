import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T

import sys

# Add project root to path so 'kernel' package is importable
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from kernel.autotuner.interface import TileLangKernelBase, select_best
from kernel.autotuner.cost_model import make_conv2d_spec

try:
    from conv2d_bn_relu import get_conv_op_shape_list, get_configs, get_sampled_conv_shapes
except ImportError:
    from conv_bn_relu import get_conv_op_shape_list, get_configs, get_sampled_conv_shapes


torch.backends.cuda.matmul.allow_tf32 = False


def get_ref_program(stride=1, padding=1, dilation=1):
    """Reference implementation: Conv2d + ReLU."""

    def conv_relu_core(X_nchw, W_oihw):
        Y_nchw = F.conv2d(X_nchw, W_oihw, stride=stride, padding=padding, dilation=dilation).to(torch.float32)
        Y_nchw.relu_()
        return Y_nchw

    compiled_conv_relu_core = None
    if hasattr(torch, "compile"):
        try:
            compiled_conv_relu_core = torch.compile(conv_relu_core)
        except Exception:
            compiled_conv_relu_core = None

    def ref_program(X, W, Y):
        nonlocal compiled_conv_relu_core
        X_nchw = X.permute(0, 3, 1, 2)
        W_oihw = W.permute(3, 2, 0, 1)

        if compiled_conv_relu_core is not None:
            try:
                Y_nchw = compiled_conv_relu_core(X_nchw, W_oihw)
            except Exception:
                compiled_conv_relu_core = None
                Y_nchw = conv_relu_core(X_nchw, W_oihw)
        else:
            Y_nchw = conv_relu_core(X_nchw, W_oihw)

        Y.copy_(Y_nchw.permute(0, 2, 3, 1).to(Y.dtype))

    return ref_program


def _conv2d_relu_kernel_factory(N, C, H, W, F_out, KH, KW, S=1, P=1, D=1):
    """Return a Conv2d+ReLU kernel factory."""
    OH = (H + 2 * P - D * (KH - 1) - 1) // S + 1
    OW = (W + 2 * P - D * (KW - 1) - 1) // S + 1

    M_eff = N * OH * OW
    K_eff = KH * KW * C

    def kernel(block_M=None, block_N=None, block_K=None, num_stages=None, thread_num=None):
        dtype = T.float16
        accum_dtype = T.float32
        out_dtype = T.float16

        @T.prim_func
        def conv_relu_kernel(
            data: T.Tensor((N, H, W, C), dtype),
            kernel_weight: T.Tensor((KH, KW, C, F_out), dtype),
            out: T.Tensor((N, OH, OW, F_out), out_dtype),
        ):
            with T.Kernel(T.ceildiv(F_out, block_N), T.ceildiv(M_eff, block_M), threads=thread_num) as (bx, by):
                data_shared = T.alloc_shared((block_M, block_K), dtype)
                kernel_shared = T.alloc_shared((block_K, block_N), dtype)
                out_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                out_shared = T.alloc_shared((block_M, block_N), out_dtype)

                kernel_flat = T.Tensor((K_eff, F_out), dtype, kernel_weight.data)
                out_flat = T.Tensor((M_eff, F_out), out_dtype, out.data)

                T.clear(out_local)

                for k_iter in T.Pipelined(T.ceildiv(K_eff, block_K), num_stages=num_stages):
                    for i, j in T.Parallel(block_M, block_K):
                        k = k_iter * block_K + j
                        m = by * block_M + i

                        access_h = m % (OH * OW) // OW * S + k // (KW * C) * D - P
                        access_w = m % OW * S + k // C % KW * D - P

                        safe_h = T.min(T.max(access_h, 0), H - 1)
                        safe_w = T.min(T.max(access_w, 0), W - 1)

                        in_bound = (
                            (access_h >= 0)
                            and (access_w >= 0)
                            and (access_h < H)
                            and (access_w < W)
                            and (m < M_eff)
                            and (k < K_eff)
                        )

                        if in_bound:
                            data_shared[i, j] = data[m // (OH * OW), safe_h, safe_w, k % C]
                        else:
                            data_shared[i, j] = T.cast(0, dtype)

                    if K_eff % block_K == 0 and F_out % block_N == 0:
                        T.copy(kernel_flat[k_iter * block_K, bx * block_N], kernel_shared)
                    else:
                        for i, j in T.Parallel(block_K, block_N):
                            k_idx = k_iter * block_K + i
                            n_idx = bx * block_N + j
                            if k_idx < K_eff and n_idx < F_out:
                                kernel_shared[i, j] = kernel_flat[k_idx, n_idx]
                            else:
                                kernel_shared[i, j] = T.cast(0, dtype)

                    T.gemm(data_shared, kernel_shared, out_local)

                T.copy(out_local, out_shared)

                for i, j in T.Parallel(block_M, block_N):
                    m_idx = by * block_M + i
                    n_idx = bx * block_N + j
                    if m_idx < M_eff and n_idx < F_out:
                        x = T.cast(out_shared[i, j], accum_dtype)
                        relu = T.max(x, T.cast(0, accum_dtype))
                        out_flat[m_idx, n_idx] = T.cast(relu, out_dtype)

        return conv_relu_kernel

    return kernel


def build_conv2d_relu_kernel(N, C, H, W, F_out, KH, KW, S=1, P=1, D=1, config=None):
    """Build a compiled Conv2d+ReLU kernel using cost-model selection or a given config."""
    OH = (H + 2 * P - D * (KH - 1) - 1) // S + 1
    OW = (W + 2 * P - D * (KW - 1) - 1) // S + 1
    if config is None:
        config, _ = select_best(DESCRIPTOR, N=N, C=C, H=H, W=W, OC=F_out,
                                KH=KH, KW=KW, stride=S, padding=P)
    factory = _conv2d_relu_kernel_factory(N, C, H, W, F_out, KH, KW, S, P, D)
    prim_func = factory(**config)
    return tilelang.compile(prim_func, target="auto")


def estimate_total_flops(N, OH, OW, F_out, C, KH, KW):
    """Estimate FLOPs including conv and fused epilogue (ReLU)."""
    conv_flops = 2 * N * OH * OW * F_out * C * KH * KW
    out_elems = N * OH * OW * F_out
    relu_flops = out_elems
    return conv_flops + relu_flops


def verify_correctness(kernel, ref_prog, N, C, H, W, F_out, KH, KW, S, P, D):
    OH = (H + 2 * P - D * (KH - 1) - 1) // S + 1
    OW = (W + 2 * P - D * (KW - 1) - 1) // S + 1

    X = torch.randn(N, H, W, C, dtype=torch.float16, device="cuda")
    W_weight = torch.randn(KH, KW, C, F_out, dtype=torch.float16, device="cuda")

    Y_ref = torch.zeros(N, OH, OW, F_out, dtype=torch.float16, device="cuda")
    Y_tl = torch.zeros(N, OH, OW, F_out, dtype=torch.float16, device="cuda")

    ref_prog(X, W_weight, Y_ref)
    kernel(X, W_weight, Y_tl)

    try:
        torch.testing.assert_close(Y_tl, Y_ref, rtol=1e-2, atol=1e-2)
        return True
    except AssertionError as e:
        print(f"  [Verification] Status: FAILED -> {str(e)[:100]}...")
        return False


def main():
    parser = argparse.ArgumentParser(description="Benchmark Conv2d+ReLU fusion with autotuning")
    parser.add_argument("--output", default="conv2d_relu_results.csv", help="Output CSV path")
    parser.add_argument("--start-index", type=int, default=0, help="Start index in shape list")
    parser.add_argument("--max-shapes", type=int, default=0, help="Max number of shapes to run")
    parser.add_argument(
        "--target-shapes",
        type=int,
        default=200,
        help="Sample shapes with preserved FLOPs-scale distribution; 0 means no sampling.",
    )
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    all_shapes = get_conv_op_shape_list()
    print(f"Total shapes to benchmark: {len(all_shapes)}")

    start = max(args.start_index, 0)
    if args.max_shapes > 0:
        end = min(start + args.max_shapes, len(all_shapes))
        candidate_shapes = all_shapes[start:end]
    else:
        candidate_shapes = all_shapes[start:]

    if args.target_shapes > 0:
        shapes_to_test = get_sampled_conv_shapes(candidate_shapes, target_count=args.target_shapes)
    else:
        shapes_to_test = candidate_shapes

    print(f"Selected shape range: start={start}, base_count={len(candidate_shapes)}, sampled_count={len(shapes_to_test)}")

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("Operator: Conv2d + ReLU")
    print(f"Results will be saved to: {out_path}\n")

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Index",
                "Fusion",
                "N",
                "C",
                "H",
                "W",
                "F_out",
                "Kernel_H",
                "Kernel_W",
                "Stride",
                "Pad",
                "Dilation",
                "Best_Config",
                "Is_Correct",
                "TileLang_Lat_ms",
                "Torch_Lat_ms",
                "TileLang_TFlops",
                "Torch_TFlops",
                "Speedup",
            ]
        )

    for idx, shape in enumerate(shapes_to_test):
        N, C, H, W, F_out, KH, KW, S, P = shape
        D = 1

        OH = (H + 2 * P - D * (KH - 1) - 1) // S + 1
        OW = (W + 2 * P - D * (KW - 1) - 1) // S + 1
        print(f"[{idx+1}/{len(shapes_to_test)}] N={N}, C={C}, H={H}, W={W}, F={F_out}, K={KH}x{KW}, S={S}, P={P}")

        print("  [Fusion] RELU")
        try:
            best_config, _ = select_best(DESCRIPTOR, N=N, C=C, H=H, W=W, OC=F_out,
                                         KH=KH, KW=KW, stride=S, padding=P)
            kernel = build_conv2d_relu_kernel(N, C, H, W, F_out, KH, KW, S, P, D,
                                              config=best_config)

            ref_prog = get_ref_program(S, P, D)
            is_correct = verify_correctness(kernel, ref_prog, N, C, H, W, F_out, KH, KW, S, P, D)
            correct_str = "True" if is_correct else "False"
            if is_correct:
                print("    [Verification] Status: SUCCESS")

            X = torch.randn(N, H, W, C, dtype=torch.float16, device="cuda")
            W_weight = torch.randn(KH, KW, C, F_out, dtype=torch.float16, device="cuda")
            Y = torch.zeros(N, OH, OW, F_out, dtype=torch.float16, device="cuda")

            import triton
            tl_lat = triton.testing.do_bench(
                lambda: kernel(X, W_weight, Y),
                warmup=50, rep=100,
            )
            torch_ref = get_ref_program(S, P, D)
            torch_lat = triton.testing.do_bench(
                lambda: torch_ref(X, W_weight, Y),
                warmup=50, rep=100,
            )

            total_flops = estimate_total_flops(N, OH, OW, F_out, C, KH, KW)
            tl_tflops = total_flops / tl_lat * 1e-9
            torch_tflops = total_flops / torch_lat * 1e-9
            speedup = torch_lat / tl_lat

            print(f"    TileLang:  {tl_lat:.4f} ms  ({tl_tflops:.2f} TFlops)")
            print(f"    Torch:     {torch_lat:.4f} ms  ({torch_tflops:.2f} TFlops)")
            print(f"    Speedup:   {speedup:.2f}x")

            with out_path.open("a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        idx + 1,
                        "relu",
                        N,
                        C,
                        H,
                        W,
                        F_out,
                        KH,
                        KW,
                        S,
                        P,
                        D,
                        str(best_config),
                        correct_str,
                        f"{tl_lat:.4f}",
                        f"{torch_lat:.4f}",
                        f"{tl_tflops:.2f}",
                        f"{torch_tflops:.2f}",
                        f"{speedup:.2f}",
                    ]
                )

        except Exception as e:
            print(f"    [Error] Failed to benchmark: {str(e)}")
            with out_path.open("a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        idx + 1,
                        "relu",
                        N,
                        C,
                        H,
                        W,
                        F_out,
                        KH,
                        KW,
                        S,
                        P,
                        D,
                        f"ERROR: {str(e)}",
                        "Error",
                        "N/A",
                        "N/A",
                        "N/A",
                        "N/A",
                        "N/A",
                    ]
                )

        print()


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Unified autotuner descriptor
# ---------------------------------------------------------------------------


class Conv2dReluDescriptor(TileLangKernelBase):
    @property
    def name(self):
        return "conv2d_relu"

    def make_op_spec(self, N, C, H, W, OC, KH, KW, stride=1, padding=0, **kw):
        return make_conv2d_spec(
            N, C, H, W, OC, KH, KW, stride=stride, padding=padding,
            primitives=["relu"],
        )

    def get_raw_configs(self, **kw):
        return get_configs()


DESCRIPTOR = Conv2dReluDescriptor()
