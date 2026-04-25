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
    from conv_bn_relu import get_configs, get_conv_op_shape_list, get_sampled_conv_shapes
except ImportError:
    from conv2d_bn_relu import get_configs, get_conv_op_shape_list, get_sampled_conv_shapes


torch.backends.cuda.matmul.allow_tf32 = False


def get_ref_program(stride=1, padding=1, dilation=1, eps=1e-5):
    """Reference implementation: Conv2d + BatchNorm(inference) + Add + ReLU."""

    def conv_bn_add_relu_core(x_nchw, w_oihw, residual_nchw, bn_gamma, bn_beta, bn_mean, bn_var):
        y_nchw = F.conv2d(x_nchw, w_oihw, stride=stride, padding=padding, dilation=dilation).to(torch.float32)

        gamma = bn_gamma.to(torch.float32).view(1, -1, 1, 1)
        beta = bn_beta.to(torch.float32).view(1, -1, 1, 1)
        mean = bn_mean.to(torch.float32).view(1, -1, 1, 1)
        # Randn autotune supply may generate negative variance.
        var = torch.clamp(bn_var.to(torch.float32), min=0.0).view(1, -1, 1, 1)

        y_nchw = (y_nchw - mean) * torch.rsqrt(var + eps) * gamma + beta
        y_nchw = y_nchw + residual_nchw.to(torch.float32)
        y_nchw = torch.relu(y_nchw)
        return y_nchw

    compiled_core = None
    if hasattr(torch, "compile"):
        try:
            compiled_core = torch.compile(conv_bn_add_relu_core)
        except Exception:
            compiled_core = None

    def ref_program(x, w, residual, bn_gamma, bn_beta, bn_mean, bn_var, y):
        nonlocal compiled_core
        x_nchw = x.permute(0, 3, 1, 2)
        w_oihw = w.permute(3, 2, 0, 1)
        residual_nchw = residual.permute(0, 3, 1, 2)

        if compiled_core is not None:
            try:
                y_nchw = compiled_core(x_nchw, w_oihw, residual_nchw, bn_gamma, bn_beta, bn_mean, bn_var)
            except Exception:
                compiled_core = None
                y_nchw = conv_bn_add_relu_core(x_nchw, w_oihw, residual_nchw, bn_gamma, bn_beta, bn_mean, bn_var)
        else:
            y_nchw = conv_bn_add_relu_core(x_nchw, w_oihw, residual_nchw, bn_gamma, bn_beta, bn_mean, bn_var)

        y.copy_(y_nchw.permute(0, 2, 3, 1).to(y.dtype))

    return ref_program


def _conv2d_bn_add_relu_kernel_factory(n, c, h, w, f_out, kh, kw, s=1, p=1, d=1, eps=1e-5):
    """Return a Conv2d+BN+Add+ReLU kernel factory."""
    oh = (h + 2 * p - d * (kh - 1) - 1) // s + 1
    ow = (w + 2 * p - d * (kw - 1) - 1) // s + 1

    m_eff = n * oh * ow
    k_eff = kh * kw * c

    def kernel(block_M=None, block_N=None, block_K=None, num_stages=None, thread_num=None):
        dtype = T.float16
        accum_dtype = T.float32
        out_dtype = T.float16

        @T.prim_func
        def conv_bn_add_relu_kernel(
            data: T.Tensor((n, h, w, c), dtype),
            kernel_weight: T.Tensor((kh, kw, c, f_out), dtype),
            residual: T.Tensor((n, oh, ow, f_out), dtype),
            bn_gamma: T.Tensor((f_out,), dtype),
            bn_beta: T.Tensor((f_out,), dtype),
            bn_mean: T.Tensor((f_out,), dtype),
            bn_var: T.Tensor((f_out,), dtype),
            out: T.Tensor((n, oh, ow, f_out), out_dtype),
        ):
            with T.Kernel(T.ceildiv(f_out, block_N), T.ceildiv(m_eff, block_M), threads=thread_num) as (bx, by):
                data_shared = T.alloc_shared((block_M, block_K), dtype)
                kernel_shared = T.alloc_shared((block_K, block_N), dtype)
                out_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                out_shared = T.alloc_shared((block_M, block_N), out_dtype)

                kernel_flat = T.Tensor((k_eff, f_out), dtype, kernel_weight.data)
                residual_flat = T.Tensor((m_eff, f_out), dtype, residual.data)
                out_flat = T.Tensor((m_eff, f_out), out_dtype, out.data)

                T.clear(out_local)

                for k_iter in T.Pipelined(T.ceildiv(k_eff, block_K), num_stages=num_stages):
                    for i, j in T.Parallel(block_M, block_K):
                        k_idx = k_iter * block_K + j
                        m_idx = by * block_M + i

                        access_h = m_idx % (oh * ow) // ow * s + k_idx // (kw * c) * d - p
                        access_w = m_idx % ow * s + k_idx // c % kw * d - p

                        safe_h = T.min(T.max(access_h, 0), h - 1)
                        safe_w = T.min(T.max(access_w, 0), w - 1)

                        in_bound = (
                            (access_h >= 0)
                            and (access_w >= 0)
                            and (access_h < h)
                            and (access_w < w)
                            and (m_idx < m_eff)
                            and (k_idx < k_eff)
                        )

                        if in_bound:
                            data_shared[i, j] = data[m_idx // (oh * ow), safe_h, safe_w, k_idx % c]
                        else:
                            data_shared[i, j] = T.cast(0, dtype)

                    if k_eff % block_K == 0 and f_out % block_N == 0:
                        T.copy(kernel_flat[k_iter * block_K, bx * block_N], kernel_shared)
                    else:
                        for i, j in T.Parallel(block_K, block_N):
                            k_idx = k_iter * block_K + i
                            n_idx = bx * block_N + j
                            if k_idx < k_eff and n_idx < f_out:
                                kernel_shared[i, j] = kernel_flat[k_idx, n_idx]
                            else:
                                kernel_shared[i, j] = T.cast(0, dtype)

                    T.gemm(data_shared, kernel_shared, out_local)

                T.copy(out_local, out_shared)

                # Fused epilogue in one pass: BN + residual add + ReLU.
                for i, j in T.Parallel(block_M, block_N):
                    m_idx = by * block_M + i
                    n_idx = bx * block_N + j
                    if m_idx < m_eff and n_idx < f_out:
                        x_val = T.cast(out_shared[i, j], T.float32)
                        gamma = T.cast(bn_gamma[n_idx], T.float32)
                        beta = T.cast(bn_beta[n_idx], T.float32)
                        mean = T.cast(bn_mean[n_idx], T.float32)
                        var = T.max(T.cast(bn_var[n_idx], T.float32), T.cast(0, T.float32))
                        inv_std = 1.0 / T.sqrt(var + T.cast(eps, T.float32))
                        y_bn = (x_val - mean) * inv_std * gamma + beta
                        y_add = y_bn + T.cast(residual_flat[m_idx, n_idx], T.float32)
                        out_flat[m_idx, n_idx] = T.cast(T.max(y_add, T.cast(0, T.float32)), out_dtype)

        return conv_bn_add_relu_kernel

    return kernel


def build_conv2d_bn_add_relu_kernel(n, c, h, w, f_out, kh, kw, s=1, p=1, d=1, config=None):
    """Build a compiled Conv2d+BN+Add+ReLU kernel using cost-model selection or a given config."""
    if config is None:
        config, _ = select_best(DESCRIPTOR, N=n, C=c, H=h, W=w, OC=f_out,
                                KH=kh, KW=kw, stride=s, padding=p)
    factory = _conv2d_bn_add_relu_kernel_factory(n, c, h, w, f_out, kh, kw, s, p, d)
    prim_func = factory(**config)
    return tilelang.compile(prim_func, target="auto")


def estimate_total_flops(n, oh, ow, f_out, c, kh, kw):
    """Estimate FLOPs including conv and fused epilogue (BN + Add + ReLU)."""
    conv_flops = 2 * n * oh * ow * f_out * c * kh * kw
    out_elems = n * oh * ow * f_out
    # BN + Add + ReLU elementwise FLOPs approximation.
    epilogue_flops = 8 * out_elems
    return conv_flops + epilogue_flops


def verify_correctness(kernel, ref_prog, n, c, h, w, f_out, kh, kw, s, p, d):
    oh = (h + 2 * p - d * (kh - 1) - 1) // s + 1
    ow = (w + 2 * p - d * (kw - 1) - 1) // s + 1

    x = torch.randn(n, h, w, c, dtype=torch.float16, device="cuda")
    w_weight = torch.randn(kh, kw, c, f_out, dtype=torch.float16, device="cuda")
    residual = torch.randn(n, oh, ow, f_out, dtype=torch.float16, device="cuda")
    bn_gamma = torch.randn(f_out, dtype=torch.float16, device="cuda")
    bn_beta = torch.randn(f_out, dtype=torch.float16, device="cuda")
    bn_mean = torch.randn(f_out, dtype=torch.float16, device="cuda")
    bn_var = torch.rand(f_out, dtype=torch.float16, device="cuda") + torch.tensor(0.5, dtype=torch.float16, device="cuda")

    y_ref = torch.zeros(n, oh, ow, f_out, dtype=torch.float16, device="cuda")
    y_tl = torch.zeros(n, oh, ow, f_out, dtype=torch.float16, device="cuda")

    ref_prog(x, w_weight, residual, bn_gamma, bn_beta, bn_mean, bn_var, y_ref)
    kernel(x, w_weight, residual, bn_gamma, bn_beta, bn_mean, bn_var, y_tl)

    try:
        torch.testing.assert_close(y_tl, y_ref, rtol=1e-2, atol=1e-2)
        return True
    except AssertionError as err:
        print(f"  [Verification] Status: FAILED -> {str(err)[:100]}...")
        return False


# ---------------------------------------------------------------------------
# Unified autotuner descriptor
# ---------------------------------------------------------------------------


class ConvBnAddReluDescriptor(TileLangKernelBase):
    @property
    def name(self):
        return "conv_bn_add_relu"

    def make_op_spec(self, N, C, H, W, OC, KH, KW, stride=1, padding=0, **kw):
        return make_conv2d_spec(
            N, C, H, W, OC, KH, KW, stride=stride, padding=padding,
            primitives=["batchnorm", "relu"],
        )

    def get_raw_configs(self, **kw):
        return get_configs()


DESCRIPTOR = ConvBnAddReluDescriptor()


def main():
    parser = argparse.ArgumentParser(description="Benchmark Conv2d+BN+Add+ReLU fusion with autotuning")
    parser.add_argument(
        "--output",
        default="conv2d_bn_add_relu_results.csv",
        help="Output CSV path (default: conv2d_bn_add_relu_results.csv)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index in shape list (default: 0)",
    )
    parser.add_argument(
        "--max-shapes",
        type=int,
        default=0,
        help="Max number of shapes to run; <=0 means all remaining shapes",
    )
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
    csv_filename = str(out_path)
    print(f"\n{'='*80}")
    print("Operator: Conv2d + BatchNorm + Add + ReLU")
    print(f"Results will be saved to: {csv_filename}\n")

    with open(csv_filename, mode="w", newline="") as f:
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
        n, c, h, w, f_out, kh, kw, s, p = shape
        d = 1

        oh = (h + 2 * p - d * (kh - 1) - 1) // s + 1
        ow = (w + 2 * p - d * (kw - 1) - 1) // s + 1
        print(f"[{idx+1}/{len(shapes_to_test)}] N={n}, C={c}, H={h}, W={w}, F={f_out}, K={kh}x{kw}, S={s}, P={p}")

        print("  [Fusion] CONV_BN_ADD_RELU")
        try:
            best_config, _ = select_best(DESCRIPTOR, N=n, C=c, H=h, W=w, OC=f_out,
                                         KH=kh, KW=kw, stride=s, padding=p)
            kernel = build_conv2d_bn_add_relu_kernel(n, c, h, w, f_out, kh, kw, s, p, d,
                                                      config=best_config)

            ref_prog = get_ref_program(s, p, d)
            is_correct = verify_correctness(kernel, ref_prog, n, c, h, w, f_out, kh, kw, s, p, d)
            correct_str = "True" if is_correct else "False"
            if is_correct:
                print("    [Verification] Status: SUCCESS")

            x = torch.randn(n, h, w, c, dtype=torch.float16, device="cuda")
            w_weight = torch.randn(kh, kw, c, f_out, dtype=torch.float16, device="cuda")
            residual = torch.randn(n, oh, ow, f_out, dtype=torch.float16, device="cuda")
            bn_gamma = torch.randn(f_out, dtype=torch.float16, device="cuda")
            bn_beta = torch.randn(f_out, dtype=torch.float16, device="cuda")
            bn_mean = torch.randn(f_out, dtype=torch.float16, device="cuda")
            bn_var = torch.rand(f_out, dtype=torch.float16, device="cuda") + 0.5
            y = torch.zeros(n, oh, ow, f_out, dtype=torch.float16, device="cuda")

            import triton
            tl_lat = triton.testing.do_bench(
                lambda: kernel(x, w_weight, residual, bn_gamma, bn_beta, bn_mean, bn_var, y),
                warmup=50, rep=100,
            )
            torch_ref = get_ref_program(s, p, d)
            torch_lat = triton.testing.do_bench(
                lambda: torch_ref(x, w_weight, residual, bn_gamma, bn_beta, bn_mean, bn_var, y),
                warmup=50, rep=100,
            )

            total_flops = estimate_total_flops(n, oh, ow, f_out, c, kh, kw)
            tl_tflops = total_flops / tl_lat * 1e-9
            torch_tflops = total_flops / torch_lat * 1e-9
            speedup = torch_lat / tl_lat

            print(f"    TileLang:  {tl_lat:.4f} ms  ({tl_tflops:.2f} TFlops)")
            print(f"    Torch:     {torch_lat:.4f} ms  ({torch_tflops:.2f} TFlops)")
            print(f"    Speedup:   {speedup:.2f}x")

            with open(csv_filename, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        idx + 1,
                        "conv_bn_add_relu",
                        n,
                        c,
                        h,
                        w,
                        f_out,
                        kh,
                        kw,
                        s,
                        p,
                        d,
                        str(best_config),
                        correct_str,
                        f"{tl_lat:.4f}",
                        f"{torch_lat:.4f}",
                        f"{tl_tflops:.2f}",
                        f"{torch_tflops:.2f}",
                        f"{speedup:.2f}",
                    ]
                )

        except Exception as err:
            print(f"    [Error] Failed to benchmark: {str(err)}")
            with open(csv_filename, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        idx + 1,
                        "conv_bn_add_relu",
                        n,
                        c,
                        h,
                        w,
                        f_out,
                        kh,
                        kw,
                        s,
                        p,
                        d,
                        f"ERROR: {str(err)}",
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
