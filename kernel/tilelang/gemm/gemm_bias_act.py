import itertools
import sys
from pathlib import Path

# Add project root to path so 'kernel' package is importable
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import tilelang
import tilelang.language as T

from kernel.autotuner.interface import TileLangKernelBase, select_best
from kernel.autotuner.cost_model import make_matmul_spec

torch.backends.cuda.matmul.allow_tf32 = False


def get_configs(M, K):
    """Generate autotune configurations for the fused kernel."""
    block_M = [16, 32, 64]
    block_N = [64, 128, 256]
    block_K = [32, 64]
    num_stages = [1, 2]
    thread_num = [128, 256]

    configs = []
    for bM, bN, bK, ns, tn in itertools.product(
        block_M, block_N, block_K, num_stages, thread_num,
    ):
        if K % bK != 0:
            continue
        configs.append({
            "block_M": bM,
            "block_N": bN,
            "block_K": bK,
            "num_stages": ns,
            "thread_num": tn,
        })
    return configs

@torch.compile
def ref_program(A, B, bias):
    """Reference PyTorch program for verification."""
    result = (A @ B).to(torch.float32) + bias.unsqueeze(0)
    return torch.relu(result)


def _kernel_factory(M, N, K):
    """Return a kernel factory that takes config kwargs and returns a prim_func."""

    def kernel(block_M=None, block_N=None, block_K=None,
               num_stages=None, thread_num=None):
        dtype = T.float16
        accum_dtype = T.float32
        out_dtype = T.float32

        @T.prim_func
        def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            Bias: T.Tensor((N,), dtype),
            C: T.Tensor((M, N), out_dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_K, block_N), dtype)
                Bias_shared = T.alloc_shared((block_N,), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

                T.clear(C_local)
                T.copy(Bias[bx * block_N], Bias_shared)

                for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(A[by * block_M, ko * block_K], A_shared)
                    T.copy(B[ko * block_K, bx * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)

                # Add bias + ReLU
                for i, j in T.Parallel(block_M, block_N):
                    C_local[i, j] = T.max(C_local[i, j] + Bias_shared[j], 0.0)

                T.copy(C_local, C[by * block_M, bx * block_N])

        return main

    return kernel


def build_kernel(M, N, K, config=None):
    """Build a compiled kernel using cost-model selection or a given config."""
    if config is None:
        config, _ = select_best(DESCRIPTOR, M=M, N=N, K=K)
    prim_func = _kernel_factory(M, N, K)(**config)
    return tilelang.compile(prim_func, out_idx=[-1], target="auto")


# ---------------------------------------------------------------------------
# Unified autotuner descriptor
# ---------------------------------------------------------------------------


class GemmBiasActDescriptor(TileLangKernelBase):
    @property
    def name(self):
        return "gemm_bias_relu"

    def make_op_spec(self, M, N, K, **kw):
        return make_matmul_spec(M=M, N=N, K=K, primitives=["bias_add", "relu"])

    def get_raw_configs(self, M, K, **kw):
        return get_configs(M, K)


DESCRIPTOR = GemmBiasActDescriptor()


# ---------------------------------------------------------------------------
# Direct benchmark
# ---------------------------------------------------------------------------


def main():
    """Benchmark the fused kernel against PyTorch baseline."""
    import triton

    Ms = list(range(1, 256, 3))
    N = 2304
    K = 768

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Shapes: M=1..255 step 3, N={N}, K={K}")
    print()
    print(f"{'M':>5} | {'Config':>30} | {'TileLang ms':>12} | {'Torch ms':>10} | {'Speedup':>8} | {'TFLOPS':>8}")
    print("-" * 90)

    for M in Ms:
        total_flops = 2 * M * N * K

        best_config, score = select_best(DESCRIPTOR, M=M, N=N, K=K)
        kernel = build_kernel(M, N, K, config=best_config)

        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)
        bias = torch.randn(N, device="cuda", dtype=torch.float16)

        tl_lat = triton.testing.do_bench(
            lambda: kernel(a, b, bias),
            warmup=100, rep=200,
        )

        torch_lat = triton.testing.do_bench(
            lambda: ref_program(a, b, bias),
            warmup=100, rep=200,
        )

        c_tl = kernel(a, b, bias)
        c_ref = ref_program(a, b, bias)
        ok = torch.allclose(c_tl.float(), c_ref.float(), rtol=1e-2, atol=1e-2)

        tl_tflops = total_flops / tl_lat * 1e-9
        speedup = torch_lat / tl_lat
        status = "PASS" if ok else "FAIL"
        cfg_str = DESCRIPTOR.format_config(best_config)

        print(f"{M:>5} | {cfg_str:>30} | {tl_lat:>12.4f} | {torch_lat:>10.4f} | {speedup:>7.2f}x | {tl_tflops:>7.2f}  {status}")

    print()


if __name__ == "__main__":
    main()
