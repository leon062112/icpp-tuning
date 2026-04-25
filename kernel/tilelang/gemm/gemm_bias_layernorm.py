import argparse
import itertools

import torch
import tilelang
import tilelang.language as T

from kernel.autotuner.interface import TileLangKernelBase, select_best
from kernel.autotuner.cost_model import make_matmul_spec

torch.backends.cuda.matmul.allow_tf32 = False


# Add project root to path so 'kernel' package is importable
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
    
def get_gemm_configs(K):
    """Generate autotune configurations for the GEMM + Bias kernel."""
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
def torch_baseline(A, B, bias, gamma, beta):
    """PyTorch baseline using torch.nn.functional.layer_norm."""
    N = A.shape[1] if B is None else B.shape[1]
    x = (A @ B).to(torch.float32) + bias.unsqueeze(0)
    x = torch.nn.functional.layer_norm(x, (N,), weight=gamma.float(), bias=beta.float())
    return x.to(torch.float16)


def _gemm_kernel_factory(M, N, K):
    """Return a GEMM+Bias kernel factory."""

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
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M),
                          threads=thread_num) as (bx, by):
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

                for i, j in T.Parallel(block_M, block_N):
                    C_local[i, j] = C_local[i, j] + Bias_shared[j]

                T.copy(C_local, C[by * block_M, bx * block_N])

        return main

    return kernel


def _build_layernorm_kernel(M, N, block_M_ln=16, block_N_ln=128,
                            thread_num_ln=128, eps=1e-5):
    """Build a standalone LayerNorm kernel."""
    ln_dtype = T.float32
    ln_out_dtype = T.float16
    num_n_tiles = N // block_N_ln
    assert N % block_N_ln == 0, f"N={N} must be divisible by block_N_ln={block_N_ln}"

    @T.prim_func
    def _ln_main(
        X: T.Tensor((M, N), ln_dtype),
        Gamma: T.Tensor((N,), ln_out_dtype),
        Beta: T.Tensor((N,), ln_out_dtype),
        C: T.Tensor((M, N), ln_out_dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M_ln), threads=thread_num_ln) as (by,):
            X_shared = T.alloc_shared((block_M_ln, block_N_ln), ln_dtype)
            Gamma_shared = T.alloc_shared((block_N_ln,), ln_out_dtype)
            Beta_shared = T.alloc_shared((block_N_ln,), ln_out_dtype)

            Acc_local = T.alloc_fragment((block_M_ln, block_N_ln), ln_dtype)
            Sum_local = T.alloc_fragment((block_M_ln,), ln_dtype)
            SumSq_local = T.alloc_fragment((block_M_ln,), ln_dtype)
            chunk_sum = T.alloc_fragment((block_M_ln,), ln_dtype)

            Mean_local = T.alloc_fragment((block_M_ln,), ln_dtype)
            InvStd_local = T.alloc_fragment((block_M_ln,), ln_dtype)

            X_local = T.alloc_fragment((block_M_ln, block_N_ln), ln_dtype)

            # Pass 1: gather statistics
            T.clear(Sum_local)
            T.clear(SumSq_local)

            for no in T.serial(num_n_tiles):
                T.copy(X[by * block_M_ln, no * block_N_ln], X_shared)

                T.reduce_sum(X_shared, chunk_sum, dim=1)
                for i in T.Parallel(block_M_ln):
                    Sum_local[i] = Sum_local[i] + chunk_sum[i]

                for i, j in T.Parallel(block_M_ln, block_N_ln):
                    Acc_local[i, j] = X_shared[i, j] * X_shared[i, j]
                T.reduce_sum(Acc_local, chunk_sum, dim=1)
                for i in T.Parallel(block_M_ln):
                    SumSq_local[i] = SumSq_local[i] + chunk_sum[i]

            for i in T.Parallel(block_M_ln):
                Mean_local[i] = Sum_local[i] / N
            for i in T.Parallel(block_M_ln):
                var = SumSq_local[i] / N - Mean_local[i] * Mean_local[i]
                InvStd_local[i] = T.rsqrt(var + eps)

            # Pass 2: normalize + scale + shift (reverse for L2 cache reuse)
            for no in T.serial(num_n_tiles):
                ridx = num_n_tiles - 1 - no
                T.copy(X[by * block_M_ln, ridx * block_N_ln], X_shared)
                T.copy(Gamma[ridx * block_N_ln], Gamma_shared)
                T.copy(Beta[ridx * block_N_ln], Beta_shared)

                for i, j in T.Parallel(block_M_ln, block_N_ln):
                    normalized = (X_shared[i, j] - Mean_local[i]) * InvStd_local[i]
                    X_local[i, j] = (
                        normalized * T.Cast(ln_dtype, Gamma_shared[j])
                        + T.Cast(ln_dtype, Beta_shared[j])
                    )

                for i, j in T.Parallel(block_M_ln, block_N_ln):
                    C[by * block_M_ln + i, ridx * block_N_ln + j] = T.Cast(
                        ln_out_dtype, X_local[i, j])

    return tilelang.compile(_ln_main, out_idx=[-1])


def build_fused_kernel(M, N, K, gemm_config=None, block_M_ln=16, block_N_ln=128,
                       thread_num_ln=128, eps=1e-5):
    """Build the two-kernel pipeline: GEMM+Bias -> LayerNorm.

    Uses cost-model selection for GEMM config if none provided.
    """
    if gemm_config is None:
        gemm_config, _ = select_best(DESCRIPTOR, M=M, N=N, K=K)

    gemm_fn = tilelang.compile(
        _gemm_kernel_factory(M, N, K)(**gemm_config),
        out_idx=[-1], target="auto",
    )
    ln_fn = _build_layernorm_kernel(M, N, block_M_ln=block_M_ln,
                                     block_N_ln=block_N_ln,
                                     thread_num_ln=thread_num_ln, eps=eps)

    def fused_kernel(A, B, bias, gamma, beta):
        x = gemm_fn(A, B, bias)
        return ln_fn(x, gamma, beta)

    return fused_kernel


# ---------------------------------------------------------------------------
# Unified autotuner descriptor
# ---------------------------------------------------------------------------


class GemmBiasLayerNormDescriptor(TileLangKernelBase):
    @property
    def name(self):
        return "gemm_bias_layernorm"

    def make_op_spec(self, M, N, K, **kw):
        return make_matmul_spec(M=M, N=N, K=K, primitives=["bias_add", "row_layernorm"])

    def get_raw_configs(self, K, **kw):
        return get_gemm_configs(K)


DESCRIPTOR = GemmBiasLayerNormDescriptor()


# ---------------------------------------------------------------------------
# Direct benchmark
# ---------------------------------------------------------------------------


def main():
    """Benchmark the fused kernel against PyTorch baseline."""
    import triton

    Ms = list(range(1, 256, 3))
    N = 2304
    K = 768

    block_N_ln = 128 if N % 128 == 0 else 256

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Shapes: M=1..255 step 3, N={N}, K={K}")
    print(f"LayerNorm config: block_M=16, block_N={block_N_ln}")
    print()
    print(f"{'M':>5} | {'Config':>30} | {'Fused ms':>10} | {'F.LN ms':>9} | {'vs F.LN':>9}")
    print("-" * 75)

    for M in Ms:
        best_config, score = select_best(DESCRIPTOR, M=M, N=N, K=K)
        kernel = build_fused_kernel(M, N, K, gemm_config=best_config, block_N_ln=block_N_ln)

        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)
        bias = torch.randn(N, device="cuda", dtype=torch.float16)
        gamma = torch.randn(N, device="cuda", dtype=torch.float16)
        beta = torch.randn(N, device="cuda", dtype=torch.float16)

        fused_lat = triton.testing.do_bench(
            lambda: kernel(a, b, bias, gamma, beta),
            warmup=100, rep=200,
        )

        torch_fn_lat = triton.testing.do_bench(
            lambda: torch_baseline(a, b, bias, gamma, beta),
            warmup=100, rep=200,
        )

        speedup_fn = torch_fn_lat / fused_lat
        cfg_str = DESCRIPTOR.format_config(best_config)

        print(
            f"{M:>5} | {cfg_str:>30} | {fused_lat:>10.4f} | {torch_fn_lat:>9.4f} | {speedup_fn:>8.2f}x"
        )

    print()


if __name__ == "__main__":
    main()
