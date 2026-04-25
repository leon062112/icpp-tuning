import itertools
import math
import torch
import tilelang
import tilelang.language as T
import sys
from pathlib import Path

# Add project root to path so 'kernel' package is importable
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from kernel.autotuner.interface import TileLangKernelBase, select_best
from kernel.autotuner.cost_model import make_matmul_spec

torch.backends.cuda.matmul.allow_tf32 = False

@torch.compile
def ref_program(A, B, C):
    C[...] = (A @ B).to(torch.float32)


def get_configs(M, K, split_k_values=(2,)):
    block_M = [16,32,64]
    block_N = [128, 256]
    block_K = [32, 64, 128]
    num_stages = [1, 2, 4,8]
    thread_num = [64, 128, 256]

    configs = []
    for bM, bN, bK, sk, ns, tn in itertools.product(
        block_M, block_N, block_K, split_k_values, num_stages, thread_num,
    ):
        if K % sk != 0:
            continue
        chunk = K // sk
        if chunk % bK != 0:
            continue
        configs.append({
            "block_M": bM,
            "block_N": bN,
            "block_K": bK,
            "split_k": sk,
            "num_stages": ns,
            "thread_num": tn,
        })
    return configs


def _kernel_factory(M, N, K):
    """Return a kernel factory that takes config kwargs and returns a prim_func."""

    def kernel(block_M=None, block_N=None, block_K=None, split_k=None,
               num_stages=None, thread_num=None):
        dtype = T.float16
        accum_dtype = T.float32
        out_dtype = T.float32
        splitK = K // split_k

        @T.prim_func
        def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), out_dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), split_k, threads=thread_num) as (bx, by, bz):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_K, block_N), dtype)
                C_shared = T.alloc_shared((block_M, block_N), out_dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                T.clear(C_local)
                for ko in T.Pipelined(T.ceildiv(splitK, block_K), num_stages=num_stages):
                    T.copy(A[by * block_M, bz * splitK + ko * block_K], A_shared)
                    T.copy(B[bz * splitK + ko * block_K, bx * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)
                T.copy(C_local, C_shared)
                T.atomic_add(C[by * block_M, bx * block_N], C_shared)
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


class GemmDescriptor(TileLangKernelBase):
    @property
    def name(self):
        return "gemm_splitk"

    def make_op_spec(self, raw_config=None, M=0, N=0, K=0, **kw):
        split_k = raw_config.get("split_k", 2) if raw_config else 2
        return make_matmul_spec(M=M, N=N, K=K // split_k)

    def get_raw_configs(self, M, K, **kw):
        return get_configs(M, K)

    def score_adjustment(self, raw, base_score, **kw):
        sk = raw.get("split_k", 2)
        return base_score * math.sqrt(sk) / sk


DESCRIPTOR = GemmDescriptor()


# ---------------------------------------------------------------------------
# Direct benchmark
# ---------------------------------------------------------------------------


def main():
    Ms = list(range(1, 256, 3))
    N = 2304
    K = 768

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    print(f"{'M':>5} | {'Config':>35} | {'TileLang ms':>12} | {'Torch ms':>10} | {'Speedup':>8} | {'TFLOPS':>8}")
    print("-" * 95)
    import triton

    for M in Ms:
        total_flops = 2 * M * N * K

        best_config, score = select_best(DESCRIPTOR, M=M, N=N, K=K)
        kernel = build_kernel(M, N, K, config=best_config)

        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)


        tl_lat = triton.testing.do_bench(
            lambda: kernel(a, b),
            warmup=100, rep=200,
        )
        torch_lat = triton.testing.do_bench(
            lambda: torch.mm(a.float(), b.float()),
            warmup=100, rep=200,
        )

        tl_tflops = total_flops / tl_lat * 1e-9
        torch_tflops = total_flops / torch_lat * 1e-9
        speedup = torch_lat / tl_lat
        cfg_str = DESCRIPTOR.format_config(best_config)

        print(f"{M:>5} | {cfg_str:>35} | {tl_lat:>12.4f} | {torch_lat:>10.4f} | {speedup:>7.2f}x | {tl_tflops:>7.2f}")
        print()


if __name__ == "__main__":
    main()
