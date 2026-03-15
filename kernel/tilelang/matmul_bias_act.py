"""
Matmul + Bias + Activation 融合算子实现

面向动态形状的硬件感知的算子融合优化与高性能模板设计

支持的融合模式:
- Matmul + Bias + ReLU
- Matmul + Bias + GELU
- Matmul + Bias + SiLU
"""

import tilelang
import tilelang.language as T
import tilelang.testing


def _make_matmul_bias_relu_kernel(
    block_M: int,
    block_N: int,
    block_K: int,
    trans_A: bool,
    trans_B: bool,
    in_dtype: str,
    out_dtype: str,
    accum_dtype: str,
    num_stages: int,
    threads: int,
):
    """Matmul + Bias + ReLU 融合算子"""
    M = T.dynamic("m")
    N = T.dynamic("n")
    K = T.dynamic("k")

    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)

    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    @T.prim_func
    def kernel(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        Bias: T.Tensor((N,), in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            Bias_shared = T.alloc_shared((block_N,), in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            T.copy(Bias[bx * block_N], Bias_shared)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_shared)
                else:
                    T.copy(A[by * block_M, k * block_K], A_shared)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local, trans_A, trans_B)

            Bias_local = T.alloc_fragment((block_N,), accum_dtype)
            T.copy(Bias_shared, Bias_local)

            # Fused Bias + ReLU
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = T.max(C_local[i, j] + T.cast(Bias_local[j], accum_dtype), T.float32(0.0))

            T.copy(C_local, C[by * block_M, bx * block_N])

    return kernel


def _make_matmul_bias_gelu_kernel(
    block_M: int,
    block_N: int,
    block_K: int,
    trans_A: bool,
    trans_B: bool,
    in_dtype: str,
    out_dtype: str,
    accum_dtype: str,
    num_stages: int,
    threads: int,
):
    """Matmul + Bias + GELU 融合算子 (tanh 近似)"""
    M = T.dynamic("m")
    N = T.dynamic("n")
    K = T.dynamic("k")

    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)

    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    @T.prim_func
    def kernel(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        Bias: T.Tensor((N,), in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            Bias_shared = T.alloc_shared((block_N,), in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            T.copy(Bias[bx * block_N], Bias_shared)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_shared)
                else:
                    T.copy(A[by * block_M, k * block_K], A_shared)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local, trans_A, trans_B)

            Bias_local = T.alloc_fragment((block_N,), accum_dtype)
            T.copy(Bias_shared, Bias_local)

            # Fused Bias + GELU (tanh approximation)
            # GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
            for i, j in T.Parallel(block_M, block_N):
                x = C_local[i, j] + T.cast(Bias_local[j], accum_dtype)
                # Use float32 constants
                inner = T.float32(0.7978845608028654) * (x + T.float32(0.044715) * x * x * x)
                C_local[i, j] = T.float32(0.5) * x * (T.float32(1.0) + T.tanh(inner))

            T.copy(C_local, C[by * block_M, bx * block_N])

    return kernel


def _make_matmul_bias_silu_kernel(
    block_M: int,
    block_N: int,
    block_K: int,
    trans_A: bool,
    trans_B: bool,
    in_dtype: str,
    out_dtype: str,
    accum_dtype: str,
    num_stages: int,
    threads: int,
):
    """Matmul + Bias + SiLU 融合算子"""
    M = T.dynamic("m")
    N = T.dynamic("n")
    K = T.dynamic("k")

    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)

    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    @T.prim_func
    def kernel(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        Bias: T.Tensor((N,), in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            Bias_shared = T.alloc_shared((block_N,), in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            T.copy(Bias[bx * block_N], Bias_shared)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_shared)
                else:
                    T.copy(A[by * block_M, k * block_K], A_shared)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local, trans_A, trans_B)

            Bias_local = T.alloc_fragment((block_N,), accum_dtype)
            T.copy(Bias_shared, Bias_local)

            # Fused Bias + SiLU
            # SiLU(x) = x * sigmoid(x)
            for i, j in T.Parallel(block_M, block_N):
                x = C_local[i, j] + T.cast(Bias_local[j], accum_dtype)
                C_local[i, j] = x * T.sigmoid(x)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return kernel


# Kernel factory mapping
_KERNEL_FACTORIES = {
    "relu": _make_matmul_bias_relu_kernel,
    "gelu": _make_matmul_bias_gelu_kernel,
    "silu": _make_matmul_bias_silu_kernel,
}


@tilelang.jit
def matmul_bias_activation_dynamic(
    block_M: int,
    block_N: int,
    block_K: int,
    trans_A: bool,
    trans_B: bool,
    in_dtype: str,
    out_dtype: str,
    accum_dtype: str,
    num_stages: int,
    threads: int,
    activation: str = "relu",
):
    """
    动态形状的 Matmul + Bias + Activation 融合算子

    Args:
        block_M: M 维度的分块大小
        block_N: N 维度的分块大小
        block_K: K 维度的分块大小
        trans_A: A 矩阵是否转置
        trans_B: B 矩阵是否转置
        in_dtype: 输入数据类型 ("float16", "float32", etc.)
        out_dtype: 输出数据类型
        accum_dtype: 累加器数据类型
        num_stages: 流水线阶段数
        threads: 每个 thread block 的线程数
        activation: 激活函数类型 ("relu", "gelu", "silu")

    Returns:
        编译后的 kernel 函数
    """
    if activation not in _KERNEL_FACTORIES:
        raise ValueError(f"Unsupported activation: {activation}. Supported: {list(_KERNEL_FACTORIES.keys())}")

    factory = _KERNEL_FACTORIES[activation]
    return tilelang.jit(factory)(
        block_M, block_N, block_K, trans_A, trans_B,
        in_dtype, out_dtype, accum_dtype, num_stages, threads
    )


# === 便捷封装函数 ===

@tilelang.jit
def matmul_bias_relu(
    block_M=128,
    block_N=128,
    block_K=32,
    trans_A=False,
    trans_B=False,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float32",
    num_stages=3,
    threads=128,
):
    """Matmul + Bias + ReLU 融合算子"""
    return _make_matmul_bias_relu_kernel(
        block_M, block_N, block_K, trans_A, trans_B,
        in_dtype, out_dtype, accum_dtype, num_stages, threads
    )


@tilelang.jit
def matmul_bias_gelu(
    block_M=128,
    block_N=128,
    block_K=32,
    trans_A=False,
    trans_B=False,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float32",
    num_stages=3,
    threads=128,
):
    """Matmul + Bias + GELU 融合算子"""
    return _make_matmul_bias_gelu_kernel(
        block_M, block_N, block_K, trans_A, trans_B,
        in_dtype, out_dtype, accum_dtype, num_stages, threads
    )


@tilelang.jit
def matmul_bias_silu(
    block_M=128,
    block_N=128,
    block_K=32,
    trans_A=False,
    trans_B=False,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float32",
    num_stages=3,
    threads=128,
):
    """Matmul + Bias + SiLU 融合算子"""
    return _make_matmul_bias_silu_kernel(
        block_M, block_N, block_K, trans_A, trans_B,
        in_dtype, out_dtype, accum_dtype, num_stages, threads
    )


# === 测试函数 ===

def test_matmul_bias_activation(M=4096, N=4096, K=4096, activation="relu"):
    """
    测试 Matmul + Bias + Activation 融合算子

    Args:
        M, N, K: 矩阵维度
        activation: 激活函数类型 ("relu", "gelu", "silu")
    """
    print(f"Testing Matmul + Bias + {activation.upper()} with M={M}, N={N}, K={K}")

    block_M, block_N, block_K = 128, 128, 32
    trans_A, trans_B = False, False
    in_dtype, out_dtype = "float16", "float16"
    accum_dtype = "float32"
    num_stages = 3
    threads = 128

    # Select the appropriate kernel based on activation
    if activation == "relu":
        kernel = matmul_bias_relu(block_M, block_N, block_K, trans_A, trans_B,
                                  in_dtype, out_dtype, accum_dtype, num_stages, threads)
    elif activation == "gelu":
        kernel = matmul_bias_gelu(block_M, block_N, block_K, trans_A, trans_B,
                                  in_dtype, out_dtype, accum_dtype, num_stages, threads)
    elif activation == "silu":
        kernel = matmul_bias_silu(block_M, block_N, block_K, trans_A, trans_B,
                                  in_dtype, out_dtype, accum_dtype, num_stages, threads)
    else:
        raise ValueError(f"Unsupported activation: {activation}")

    import torch

    A = torch.rand(M, K, device="cuda", dtype=torch.float16)
    B = torch.rand(K, N, device="cuda", dtype=torch.float16)
    Bias = torch.rand(N, device="cuda", dtype=torch.float16)
    C = torch.zeros(M, N, device="cuda", dtype=torch.float16)

    kernel(A, B, Bias, C)

    # Reference implementation
    ref_C = torch.matmul(A.to(torch.float32), B.to(torch.float32))
    ref_C = ref_C + Bias.to(torch.float32)

    if activation == "relu":
        ref_C = torch.nn.functional.relu(ref_C)
    elif activation == "gelu":
        ref_C = torch.nn.functional.gelu(ref_C)
    elif activation == "silu":
        ref_C = torch.nn.functional.silu(ref_C)

    ref_C = ref_C.to(torch.float16)

    # Verify correctness
    torch.testing.assert_close(C, ref_C, rtol=1e-2, atol=1e-2)
    print(f"✓ Matmul + Bias + {activation.upper()} test passed!")

    # Benchmark
    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)
    latency = profiler.do_bench(input_tensors=[A, B, Bias, C])
    print(f"Latency: {latency:.3f} ms")

    # Calculate TFLOPS
    tflops = 2 * M * N * K / (latency * 1e-3) / 1e12
    print(f"Performance: {tflops:.2f} TFLOPS")

    return latency


def run_perf_comparison(M=4096, N=4096, K=4096):
    """
    性能对比：融合算子 vs 分开调用
    """
    import torch
    import time

    print(f"\n=== Performance Comparison (M={M}, N={N}, K={K}) ===\n")

    # Prepare data
    A = torch.rand(M, K, device="cuda", dtype=torch.float16)
    B = torch.rand(K, N, device="cuda", dtype=torch.float16)
    Bias = torch.rand(N, device="cuda", dtype=torch.float16)

    # Warm up
    for _ in range(10):
        C1 = torch.matmul(A, B)
        C1 = C1 + Bias
        C1 = torch.nn.functional.relu(C1)

    # Baseline: separate calls
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        C1 = torch.matmul(A, B)
        C1 = C1 + Bias
        C1 = torch.nn.functional.relu(C1)
    torch.cuda.synchronize()
    baseline_time = (time.time() - start) / 100 * 1000

    print(f"PyTorch (separate): {baseline_time:.3f} ms")

    # Fused kernel
    kernel = matmul_bias_relu()
    C2 = torch.zeros(M, N, device="cuda", dtype=torch.float16)

    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)
    fused_time = profiler.do_bench(input_tensors=[A, B, Bias, C2])

    print(f"TileLang (fused):   {fused_time:.3f} ms")
    print(f"Speedup:            {baseline_time / fused_time:.2f}x")


def main():
    """主函数：运行所有测试"""
    print("=" * 60)
    print("Matmul + Bias + Activation Fused Kernel Tests")
    print("=" * 60)

    # Test different activations
    for activation in ["relu", "gelu", "silu"]:
        print()
        test_matmul_bias_activation(M=4096, N=4096, K=4096, activation=activation)

    # Performance comparison
    print()
    run_perf_comparison(M=4096, N=4096, K=4096)


if __name__ == "__main__":
    main()
