"""
动态形状的 LayerNorm/RMSNorm 算子实现 (优化版本)

面向动态形状的硬件感知的算子融合优化与高性能模板设计

参考 rms_norm.py 的优化策略:
1. 使用 T.reduce_sum 进行并行归约
2. 使用 T.rsqrt 直接计算
3. 使用 T.Parallel 并行计算输出
"""

import tilelang
import tilelang.language as T
import tilelang.testing


@tilelang.jit(out_idx=[-1])
def layernorm_optimized(
    M: int,
    N: int,
    blk_m: int,
    dtype: str = "float",
    eps: float = 1e-5,
):
    """
    优化的 LayerNorm 算子

    计算: y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias

    使用 T.reduce_sum 进行并行归约
    """
    @T.prim_func
    def main(
        X: T.Tensor((M, N), dtype),
        Weight: T.Tensor((N,), dtype),
        Bias: T.Tensor((N,), dtype),
        Y: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(M, blk_m), threads=128) as bx:
            X_shared = T.alloc_shared((blk_m, N), dtype)
            W_shared = T.alloc_shared((N,), dtype)
            B_shared = T.alloc_shared((N,), dtype)

            # Fragment buffers
            X_local = T.alloc_fragment((blk_m, N), dtype)
            X_pow_local = T.alloc_fragment((blk_m, N), dtype)
            mean_local = T.alloc_fragment((blk_m,), dtype)
            var_local = T.alloc_fragment((blk_m,), dtype)

            # 加载数据
            T.copy(X[bx * blk_m : (bx + 1) * blk_m, :], X_shared)
            T.copy(Weight[:], W_shared)
            T.copy(Bias[:], B_shared)
            T.copy(X_shared, X_local)

            # 计算 mean = sum(x) / N
            T.reduce_sum(X_local, mean_local, dim=1)
            for i in T.Parallel(blk_m):
                mean_local[i] = mean_local[i] / N

            # 计算 var = sum((x - mean)^2) / N
            # 使用 E[x^2] - E[x]^2 公式
            for i, j in T.Parallel(blk_m, N):
                X_pow_local[i, j] = X_local[i, j] * X_local[i, j]
            T.reduce_sum(X_pow_local, var_local, dim=1)
            for i in T.Parallel(blk_m):
                var_local[i] = var_local[i] / N - mean_local[i] * mean_local[i]
                # 计算 rstd = 1 / sqrt(var + eps)
                var_local[i] = T.rsqrt(var_local[i] + eps)

            # 计算 y = (x - mean) * rstd * weight + bias
            for i, j in T.Parallel(blk_m, N):
                X_local[i, j] = (X_local[i, j] - mean_local[i]) * var_local[i]
                X_local[i, j] = X_local[i, j] * W_shared[j] + B_shared[j]

            T.copy(X_local, Y[bx * blk_m : (bx + 1) * blk_m, :])

    return main


@tilelang.jit(out_idx=[-1])
def rmsnorm_optimized(
    M: int,
    N: int,
    blk_m: int,
    dtype: str = "float",
    eps: float = 1e-5,
):
    """
    优化的 RMSNorm 算子

    计算: y = x / sqrt(mean(x^2) + eps) * weight

    使用 T.reduce_sum 进行并行归约
    """
    @T.prim_func
    def main(
        X: T.Tensor((M, N), dtype),
        Weight: T.Tensor((N,), dtype),
        Y: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(M, blk_m), threads=128) as bx:
            X_shared = T.alloc_shared((blk_m, N), dtype)
            W_shared = T.alloc_shared((N,), dtype)

            X_local = T.alloc_fragment((blk_m, N), dtype)
            X_pow_local = T.alloc_fragment((blk_m, N), dtype)
            rstd_local = T.alloc_fragment((blk_m,), dtype)

            # 加载数据
            T.copy(X[bx * blk_m : (bx + 1) * blk_m, :], X_shared)
            T.copy(Weight[:], W_shared)
            T.copy(X_shared, X_local)

            # 计算 sum(x^2)
            for i, j in T.Parallel(blk_m, N):
                X_pow_local[i, j] = X_local[i, j] * X_local[i, j]
            T.reduce_sum(X_pow_local, rstd_local, dim=1)

            # 计算 rstd = 1 / sqrt(mean(x^2) + eps)
            for i in T.Parallel(blk_m):
                rstd_local[i] = T.rsqrt(rstd_local[i] / N + eps)

            # 计算 y = x * rstd * weight
            for i, j in T.Parallel(blk_m, N):
                X_local[i, j] = X_local[i, j] * rstd_local[i] * W_shared[j]

            T.copy(X_local, Y[bx * blk_m : (bx + 1) * blk_m, :])

    return main


# === 支持 float16 输入的版本 ===

@tilelang.jit(out_idx=[-1])
def layernorm_fp16(
    M: int,
    N: int,
    blk_m: int,
    eps: float = 1e-5,
):
    """
    支持 float16 输入的 LayerNorm 算子

    内部使用 float32 累加以保证精度
    """
    in_dtype = "float16"
    acc_dtype = "float"

    @T.prim_func
    def main(
        X: T.Tensor((M, N), in_dtype),
        Weight: T.Tensor((N,), in_dtype),
        Bias: T.Tensor((N,), in_dtype),
        Y: T.Tensor((M, N), in_dtype),
    ):
        with T.Kernel(T.ceildiv(M, blk_m), threads=128) as bx:
            X_shared = T.alloc_shared((blk_m, N), in_dtype)
            W_shared = T.alloc_shared((N,), in_dtype)
            B_shared = T.alloc_shared((N,), in_dtype)

            # 使用 float32 进行计算
            X_local = T.alloc_fragment((blk_m, N), acc_dtype)
            X_pow_local = T.alloc_fragment((blk_m, N), acc_dtype)
            mean_local = T.alloc_fragment((blk_m,), acc_dtype)
            var_local = T.alloc_fragment((blk_m,), acc_dtype)
            Y_local = T.alloc_fragment((blk_m, N), acc_dtype)

            # 加载数据
            T.copy(X[bx * blk_m : (bx + 1) * blk_m, :], X_shared)
            T.copy(Weight[:], W_shared)
            T.copy(Bias[:], B_shared)

            # 转换为 float32 并复制
            for i, j in T.Parallel(blk_m, N):
                X_local[i, j] = T.cast(X_shared[i, j], acc_dtype)

            # 计算 mean = sum(x) / N
            T.reduce_sum(X_local, mean_local, dim=1)
            for i in T.Parallel(blk_m):
                mean_local[i] = mean_local[i] / N

            # 计算 var = E[x^2] - E[x]^2
            for i, j in T.Parallel(blk_m, N):
                X_pow_local[i, j] = X_local[i, j] * X_local[i, j]
            T.reduce_sum(X_pow_local, var_local, dim=1)
            for i in T.Parallel(blk_m):
                var_local[i] = var_local[i] / N - mean_local[i] * mean_local[i]
                var_local[i] = T.rsqrt(var_local[i] + eps)

            # 计算 y = (x - mean) * rstd * weight + bias
            for i, j in T.Parallel(blk_m, N):
                Y_local[i, j] = (X_local[i, j] - mean_local[i]) * var_local[i]
                Y_local[i, j] = Y_local[i, j] * T.cast(W_shared[j], acc_dtype) + T.cast(B_shared[j], acc_dtype)

            # 转换回 float16
            for i, j in T.Parallel(blk_m, N):
                X_shared[i, j] = T.cast(Y_local[i, j], in_dtype)

            T.copy(X_shared, Y[bx * blk_m : (bx + 1) * blk_m, :])

    return main


@tilelang.jit(out_idx=[-1])
def rmsnorm_fp16(
    M: int,
    N: int,
    blk_m: int,
    eps: float = 1e-5,
):
    """
    支持 float16 输入的 RMSNorm 算子

    内部使用 float32 累加以保证精度
    """
    in_dtype = "float16"
    acc_dtype = "float"

    @T.prim_func
    def main(
        X: T.Tensor((M, N), in_dtype),
        Weight: T.Tensor((N,), in_dtype),
        Y: T.Tensor((M, N), in_dtype),
    ):
        with T.Kernel(T.ceildiv(M, blk_m), threads=128) as bx:
            X_shared = T.alloc_shared((blk_m, N), in_dtype)
            W_shared = T.alloc_shared((N,), in_dtype)

            X_local = T.alloc_fragment((blk_m, N), acc_dtype)
            X_pow_local = T.alloc_fragment((blk_m, N), acc_dtype)
            rstd_local = T.alloc_fragment((blk_m,), acc_dtype)
            Y_local = T.alloc_fragment((blk_m, N), acc_dtype)

            # 加载数据
            T.copy(X[bx * blk_m : (bx + 1) * blk_m, :], X_shared)
            T.copy(Weight[:], W_shared)

            # 转换为 float32
            for i, j in T.Parallel(blk_m, N):
                X_local[i, j] = T.cast(X_shared[i, j], acc_dtype)

            # 计算 sum(x^2)
            for i, j in T.Parallel(blk_m, N):
                X_pow_local[i, j] = X_local[i, j] * X_local[i, j]
            T.reduce_sum(X_pow_local, rstd_local, dim=1)

            # 计算 rstd = 1 / sqrt(mean(x^2) + eps)
            for i in T.Parallel(blk_m):
                rstd_local[i] = T.rsqrt(rstd_local[i] / N + eps)

            # 计算 y = x * rstd * weight
            for i, j in T.Parallel(blk_m, N):
                Y_local[i, j] = X_local[i, j] * rstd_local[i] * T.cast(W_shared[j], acc_dtype)

            # 转换回 float16
            for i, j in T.Parallel(blk_m, N):
                X_shared[i, j] = T.cast(Y_local[i, j], in_dtype)

            T.copy(X_shared, Y[bx * blk_m : (bx + 1) * blk_m, :])

    return main


# === 动态形状版本 ===

@tilelang.jit(out_idx=[-1])
def layernorm_dynamic(
    N: int,
    blk_m: int = 1,
    dtype: str = "float16",
    eps: float = 1e-5,
):
    """
    动态形状的 LayerNorm 算子

    M 维度动态，N 固定
    """
    M = T.dynamic("m")
    acc_dtype = "float"

    @T.prim_func
    def main(
        X: T.Tensor((M, N), dtype),
        Weight: T.Tensor((N,), dtype),
        Bias: T.Tensor((N,), dtype),
        Y: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(M, blk_m), threads=128) as bx:
            X_shared = T.alloc_shared((blk_m, N), dtype)
            W_shared = T.alloc_shared((N,), dtype)
            B_shared = T.alloc_shared((N,), dtype)

            X_local = T.alloc_fragment((blk_m, N), acc_dtype)
            X_pow_local = T.alloc_fragment((blk_m, N), acc_dtype)
            mean_local = T.alloc_fragment((blk_m,), acc_dtype)
            var_local = T.alloc_fragment((blk_m,), acc_dtype)

            T.copy(X[bx * blk_m : (bx + 1) * blk_m, :], X_shared)
            T.copy(Weight[:], W_shared)
            T.copy(Bias[:], B_shared)

            for i, j in T.Parallel(blk_m, N):
                X_local[i, j] = T.cast(X_shared[i, j], acc_dtype)

            T.reduce_sum(X_local, mean_local, dim=1)
            for i in T.Parallel(blk_m):
                mean_local[i] = mean_local[i] / N

            for i, j in T.Parallel(blk_m, N):
                X_pow_local[i, j] = X_local[i, j] * X_local[i, j]
            T.reduce_sum(X_pow_local, var_local, dim=1)
            for i in T.Parallel(blk_m):
                var_local[i] = var_local[i] / N - mean_local[i] * mean_local[i]
                var_local[i] = T.rsqrt(var_local[i] + eps)

            for i, j in T.Parallel(blk_m, N):
                X_local[i, j] = (X_local[i, j] - mean_local[i]) * var_local[i]
                X_local[i, j] = X_local[i, j] * T.cast(W_shared[j], acc_dtype) + T.cast(B_shared[j], acc_dtype)

            for i, j in T.Parallel(blk_m, N):
                X_shared[i, j] = T.cast(X_local[i, j], dtype)

            T.copy(X_shared, Y[bx * blk_m : (bx + 1) * blk_m, :])

    return main


@tilelang.jit(out_idx=[-1])
def rmsnorm_dynamic(
    N: int,
    blk_m: int = 1,
    dtype: str = "float16",
    eps: float = 1e-5,
):
    """
    动态形状的 RMSNorm 算子

    M 维度动态，N 固定
    """
    M = T.dynamic("m")
    acc_dtype = "float"

    @T.prim_func
    def main(
        X: T.Tensor((M, N), dtype),
        Weight: T.Tensor((N,), dtype),
        Y: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(M, blk_m), threads=128) as bx:
            X_shared = T.alloc_shared((blk_m, N), dtype)
            W_shared = T.alloc_shared((N,), dtype)

            X_local = T.alloc_fragment((blk_m, N), acc_dtype)
            X_pow_local = T.alloc_fragment((blk_m, N), acc_dtype)
            rstd_local = T.alloc_fragment((blk_m,), acc_dtype)

            T.copy(X[bx * blk_m : (bx + 1) * blk_m, :], X_shared)
            T.copy(Weight[:], W_shared)

            for i, j in T.Parallel(blk_m, N):
                X_local[i, j] = T.cast(X_shared[i, j], acc_dtype)

            for i, j in T.Parallel(blk_m, N):
                X_pow_local[i, j] = X_local[i, j] * X_local[i, j]
            T.reduce_sum(X_pow_local, rstd_local, dim=1)

            for i in T.Parallel(blk_m):
                rstd_local[i] = T.rsqrt(rstd_local[i] / N + eps)

            for i, j in T.Parallel(blk_m, N):
                X_local[i, j] = X_local[i, j] * rstd_local[i] * T.cast(W_shared[j], acc_dtype)

            for i, j in T.Parallel(blk_m, N):
                X_shared[i, j] = T.cast(X_local[i, j], dtype)

            T.copy(X_shared, Y[bx * blk_m : (bx + 1) * blk_m, :])

    return main


# === 测试函数 ===

def test_layernorm(M=8192, N=8192, blk_m=1):
    """测试 LayerNorm 算子"""
    import torch

    print(f"Testing LayerNorm: M={M}, N={N}, blk_m={blk_m}")

    kernel = layernorm_optimized(M, N, blk_m)
    profiler = kernel.get_profiler()

    def ref_program(x, w, b):
        return torch.nn.functional.layer_norm(x, (N,), weight=w, bias=b, eps=1e-5)

    profiler.assert_allclose(ref_program, rtol=0.01, atol=0.01)
    print("✓ LayerNorm test passed!")

    latency = profiler.do_bench(ref_program, warmup=100)
    print(f"PyTorch: {latency:.2f} ms")

    latency = profiler.do_bench(warmup=100)
    print(f"TileLang: {latency:.2f} ms")
    print(f"Speedup: {latency_ref / latency:.2f}x" if 'latency_ref' in dir() else "")

    return latency


def test_rmsnorm(M=8192, N=8192, blk_m=1):
    """测试 RMSNorm 算子"""
    import torch

    print(f"\nTesting RMSNorm: M={M}, N={N}, blk_m={blk_m}")

    kernel = rmsnorm_optimized(M, N, blk_m)
    profiler = kernel.get_profiler()

    def ref_program(x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * w

    profiler.assert_allclose(ref_program, rtol=0.01, atol=0.01)
    print("✓ RMSNorm test passed!")

    latency = profiler.do_bench(ref_program, warmup=100)
    print(f"PyTorch: {latency:.2f} ms")

    latency = profiler.do_bench(warmup=100)
    print(f"TileLang: {latency:.2f} ms")


def test_layernorm_fp16(M=8192, N=8192, blk_m=1):
    """测试 float16 LayerNorm"""
    import torch

    print(f"\nTesting LayerNorm FP16: M={M}, N={N}, blk_m={blk_m}")

    kernel = layernorm_fp16(M, N, blk_m)
    profiler = kernel.get_profiler()

    def ref_program(x, w, b):
        return torch.nn.functional.layer_norm(x.float(), (N,), weight=w.float(), bias=b.float(), eps=1e-5).half()

    profiler.assert_allclose(ref_program, rtol=0.01, atol=0.01)
    print("✓ LayerNorm FP16 test passed!")

    latency = profiler.do_bench(ref_program, warmup=100)
    print(f"PyTorch: {latency:.2f} ms")

    latency = profiler.do_bench(warmup=100)
    print(f"TileLang: {latency:.2f} ms")


def benchmark_comparison():
    """性能对比测试"""
    import torch
    import time

    M, N = 8192, 8192

    print("\n" + "=" * 60)
    print(f"Performance Comparison (M={M}, N={N})")
    print("=" * 60)

    # PyTorch LayerNorm
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    w = torch.randn(N, device='cuda', dtype=torch.float32)
    b = torch.randn(N, device='cuda', dtype=torch.float32)

    # Warm up
    for _ in range(10):
        _ = torch.nn.functional.layer_norm(x, (N,), weight=w, bias=b)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = torch.nn.functional.layer_norm(x, (N,), weight=w, bias=b)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / 100 * 1000

    # TileLang
    kernel = layernorm_optimized(M, N, blk_m=1)
    profiler = kernel.get_profiler()
    tilelang_time = profiler.do_bench(warmup=100)

    print(f"\n{'Implementation':<20} {'Latency (ms)':<15} {'Bandwidth (GB/s)':<15}")
    print("-" * 50)
    print(f"{'PyTorch':<20} {pytorch_time:<15.2f} {M*N*4*4/(pytorch_time*1e-3)/1e9:<15.0f}")
    print(f"{'TileLang':<20} {tilelang_time:<15.2f} {M*N*4*4/(tilelang_time*1e-3)/1e9:<15.0f}")
    print(f"\nSpeedup: {pytorch_time/tilelang_time:.2f}x")


def main():
    """主函数"""
    print("=" * 60)
    print("LayerNorm/RMSNorm Optimized Tests")
    print("=" * 60)

    test_layernorm()
    test_rmsnorm()
    test_layernorm_fp16()
    benchmark_comparison()


if __name__ == "__main__":
    main()
