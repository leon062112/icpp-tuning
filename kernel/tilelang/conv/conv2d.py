import csv
import os
import itertools
import argparse
import math
import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T

import sys
from pathlib import Path

# Add project root to path so 'kernel' package is importable
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from kernel.autotuner.interface import TileLangKernelBase, select_best
from kernel.autotuner.cost_model import make_conv2d_spec

torch.backends.cuda.matmul.allow_tf32 = False


def get_conv_op_shape_list():

    CNN_BatchList = [1, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]
    CNN_shape_list_without_batch = [(3, 224, 224, 96, 11, 11, 4, 0),(96, 26, 26, 256, 5, 5, 1, 2),(256, 12, 12, 384, 3, 3, 1, 1),(384, 12, 12, 384, 3, 3, 1, 1),(384, 12, 12, 256, 3, 3, 1, 1),(3, 224, 224, 64, 7, 7, 2, 3),(64, 56, 56, 192, 3, 3, 1, 1),(192, 28, 28, 64, 1, 1, 1, 0),(192, 28, 28, 96, 1, 1, 1, 0),(96, 28, 28, 128, 3, 3, 1, 1),(192, 28, 28, 16, 1, 1, 1, 0),(16, 28, 28, 32, 5, 5, 1, 2),(192, 28, 28, 32, 1, 1, 1, 0),(256, 28, 28, 128, 1, 1, 1, 0),(256, 28, 28, 128, 1, 1, 1, 0),(128, 28, 28, 192, 3, 3, 1, 1),(256, 28, 28, 32, 1, 1, 1, 0),(32, 28, 28, 96, 5, 5, 1, 2),(256, 28, 28, 64, 1, 1, 1, 0),(480, 14, 14, 192, 1, 1, 1, 0),(480, 14, 14, 96, 1, 1, 1, 0),(96, 14, 14, 208, 3, 3, 1, 1),(480, 14, 14, 16, 1, 1, 1, 0),(16, 14, 14, 48, 5, 5, 1, 2),(480, 14, 14, 64, 1, 1, 1, 0),(512, 14, 14, 160, 1, 1, 1, 0),(512, 14, 14, 112, 1, 1, 1, 0),(112, 14, 14, 224, 3, 3, 1, 1),(512, 14, 14, 24, 1, 1, 1, 0),(25, 14, 14, 64, 5, 5, 1, 2),(512, 14, 14, 64, 1, 1, 1, 0),(512, 14, 14, 128, 1, 1, 1, 0),(512, 14, 14, 128, 1, 1, 1, 0),(128, 14, 14, 256, 3, 3, 1, 1),(512, 14, 14, 24, 1, 1, 1, 0),(24, 14, 14, 64, 5, 5, 1, 2),(512, 14, 14, 64, 1, 1, 1, 0),(512, 14, 14, 112, 1, 1, 1, 0),(512, 14, 14, 144, 1, 1, 1, 0),(144, 14, 14, 288, 3, 3, 1, 1),(512, 14, 14, 32, 1, 1, 1, 0),(32, 14, 14, 64, 5, 5, 1, 2),(512, 14, 14, 64, 1, 1, 1, 0),(528, 14, 14, 256, 1, 1, 1, 0),(528, 14, 14, 160, 1, 1, 1, 0),(160, 14, 14, 320, 3, 3, 1, 1),(528, 14, 14, 32, 1, 1, 1, 0),(32, 14, 14, 128, 5, 5, 1, 2),(528, 14, 14, 128, 1, 1, 1, 0),(832, 7, 7, 256, 1, 1, 1, 0),(832, 7, 7, 160, 1, 1, 1, 0),(832, 7, 7, 320, 3, 3, 1, 1),(320, 7, 7, 32, 1, 1, 1, 0),(32, 7, 7, 128, 5, 5, 1, 2),(832, 7, 7, 128, 1, 1, 1, 0),(832, 7, 7, 384, 1, 1, 1, 0),(832, 7, 7, 192, 1, 1, 1, 0),(192, 7, 7, 384, 3, 3, 1, 1),(832, 7, 7, 48, 1, 1, 1, 0),(48, 7, 7, 128, 5, 5, 1, 2),(832, 7, 7, 128, 1, 1, 1, 0),(3, 224, 224, 64, 3, 3, 1, 1),(64, 224, 224, 64, 3, 3, 1, 1),(64, 112, 112, 128, 3, 3, 1, 1),(128, 112, 112, 128, 3, 3, 1, 1),(128, 56, 56, 256, 3, 3, 1, 1),(256, 56, 56, 256, 3, 3, 1, 1),(256, 56, 56, 256, 3, 3, 1, 1),(256, 56, 56, 256, 3, 3, 1, 1),(256, 28, 28, 512, 3, 3, 1, 1),(512, 28, 28, 512, 3, 3, 1, 1),(512, 28, 28, 512, 3, 3, 1, 1),(512, 28, 28, 512, 3, 3, 1, 1),(512, 14, 14, 512, 3, 3, 1, 1),(512, 14, 14, 512, 3, 3, 1, 1),(512, 14, 14, 512, 3, 3, 1, 1),(512, 14, 14, 512, 3, 3, 1, 1),(3, 224, 224, 64, 3, 3, 1, 1),(64, 112, 112, 64, 3, 3, 1, 1),(64, 112, 112, 64, 3, 3, 1, 1),(64, 112, 112, 64, 3, 3, 1, 1),(64, 112, 112, 64, 3, 3, 1, 1),(64, 112, 112, 128, 3, 3, 2, 1),(128, 56, 56, 128, 3, 3, 1, 1),(64, 112, 112, 128, 1, 1, 2, 0),(128, 56, 56, 128, 3, 3, 1, 1),(128, 56, 56, 128, 3, 3, 1, 1),(128, 56, 56, 256, 3, 3, 2, 1),(256, 28, 28, 256, 3, 3, 1, 1),(128, 56, 56, 256, 1, 1, 2, 0),(256, 28, 28, 256, 3, 3, 1, 1),(256, 28, 28, 256, 3, 3, 1, 1),(256, 28, 28, 512, 3, 3, 2, 1),(512, 28, 28, 512, 3, 3, 1, 1),(256, 28, 28, 512, 1, 1, 2, 0),(512, 14, 14, 512, 3, 3, 1, 1),(512, 14, 14, 512, 3, 3, 1, 1)]
    CNN_shape_list = [(batch,) + shape for batch in CNN_BatchList for shape in CNN_shape_list_without_batch]
    DeepBench_shape_list = [(1, 1, 161, 700, 32, 5, 20, 2, 0),(2, 1, 161, 700, 32, 5, 20, 2, 0),(4, 1, 161, 700, 32, 5, 20, 2, 0),(1, 32, 79, 341, 32, 5, 10, 2, 0),(2, 32, 79, 341, 32, 5, 10, 2, 0),(4, 32, 79, 341, 32, 5, 10, 2, 0),(1, 1, 48, 480, 16, 3, 3, 1, 1),(1, 16, 24, 240, 32, 3, 3, 1, 1),(1, 32, 12, 120, 64, 3, 3, 1, 1),(1, 64, 6, 60, 128, 3, 3, 1, 1),(1, 3, 108, 108, 64, 3, 3, 2, 1),(1, 64, 54, 54, 64, 3, 3, 1, 1),(1, 128, 27, 27, 128, 3, 3, 1, 1),(1, 128, 14, 14, 256, 3, 3, 1, 1),(1, 256, 7, 7, 512, 3, 3, 1, 1),(1, 3, 224, 224, 64, 3, 3, 1, 1),(1, 64, 112, 112, 128, 3, 3, 1, 1),(1, 128, 56, 56, 256, 3, 3, 1, 1),(1, 256, 28, 28, 512, 3, 3, 1, 1),(1, 512, 14, 14, 512, 3, 3, 1, 1),(1, 512, 7, 7, 512, 3, 3, 1, 1),(2, 3, 224, 224, 64, 3, 3, 1, 1),(2, 64, 112, 112, 128, 3, 3, 1, 1),(2, 128, 56, 56, 256, 3, 3, 1, 1),(2, 256, 28, 28, 512, 3, 3, 1, 1),(2, 512, 14, 14, 512, 3, 3, 1, 1),(2, 512, 7, 7, 512, 3, 3, 1, 1),(1, 3, 224, 224, 64, 7, 7, 2, 3),(1, 192, 28, 28, 32, 5, 5, 1, 2),(1, 192, 28, 28, 64, 1, 1, 1, 0),(1, 512, 14, 14, 48, 5, 5, 1, 2),(1, 512, 14, 14, 192, 1, 1, 1, 0),(1, 832, 7, 7, 256, 1, 1, 1, 0),(1, 832, 7, 7, 128, 5, 5, 1, 2),(2, 3, 224, 224, 64, 7, 7, 2, 3),(2, 192, 28, 28, 32, 5, 5, 1, 2),(2, 192, 28, 28, 64, 1, 1, 1, 0),(2, 512, 14, 14, 48, 5, 5, 1, 2),(2, 512, 14, 14, 192, 1, 1, 1, 0),(2, 832, 7, 7, 256, 1, 1, 1, 0),(2, 832, 7, 7, 128, 5, 5, 1, 2),(1, 64, 56, 56, 64, 3, 3, 1, 1),(1, 64, 56, 56, 256, 1, 1, 2, 0),(1, 128, 28, 28, 128, 3, 3, 1, 1),(1, 128, 28, 28, 512, 1, 1, 2, 0),(1, 256, 14, 14, 256, 1, 1, 1, 0),(1, 256, 14, 14, 256, 3, 3, 1, 1),(1, 256, 14, 14, 1024, 1, 1, 2, 0),(1, 512, 7, 7, 512, 1, 1, 1, 0),(1, 2048, 7, 7, 512, 1, 1, 2, 3),(2, 64, 56, 56, 64, 3, 3, 1, 1),(2, 64, 56, 56, 256, 1, 1, 2, 0),(2, 128, 28, 28, 128, 3, 3, 1, 1),(2, 128, 28, 28, 512, 1, 1, 2, 0),(2, 256, 14, 14, 256, 1, 1, 1, 0),(2, 256, 14, 14, 256, 3, 3, 1, 1),(2, 256, 14, 14, 1024, 1, 1, 2, 0),(2, 512, 7, 7, 512, 1, 1, 1, 0),(2, 2048, 7, 7, 512, 1, 1, 2, 3),(1, 1, 161, 700, 64, 5, 5, 2, 1),(1, 64, 80, 350, 64, 3, 3, 1, 1),(1, 64, 80, 350, 128, 5, 5, 2, 1),(1, 128, 40, 175, 128, 3, 3, 1, 1),(1, 128, 40, 175, 256, 5, 5, 2, 1),(1, 256, 20, 84, 256, 3, 3, 1, 1),(1, 256, 20, 84, 512, 5, 5, 2, 1),(1, 512, 10, 42, 512, 3, 3, 1, 1),(2, 1, 161, 700, 64, 5, 5, 2, 1),(2, 64, 80, 350, 64, 3, 3, 1, 1),(2, 64, 80, 350, 128, 5, 5, 2, 1),(2, 128, 40, 175, 128, 3, 3, 1, 1),(2, 128, 40, 175, 256, 5, 5, 2, 1),(2, 256, 20, 84, 256, 3, 3, 1, 1),(2, 256, 20, 84, 512, 5, 5, 2, 1),(2, 512, 10, 42, 512, 3, 3, 1, 1),(1, 64, 112, 112, 64, 1, 1, 1, 0),(1, 64, 56, 56, 256, 1, 1, 1, 0),(1, 256, 56, 56, 64, 1, 1, 1, 0),(1, 256, 56, 56, 128, 1, 1, 2, 0),(1, 128, 28, 28, 512, 1, 1, 1, 0),(1, 512, 28, 28, 128, 1, 1, 1, 0),(1, 512, 28, 28, 256, 1, 1, 2, 0),(1, 256, 14, 14, 1024, 1, 1, 1, 0),(1, 512, 28, 28, 1024, 1, 1, 2, 0),(1, 1024, 14, 14, 256, 1, 1, 1, 0),(1, 256, 14, 14, 1024, 1, 1, 1, 0),(1, 1024, 14, 14, 512, 1, 1, 2, 0),(1, 512, 7, 7, 512, 3, 3, 1, 1),(1, 512, 7, 7, 2048, 1, 1, 1, 0),(1, 1024, 14, 14, 2048, 1, 1, 2, 0),(1, 2048, 7, 7, 512, 1, 1, 1, 0),(2, 64, 112, 112, 64, 1, 1, 1, 0),(2, 64, 56, 56, 256, 1, 1, 1, 0),(2, 256, 56, 56, 64, 1, 1, 1, 0),(2, 256, 56, 56, 128, 1, 1, 2, 0),(2, 128, 28, 28, 512, 1, 1, 1, 0),(2, 512, 28, 28, 128, 1, 1, 1, 0),(2, 512, 28, 28, 256, 1, 1, 2, 0),(2, 256, 14, 14, 1024, 1, 1, 1, 0),(2, 512, 28, 28, 1024, 1, 1, 2, 0),(2, 1024, 14, 14, 256, 1, 1, 1, 0),(2, 256, 14, 14, 1024, 1, 1, 1, 0),(2, 1024, 14, 14, 512, 1, 1, 2, 0),(2, 512, 7, 7, 512, 3, 3, 1, 1),(2, 512, 7, 7, 2048, 1, 1, 1, 0),(2, 1024, 14, 14, 2048, 1, 1, 2, 0),(2, 2048, 7, 7, 512, 1, 1, 1, 0)]

    return CNN_shape_list + DeepBench_shape_list


def get_ref_program(stride=1, padding=1, dilation=1, use_compile=False):
    def conv2d_core(X_nchw, W_oihw):
        return F.conv2d(X_nchw, W_oihw, stride=stride, padding=padding, dilation=dilation)

    compiled_conv2d_core = None
    if use_compile and hasattr(torch, "compile"):
        try:
            compiled_conv2d_core = torch.compile(conv2d_core)
        except Exception:
            compiled_conv2d_core = None

    def ref_program(X, W, Y):
        # Keep inputs/weights in fp16 for a mixed-precision path.
        nonlocal compiled_conv2d_core
        X_nchw = X.permute(0, 3, 1, 2)
        W_oihw = W.permute(3, 2, 0, 1)

        if compiled_conv2d_core is not None:
            try:
                Y_nchw = compiled_conv2d_core(X_nchw, W_oihw)
            except Exception:
                compiled_conv2d_core = None
                Y_nchw = conv2d_core(X_nchw, W_oihw)
        else:
            Y_nchw = conv2d_core(X_nchw, W_oihw)

        Y[...] = Y_nchw.permute(0, 2, 3, 1).to(Y.dtype)
    return ref_program


def calc_oh_ow(H, W, KH, KW, S, P, D):
    OH = (H + 2 * P - D * (KH - 1) - 1) // S + 1
    OW = (W + 2 * P - D * (KW - 1) - 1) // S + 1
    return OH, OW


def calc_conv_flops(N, C, OH, OW, F_out, KH, KW):
    return 2 * N * OH * OW * F_out * C * KH * KW


def reduce_shapes_preserve_scale_distribution(shapes, target_count=200, num_bins=10):
    if target_count <= 0 or len(shapes) <= target_count:
        return list(shapes)

    flops = []
    for shape in shapes:
        N, C, H, W, F_out, KH, KW, S, P = shape
        D = 1
        OH, OW = calc_oh_ow(H, W, KH, KW, S, P, D)
        flops.append(calc_conv_flops(N, C, OH, OW, F_out, KH, KW))

    logs = [math.log10(max(v, 1)) for v in flops]
    log_min = min(logs)
    log_max = max(logs)
    if log_max == log_min:
        step = 1.0
    else:
        step = (log_max - log_min) / max(num_bins, 1)

    bins = [[] for _ in range(num_bins)]
    for idx, lv in enumerate(logs):
        if log_max == log_min:
            bid = 0
        else:
            bid = int((lv - log_min) / step)
            if bid >= num_bins:
                bid = num_bins - 1
        bins[bid].append(idx)

    non_empty_bins = [b for b in bins if b]
    if not non_empty_bins:
        return list(shapes)[:target_count]

    desired = [len(b) * target_count / len(shapes) for b in bins]
    alloc = [int(x) for x in desired]

    if target_count >= len(non_empty_bins):
        for i, b in enumerate(bins):
            if b and alloc[i] == 0:
                alloc[i] = 1

    total_alloc = sum(alloc)
    if total_alloc > target_count:
        order = sorted(
            range(num_bins),
            key=lambda i: (desired[i] - alloc[i], -alloc[i])
        )
        for i in order:
            if total_alloc == target_count:
                break
            if alloc[i] > 0:
                alloc[i] -= 1
                total_alloc -= 1
    elif total_alloc < target_count:
        order = sorted(
            range(num_bins),
            key=lambda i: (desired[i] - alloc[i], len(bins[i])),
            reverse=True
        )
        ptr = 0
        while total_alloc < target_count and order:
            i = order[ptr % len(order)]
            if alloc[i] < len(bins[i]):
                alloc[i] += 1
                total_alloc += 1
            ptr += 1
            if ptr > target_count * max(1, num_bins):
                break

    selected = []
    for i, b in enumerate(bins):
        k = min(alloc[i], len(b))
        if k <= 0:
            continue
        if k == len(b):
            selected.extend(b)
            continue
        # Evenly sample in each scale bin to keep intra-bin coverage.
        for j in range(k):
            pos = int(j * len(b) / k)
            selected.append(b[pos])

    selected = sorted(set(selected))
    if len(selected) > target_count:
        selected = selected[:target_count]
    elif len(selected) < target_count:
        selected_set = set(selected)
        for idx in range(len(shapes)):
            if idx not in selected_set:
                selected.append(idx)
                if len(selected) == target_count:
                    break

    return [shapes[i] for i in selected]


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


def get_configs():
    block_M = [16, 32, 64]
    block_N = [64, 128, 256]
    block_K = [32, 64]
    num_stages = [1, 2]
    thread_num = [128, 256]

    configs = []
    for bM, bN, bK, ns, tn in itertools.product(
        block_M, block_N, block_K, num_stages, thread_num,
    ):
            
        configs.append({
            "block_M": bM,
            "block_N": bN,
            "block_K": bK,
            "num_stages": ns,
            "thread_num": tn,
        })
    return configs


def make_conv2d_kernel(
    N, C, H, W, F_out, KH, KW, S=1, P=1, D=1
):
    OH, OW = calc_oh_ow(H, W, KH, KW, S, P, D)
    M_eff = N * OH * OW
    K_eff = KH * KW * C

    def kernel(block_M=None, block_N=None, block_K=None,
               num_stages=None, thread_num=None):
        dtype = T.float16
        accum_dtype = T.float32
        out_dtype = T.float16

        @T.prim_func
        def conv_kernel(
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
                        
                        in_bound = (access_h >= 0) and (access_w >= 0) and (access_h < H) and (access_w < W) and (m < M_eff) and (k < K_eff)
                        
                        val = data[m // (OH * OW), safe_h, safe_w, k % C]
                        data_shared[i, j] = T.if_then_else(in_bound, val, T.cast(0, dtype))
                        
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
                
                if M_eff % block_M == 0 and F_out % block_N == 0:
                    T.copy(out_shared, out_flat[by * block_M, bx * block_N])
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        m_idx = by * block_M + i
                        n_idx = bx * block_N + j
                        if m_idx < M_eff and n_idx < F_out:
                            out_flat[m_idx, n_idx] = out_shared[i, j]
                
        return conv_kernel

    return kernel


def build_conv2d_kernel(N, C, H, W, F_out, KH, KW, S=1, P=1, D=1, config=None):
    """Build a compiled Conv2d kernel using cost-model selection or a given config."""
    OH, OW = calc_oh_ow(H, W, KH, KW, S, P, D)
    if config is None:
        config, _ = select_best(DESCRIPTOR, N=N, C=C, H=H, W=W, OC=F_out,
                                KH=KH, KW=KW, stride=S, padding=P)
    factory = make_conv2d_kernel(N, C, H, W, F_out, KH, KW, S, P, D)
    prim_func = factory(**config)
    return tilelang.compile(prim_func, target="auto")


def verify_correctness(kernel, ref_prog, N, C, H, W, F_out, KH, KW, S, P, D):
    """
    修改为返回布尔值：
    返回 True 代表验证通过；False 代表存在数值错误或越界。
    """
    OH, OW = calc_oh_ow(H, W, KH, KW, S, P, D)

    X = torch.randn(N, H, W, C, dtype=torch.float16, device='cuda')
    W_weight = torch.randn(KH, KW, C, F_out, dtype=torch.float16, device='cuda')
    
    Y_ref = torch.zeros(N, OH, OW, F_out, dtype=torch.float16, device='cuda')
    Y_tl = torch.zeros(N, OH, OW, F_out, dtype=torch.float16, device='cuda')

    ref_prog(X, W_weight, Y_ref)
    kernel(X, W_weight, Y_tl)

    try:
        torch.testing.assert_close(Y_tl, Y_ref, rtol=1e-2, atol=1e-2)
        return True
    except AssertionError as e:
        print(f"  [Verification] Status: FAILED ❌ -> {str(e)[:100]}...")
        return False


def benchmark_single_backend(backend, X, W_weight, N, C, H, W, F_out, KH, KW, S, P, D, warmup, rep):
    OH, OW = calc_oh_ow(H, W, KH, KW, S, P, D)
    Y = torch.zeros(N, OH, OW, F_out, dtype=torch.float16, device="cuda")
    total_flops = calc_conv_flops(N, C, OH, OW, F_out, KH, KW)

    if backend == "torch_eager":
        prog = get_ref_program(S, P, D, use_compile=False)
        lat_ms = do_bench_ms(prog, X, W_weight, Y, warmup=warmup, rep=rep)
        return {
            "backend": backend,
            "best_config": "N/A",
            "is_correct": "N/A",
            "lat_ms": lat_ms,
            "tflops": total_flops / lat_ms * 1e-9,
            "status": "OK",
        }

    if backend == "torch_compile":
        prog = get_ref_program(S, P, D, use_compile=True)
        prog(X, W_weight, Y)
        torch.cuda.synchronize()
        lat_ms = do_bench_ms(prog, X, W_weight, Y, warmup=warmup, rep=rep)
        return {
            "backend": backend,
            "best_config": "N/A",
            "is_correct": "N/A",
            "lat_ms": lat_ms,
            "tflops": total_flops / lat_ms * 1e-9,
            "status": "OK",
        }

    if backend == "tilelang":
        kernel = build_conv2d_kernel(N, C, H, W, F_out, KH, KW, S, P, D)
        best_config, _ = select_best(DESCRIPTOR, N=N, C=C, H=H, W=W, OC=F_out,
                                     KH=KH, KW=KW, stride=S, padding=P)
        ref_prog = get_ref_program(S, P, D, use_compile=False)
        is_correct = verify_correctness(kernel, ref_prog, N, C, H, W, F_out, KH, KW, S, P, D)
        lat_ms = do_bench_ms(kernel, X, W_weight, Y, warmup=warmup, rep=rep)
        return {
            "backend": backend,
            "best_config": str(best_config),
            "is_correct": "True" if is_correct else "False",
            "lat_ms": lat_ms,
            "tflops": total_flops / lat_ms * 1e-9,
            "status": "OK",
        }

    raise ValueError(f"Unsupported backend: {backend}")


def parse_args():
    parser = argparse.ArgumentParser(description="Conv2d benchmark for torch eager / torch.compile / tilelang")
    parser.add_argument(
        "--backend",
        type=str,
        default="all",
        choices=["all", "torch_eager", "torch_compile", "tilelang"],
        help="Which backend to benchmark.",
    )
    parser.add_argument("--csv", type=str, default="conv2d_benchmark_results.csv", help="Output CSV path.")
    parser.add_argument("--limit", type=int, default=0, help="Only run first N shapes. 0 means run all.")
    parser.add_argument(
        "--target-shapes",
        type=int,
        default=200,
        help="Reduce shape count while preserving FLOPs scale distribution. 0 means no reduction.",
    )
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations for timing.")
    parser.add_argument("--rep", type=int, default=100, help="Timing iterations.")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    all_shapes = get_conv_op_shape_list()
    print(f"Total shapes to benchmark: {len(all_shapes)}")

    base_shapes = all_shapes[:args.limit] if args.limit > 0 else all_shapes
    shapes_to_test = reduce_shapes_preserve_scale_distribution(
        base_shapes, target_count=args.target_shapes
    )
    backends = ["torch_eager", "torch_compile", "tilelang"] if args.backend == "all" else [args.backend]

    csv_filename = args.csv
    os.makedirs(os.path.dirname(csv_filename) or ".", exist_ok=True)
    print(f"Shapes selected for run: {len(shapes_to_test)}")
    print(f"Backends to run: {backends}")
    print(f"Results will be saved to: {csv_filename}\n")

    with open(csv_filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Index", "N", "C", "H", "W", "F_out", "Kernel_H", "Kernel_W", "Stride", "Pad", "Dilation",
            "Backend", "Best_Config", "Is_Correct", "Latency_ms", "TFlops", "Status"
        ])

    for idx, shape in enumerate(shapes_to_test):
        N, C, H, W, F_out, KH, KW, S, P = shape
        D = 1
        print(f"[{idx+1}/{len(shapes_to_test)}] N={N}, C={C}, H={H}, W={W}, F={F_out}, Kernel={KH}x{KW}, Stride={S}, Pad={P}")

        X = torch.randn(N, H, W, C, dtype=torch.float16, device="cuda")
        W_weight = torch.randn(KH, KW, C, F_out, dtype=torch.float16, device="cuda")

        for backend in backends:
            print(f"  Running backend: {backend}")
            try:
                out = benchmark_single_backend(
                    backend=backend,
                    X=X,
                    W_weight=W_weight,
                    N=N,
                    C=C,
                    H=H,
                    W=W,
                    F_out=F_out,
                    KH=KH,
                    KW=KW,
                    S=S,
                    P=P,
                    D=D,
                    warmup=args.warmup,
                    rep=args.rep,
                )
                print(f"    Latency: {out['lat_ms']:.4f} ms, TFLOPS: {out['tflops']:.2f}")
                if backend == "tilelang":
                    print(f"    Best config: {out['best_config']}, Is_Correct: {out['is_correct']}")

                with open(csv_filename, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        idx + 1, N, C, H, W, F_out, KH, KW, S, P, D,
                        backend, out["best_config"], out["is_correct"],
                        f"{out['lat_ms']:.4f}", f"{out['tflops']:.2f}", out["status"]
                    ])
            except Exception as e:
                print(f"    [Error] backend={backend} failed: {str(e)}")
                with open(csv_filename, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        idx + 1, N, C, H, W, F_out, KH, KW, S, P, D,
                        backend, f"ERROR: {str(e)}", "Error", "N/A", "N/A", "Error"
                    ])

# ---------------------------------------------------------------------------
# Unified autotuner descriptor
# ---------------------------------------------------------------------------


class Conv2dDescriptor(TileLangKernelBase):
    @property
    def name(self):
        return "conv2d"

    def make_op_spec(self, N, C, H, W, OC, KH, KW, stride=1, padding=0, **kw):
        return make_conv2d_spec(N, C, H, W, OC, KH, KW, stride=stride, padding=padding)

    def get_raw_configs(self, **kw):
        return get_configs()


DESCRIPTOR = Conv2dDescriptor()


if __name__ == "__main__":
    main()
