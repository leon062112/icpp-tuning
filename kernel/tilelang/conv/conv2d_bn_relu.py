import argparse
import csv
import itertools
import math
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

torch.backends.cuda.matmul.allow_tf32 = False

def get_conv_op_shape_list():
    """Get CNN and DeepBench convolution shapes."""
    CNN_BatchList = [1, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]
    CNN_shape_list_without_batch = [(3, 224, 224, 96, 11, 11, 4, 0),(96, 26, 26, 256, 5, 5, 1, 2),(256, 12, 12, 384, 3, 3, 1, 1),(384, 12, 12, 384, 3, 3, 1, 1),(384, 12, 12, 256, 3, 3, 1, 1),(3, 224, 224, 64, 7, 7, 2, 3),(64, 56, 56, 192, 3, 3, 1, 1),(192, 28, 28, 64, 1, 1, 1, 0),(192, 28, 28, 96, 1, 1, 1, 0),(96, 28, 28, 128, 3, 3, 1, 1),(192, 28, 28, 16, 1, 1, 1, 0),(16, 28, 28, 32, 5, 5, 1, 2),(192, 28, 28, 32, 1, 1, 1, 0),(256, 28, 28, 128, 1, 1, 1, 0),(256, 28, 28, 128, 1, 1, 1, 0),(128, 28, 28, 192, 3, 3, 1, 1),(256, 28, 28, 32, 1, 1, 1, 0),(32, 28, 28, 96, 5, 5, 1, 2),(256, 28, 28, 64, 1, 1, 1, 0),(480, 14, 14, 192, 1, 1, 1, 0),(480, 14, 14, 96, 1, 1, 1, 0),(96, 14, 14, 208, 3, 3, 1, 1),(480, 14, 14, 16, 1, 1, 1, 0),(16, 14, 14, 48, 5, 5, 1, 2),(480, 14, 14, 64, 1, 1, 1, 0),(512, 14, 14, 160, 1, 1, 1, 0),(512, 14, 14, 112, 1, 1, 1, 0),(112, 14, 14, 224, 3, 3, 1, 1),(512, 14, 14, 24, 1, 1, 1, 0),(25, 14, 14, 64, 5, 5, 1, 2),(512, 14, 14, 64, 1, 1, 1, 0),(512, 14, 14, 128, 1, 1, 1, 0),(512, 14, 14, 128, 1, 1, 1, 0),(128, 14, 14, 256, 3, 3, 1, 1),(512, 14, 14, 24, 1, 1, 1, 0),(24, 14, 14, 64, 5, 5, 1, 2),(512, 14, 14, 64, 1, 1, 1, 0),(512, 14, 14, 112, 1, 1, 1, 0),(512, 14, 14, 144, 1, 1, 1, 0),(144, 14, 14, 288, 3, 3, 1, 1),(512, 14, 14, 32, 1, 1, 1, 0),(32, 14, 14, 64, 5, 5, 1, 2),(512, 14, 14, 64, 1, 1, 1, 0),(528, 14, 14, 256, 1, 1, 1, 0),(528, 14, 14, 160, 1, 1, 1, 0),(160, 14, 14, 320, 3, 3, 1, 1),(528, 14, 14, 32, 1, 1, 1, 0),(32, 14, 14, 128, 5, 5, 1, 2),(528, 14, 14, 128, 1, 1, 1, 0),(832, 7, 7, 256, 1, 1, 1, 0),(832, 7, 7, 160, 1, 1, 1, 0),(832, 7, 7, 320, 3, 3, 1, 1),(320, 7, 7, 32, 1, 1, 1, 0),(32, 7, 7, 128, 5, 5, 1, 2),(832, 7, 7, 128, 1, 1, 1, 0),(832, 7, 7, 384, 1, 1, 1, 0),(832, 7, 7, 192, 1, 1, 1, 0),(192, 7, 7, 384, 3, 3, 1, 1),(832, 7, 7, 48, 1, 1, 1, 0),(48, 7, 7, 128, 5, 5, 1, 2),(832, 7, 7, 128, 1, 1, 1, 0),(3, 224, 224, 64, 3, 3, 1, 1),(64, 224, 224, 64, 3, 3, 1, 1),(64, 112, 112, 128, 3, 3, 1, 1),(128, 112, 112, 128, 3, 3, 1, 1),(128, 56, 56, 256, 3, 3, 1, 1),(256, 56, 56, 256, 3, 3, 1, 1),(256, 56, 56, 256, 3, 3, 1, 1),(256, 56, 56, 256, 3, 3, 1, 1),(256, 28, 28, 512, 3, 3, 1, 1),(512, 28, 28, 512, 3, 3, 1, 1),(512, 28, 28, 512, 3, 3, 1, 1),(512, 28, 28, 512, 3, 3, 1, 1),(512, 14, 14, 512, 3, 3, 1, 1),(512, 14, 14, 512, 3, 3, 1, 1),(512, 14, 14, 512, 3, 3, 1, 1),(512, 14, 14, 512, 3, 3, 1, 1),(3, 224, 224, 64, 3, 3, 1, 1),(64, 112, 112, 64, 3, 3, 1, 1),(64, 112, 112, 64, 3, 3, 1, 1),(64, 112, 112, 64, 3, 3, 1, 1),(64, 112, 112, 64, 3, 3, 1, 1),(64, 112, 112, 128, 3, 3, 2, 1),(128, 56, 56, 128, 3, 3, 1, 1),(64, 112, 112, 128, 1, 1, 2, 0),(128, 56, 56, 128, 3, 3, 1, 1),(128, 56, 56, 128, 3, 3, 1, 1),(128, 56, 56, 256, 3, 3, 2, 1),(256, 28, 28, 256, 3, 3, 1, 1),(128, 56, 56, 256, 1, 1, 2, 0),(256, 28, 28, 256, 3, 3, 1, 1),(256, 28, 28, 256, 3, 3, 1, 1),(256, 28, 28, 512, 3, 3, 2, 1),(512, 28, 28, 512, 3, 3, 1, 1),(256, 28, 28, 512, 1, 1, 2, 0),(512, 14, 14, 512, 3, 3, 1, 1),(512, 14, 14, 512, 3, 3, 1, 1)]
    CNN_shape_list = [(batch,) + shape for batch in CNN_BatchList for shape in CNN_shape_list_without_batch]
    DeepBench_shape_list = [(1, 1, 161, 700, 32, 5, 20, 2, 0),(2, 1, 161, 700, 32, 5, 20, 2, 0),(4, 1, 161, 700, 32, 5, 20, 2, 0),(1, 32, 79, 341, 32, 5, 10, 2, 0),(2, 32, 79, 341, 32, 5, 10, 2, 0),(4, 32, 79, 341, 32, 5, 10, 2, 0),(1, 1, 48, 480, 16, 3, 3, 1, 1),(1, 16, 24, 240, 32, 3, 3, 1, 1),(1, 32, 12, 120, 64, 3, 3, 1, 1),(1, 64, 6, 60, 128, 3, 3, 1, 1),(1, 3, 108, 108, 64, 3, 3, 2, 1),(1, 64, 54, 54, 64, 3, 3, 1, 1),(1, 128, 27, 27, 128, 3, 3, 1, 1),(1, 128, 14, 14, 256, 3, 3, 1, 1),(1, 256, 7, 7, 512, 3, 3, 1, 1),(1, 3, 224, 224, 64, 3, 3, 1, 1),(1, 64, 112, 112, 128, 3, 3, 1, 1),(1, 128, 56, 56, 256, 3, 3, 1, 1),(1, 256, 28, 28, 512, 3, 3, 1, 1),(1, 512, 14, 14, 512, 3, 3, 1, 1),(1, 512, 7, 7, 512, 3, 3, 1, 1),(2, 3, 224, 224, 64, 3, 3, 1, 1),(2, 64, 112, 112, 128, 3, 3, 1, 1),(2, 128, 56, 56, 256, 3, 3, 1, 1),(2, 256, 28, 28, 512, 3, 3, 1, 1),(2, 512, 14, 14, 512, 3, 3, 1, 1),(2, 512, 7, 7, 512, 3, 3, 1, 1),(1, 3, 224, 224, 64, 7, 7, 2, 3),(1, 192, 28, 28, 32, 5, 5, 1, 2),(1, 192, 28, 28, 64, 1, 1, 1, 0),(1, 512, 14, 14, 48, 5, 5, 1, 2),(1, 512, 14, 14, 192, 1, 1, 1, 0),(1, 832, 7, 7, 256, 1, 1, 1, 0),(1, 832, 7, 7, 128, 5, 5, 1, 2),(2, 3, 224, 224, 64, 7, 7, 2, 3),(2, 192, 28, 28, 32, 5, 5, 1, 2),(2, 192, 28, 28, 64, 1, 1, 1, 0),(2, 512, 14, 14, 48, 5, 5, 1, 2),(2, 512, 14, 14, 192, 1, 1, 1, 0),(2, 832, 7, 7, 256, 1, 1, 1, 0),(2, 832, 7, 7, 128, 5, 5, 1, 2),(1, 64, 56, 56, 64, 3, 3, 1, 1),(1, 64, 56, 56, 256, 1, 1, 2, 0),(1, 128, 28, 28, 128, 3, 3, 1, 1),(1, 128, 28, 28, 512, 1, 1, 2, 0),(1, 256, 14, 14, 256, 1, 1, 1, 0),(1, 256, 14, 14, 256, 3, 3, 1, 1),(1, 256, 14, 14, 1024, 1, 1, 2, 0),(1, 512, 7, 7, 512, 1, 1, 1, 0),(1, 2048, 7, 7, 512, 1, 1, 2, 3),(2, 64, 56, 56, 64, 3, 3, 1, 1),(2, 64, 56, 56, 256, 1, 1, 2, 0),(2, 128, 28, 28, 128, 3, 3, 1, 1),(2, 128, 28, 28, 512, 1, 1, 2, 0),(2, 256, 14, 14, 256, 1, 1, 1, 0),(2, 256, 14, 14, 256, 3, 3, 1, 1),(2, 256, 14, 14, 1024, 1, 1, 2, 0),(2, 512, 7, 7, 512, 1, 1, 1, 0),(2, 2048, 7, 7, 512, 1, 1, 2, 3),(1, 1, 161, 700, 64, 5, 5, 2, 1),(1, 64, 80, 350, 64, 3, 3, 1, 1),(1, 64, 80, 350, 128, 5, 5, 2, 1),(1, 128, 40, 175, 128, 3, 3, 1, 1),(1, 128, 40, 175, 256, 5, 5, 2, 1),(1, 256, 20, 84, 256, 3, 3, 1, 1),(1, 256, 20, 84, 512, 5, 5, 2, 1),(1, 512, 10, 42, 512, 3, 3, 1, 1),(2, 1, 161, 700, 64, 5, 5, 2, 1),(2, 64, 80, 350, 64, 3, 3, 1, 1),(2, 64, 80, 350, 128, 5, 5, 2, 1),(2, 128, 40, 175, 128, 3, 3, 1, 1),(2, 128, 40, 175, 256, 5, 5, 2, 1),(2, 256, 20, 84, 256, 3, 3, 1, 1),(2, 256, 20, 84, 512, 5, 5, 2, 1),(2, 512, 10, 42, 512, 3, 3, 1, 1)]

    return CNN_shape_list + DeepBench_shape_list


def _calc_oh_ow(h, w, kh, kw, s, p, d=1):
    oh = (h + 2 * p - d * (kh - 1) - 1) // s + 1
    ow = (w + 2 * p - d * (kw - 1) - 1) // s + 1
    return oh, ow


def _calc_conv_flops(n, c, oh, ow, f_out, kh, kw):
    return 2 * n * c * oh * ow * f_out * kh * kw


def get_sampled_conv_shapes(shapes, target_count=200, num_bins=10):
    """Deterministic FLOPs-scale stratified sampling."""
    if target_count <= 0 or len(shapes) <= target_count:
        return list(shapes)

    flops = []
    for n, c, h, w, f_out, kh, kw, s, p in shapes:
        oh, ow = _calc_oh_ow(h, w, kh, kw, s, p, d=1)
        flops.append(_calc_conv_flops(n, c, oh, ow, f_out, kh, kw))

    logs = [math.log10(max(v, 1)) for v in flops]
    log_min, log_max = min(logs), max(logs)
    step = 1.0 if log_max == log_min else (log_max - log_min) / max(num_bins, 1)

    bins = [[] for _ in range(num_bins)]
    for idx, lv in enumerate(logs):
        if log_max == log_min:
            bid = 0
        else:
            bid = int((lv - log_min) / step)
            if bid >= num_bins:
                bid = num_bins - 1
        bins[bid].append(idx)

    desired = [len(b) * target_count / len(shapes) for b in bins]
    alloc = [int(x) for x in desired]

    non_empty_bins = [i for i, b in enumerate(bins) if b]
    if target_count >= len(non_empty_bins):
        for i in non_empty_bins:
            if alloc[i] == 0:
                alloc[i] = 1

    total_alloc = sum(alloc)
    if total_alloc > target_count:
        order = sorted(range(num_bins), key=lambda i: (desired[i] - alloc[i], -alloc[i]))
        for i in order:
            if total_alloc == target_count:
                break
            if alloc[i] > 0:
                alloc[i] -= 1
                total_alloc -= 1
    elif total_alloc < target_count:
        order = sorted(range(num_bins), key=lambda i: (desired[i] - alloc[i], len(bins[i])), reverse=True)
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
        else:
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


def get_ref_program(stride=1, padding=1, dilation=1, eps=1e-5):
    """Reference implementation: Conv2d + BatchNorm(inference) + ReLU."""

    def conv_bn_relu_core(X_nchw, W_oihw, BN_Gamma, BN_Beta, BN_Mean, BN_Var):
        Y_nchw = F.conv2d(X_nchw, W_oihw, stride=stride, padding=padding, dilation=dilation).to(torch.float32)

        gamma = BN_Gamma.to(torch.float32).view(1, -1, 1, 1)
        beta = BN_Beta.to(torch.float32).view(1, -1, 1, 1)
        mean = BN_Mean.to(torch.float32).view(1, -1, 1, 1)
        var = BN_Var.to(torch.float32).view(1, -1, 1, 1)

        Y_nchw = (Y_nchw - mean) * torch.rsqrt(var + eps) * gamma + beta
        Y_nchw.relu_()
        return Y_nchw

    compiled_conv_bn_relu_core = None
    if hasattr(torch, "compile"):
        try:
            compiled_conv_bn_relu_core = torch.compile(conv_bn_relu_core)
        except Exception:
            compiled_conv_bn_relu_core = None

    def ref_program(X, W, BN_Gamma, BN_Beta, BN_Mean, BN_Var, Y):
        nonlocal compiled_conv_bn_relu_core
        X_nchw = X.permute(0, 3, 1, 2)
        W_oihw = W.permute(3, 2, 0, 1)

        if compiled_conv_bn_relu_core is not None:
            try:
                Y_nchw = compiled_conv_bn_relu_core(X_nchw, W_oihw, BN_Gamma, BN_Beta, BN_Mean, BN_Var)
            except Exception:
                compiled_conv_bn_relu_core = None
                Y_nchw = conv_bn_relu_core(X_nchw, W_oihw, BN_Gamma, BN_Beta, BN_Mean, BN_Var)
        else:
            Y_nchw = conv_bn_relu_core(X_nchw, W_oihw, BN_Gamma, BN_Beta, BN_Mean, BN_Var)

        Y.copy_(Y_nchw.permute(0, 2, 3, 1).to(Y.dtype))
    return ref_program


def get_configs():
    """Generate autotuning configurations."""
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


def _conv2d_bn_relu_kernel_factory(N, C, H, W, F_out, KH, KW, S=1, P=1, D=1):
    """Return a Conv2d+BN+ReLU kernel factory."""
    OH = (H + 2 * P - D * (KH - 1) - 1) // S + 1
    OW = (W + 2 * P - D * (KW - 1) - 1) // S + 1

    M_eff = N * OH * OW
    K_eff = KH * KW * C

    def kernel(block_M=None, block_N=None, block_K=None,
               num_stages=None, thread_num=None):
        dtype = T.float16
        accum_dtype = T.float32
        out_dtype = T.float16

        @T.prim_func
        def conv_bn_relu_kernel(
            data: T.Tensor((N, H, W, C), dtype),
            kernel_weight: T.Tensor((KH, KW, C, F_out), dtype),
            bn_gamma: T.Tensor((F_out,), dtype),
            bn_beta: T.Tensor((F_out,), dtype),
            bn_mean: T.Tensor((F_out,), dtype),
            bn_var: T.Tensor((F_out,), dtype),
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
                        x = out_shared[i, j]
                        gamma = T.cast(bn_gamma[n_idx], out_dtype)
                        beta = T.cast(bn_beta[n_idx], out_dtype)
                        mean = T.cast(bn_mean[n_idx], out_dtype)
                        var = T.cast(bn_var[n_idx], out_dtype)

                        inv_std = 1.0 / T.sqrt(var + T.cast(1e-5, out_dtype))
                        y = (x - mean) * inv_std * gamma + beta
                        out_flat[m_idx, n_idx] = T.max(y, T.cast(0, out_dtype))

        return conv_bn_relu_kernel

    return kernel


def build_conv2d_bn_relu_kernel(N, C, H, W, F_out, KH, KW, S=1, P=1, D=1, config=None):
    """Build a compiled Conv2d+BN+ReLU kernel using cost-model selection or a given config."""
    if config is None:
        config, _ = select_best(DESCRIPTOR, N=N, C=C, H=H, W=W, OC=F_out,
                                KH=KH, KW=KW, stride=S, padding=P)
    factory = _conv2d_bn_relu_kernel_factory(N, C, H, W, F_out, KH, KW, S, P, D)
    prim_func = factory(**config)
    return tilelang.compile(prim_func, target="auto")


def estimate_total_flops(N, OH, OW, F_out, C, KH, KW):
    """Estimate FLOPs including conv and fused epilogue (BN + ReLU)."""
    conv_flops = 2 * N * OH * OW * F_out * C * KH * KW
    out_elems = N * OH * OW * F_out

    # BN+ReLU approximate elementwise FLOPs per output element.
    bn_relu_flops = 7 * out_elems

    return conv_flops + bn_relu_flops


def verify_correctness(kernel, ref_prog, N, C, H, W, F_out, KH, KW, S, P, D):
    """Verify correctness. Returns True if passed, False otherwise."""
    OH = (H + 2 * P - D * (KH - 1) - 1) // S + 1
    OW = (W + 2 * P - D * (KW - 1) - 1) // S + 1

    X = torch.randn(N, H, W, C, dtype=torch.float16, device='cuda')
    W_weight = torch.randn(KH, KW, C, F_out, dtype=torch.float16, device='cuda')
    BN_Gamma = torch.randn(F_out, dtype=torch.float16, device='cuda')
    BN_Beta = torch.randn(F_out, dtype=torch.float16, device='cuda')
    BN_Mean = torch.randn(F_out, dtype=torch.float16, device='cuda')
    BN_Var = torch.rand(F_out, dtype=torch.float16, device='cuda') + torch.tensor(0.5, dtype=torch.float16, device='cuda')
    
    Y_ref = torch.zeros(N, OH, OW, F_out, dtype=torch.float16, device='cuda')
    Y_tl = torch.zeros(N, OH, OW, F_out, dtype=torch.float16, device='cuda')

    ref_prog(X, W_weight, BN_Gamma, BN_Beta, BN_Mean, BN_Var, Y_ref)
    kernel(X, W_weight, BN_Gamma, BN_Beta, BN_Mean, BN_Var, Y_tl)

    try:
        torch.testing.assert_close(Y_tl, Y_ref, rtol=1e-2, atol=1e-2)
        return True
    except AssertionError as e:
        print(f"  [Verification] Status: FAILED ❌ -> {str(e)[:100]}...")
        return False


def main():
    parser = argparse.ArgumentParser(description="Benchmark Conv2d+BN+ReLU fusion with autotuning")
    parser.add_argument(
        "--output",
        default="conv2d_bn_relu_results.csv",
        help="Output CSV path (default: conv2d_bn_relu_results.csv)",
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
    print("Operator: Conv2d + BatchNorm + ReLU")
    print(f"Results will be saved to: {csv_filename}\n")

    # Write CSV header
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Index", "Fusion", "N", "C", "H", "W", "F_out", "Kernel_H", "Kernel_W", "Stride", "Pad", "Dilation",
            "Best_Config", "Is_Correct", "TileLang_Lat_ms", "Torch_Lat_ms", "TileLang_TFlops", "Torch_TFlops", "Speedup"
        ])

    for idx, shape in enumerate(shapes_to_test):
        N, C, H, W, F_out, KH, KW, S, P = shape
        D = 1

        OH = (H + 2 * P - D * (KH - 1) - 1) // S + 1
        OW = (W + 2 * P - D * (KW - 1) - 1) // S + 1
        print(f"[{idx+1}/{len(shapes_to_test)}] N={N}, C={C}, H={H}, W={W}, F={F_out}, K={KH}x{KW}, S={S}, P={P}")

        print("  [Fusion] BN_RELU")
        try:
            best_config, _ = select_best(DESCRIPTOR, N=N, C=C, H=H, W=W, OC=F_out,
                                         KH=KH, KW=KW, stride=S, padding=P)
            kernel = build_conv2d_bn_relu_kernel(N, C, H, W, F_out, KH, KW, S, P, D,
                                                  config=best_config)

            ref_prog = get_ref_program(S, P, D)

            is_correct = verify_correctness(kernel, ref_prog, N, C, H, W, F_out, KH, KW, S, P, D)
            correct_str = "True" if is_correct else "False"
            if is_correct:
                print("    [Verification] Status: SUCCESS")

            X = torch.randn(N, H, W, C, dtype=torch.float16, device="cuda")
            W_weight = torch.randn(KH, KW, C, F_out, dtype=torch.float16, device="cuda")
            BN_Gamma = torch.randn(F_out, dtype=torch.float16, device="cuda")
            BN_Beta = torch.randn(F_out, dtype=torch.float16, device="cuda")
            BN_Mean = torch.randn(F_out, dtype=torch.float16, device="cuda")
            BN_Var = torch.rand(F_out, dtype=torch.float16, device="cuda") + 0.5
            Y = torch.zeros(N, OH, OW, F_out, dtype=torch.float16, device="cuda")

            import triton
            tl_lat = triton.testing.do_bench(
                lambda: kernel(X, W_weight, BN_Gamma, BN_Beta, BN_Mean, BN_Var, Y),
                warmup=50, rep=100,
            )
            torch_ref = get_ref_program(S, P, D)
            torch_lat = triton.testing.do_bench(
                lambda: torch_ref(X, W_weight, BN_Gamma, BN_Beta, BN_Mean, BN_Var, Y),
                warmup=50, rep=100,
            )

            total_flops = estimate_total_flops(N, OH, OW, F_out, C, KH, KW)

            tl_tflops = total_flops / tl_lat * 1e-9
            torch_tflops = total_flops / torch_lat * 1e-9
            speedup = torch_lat / tl_lat

            print(f"    TileLang:  {tl_lat:.4f} ms  ({tl_tflops:.2f} TFlops)")
            print(f"    Torch:     {torch_lat:.4f} ms  ({torch_tflops:.2f} TFlops)")
            print(f"    Speedup:   {speedup:.2f}x")

            # Append results to CSV
            with open(csv_filename, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    idx + 1, "bn_relu", N, C, H, W, F_out, KH, KW, S, P, D,
                    str(best_config), correct_str,
                    f"{tl_lat:.4f}", f"{torch_lat:.4f}",
                    f"{tl_tflops:.2f}", f"{torch_tflops:.2f}", f"{speedup:.2f}"
                ])

        except Exception as e:
            print(f"    [Error] Failed to benchmark: {str(e)}")
            with open(csv_filename, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    idx + 1, "bn_relu", N, C, H, W, F_out, KH, KW, S, P, D,
                    f"ERROR: {str(e)}", "Error", "N/A", "N/A", "N/A", "N/A", "N/A"
                ])

        print()


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Unified autotuner descriptor
# ---------------------------------------------------------------------------


class Conv2dBnReluDescriptor(TileLangKernelBase):
    @property
    def name(self):
        return "conv2d_bn_relu"

    def make_op_spec(self, N, C, H, W, OC, KH, KW, stride=1, padding=0, **kw):
        return make_conv2d_spec(
            N, C, H, W, OC, KH, KW, stride=stride, padding=padding,
            primitives=["batchnorm", "relu"],
        )

    def get_raw_configs(self, **kw):
        return get_configs()


DESCRIPTOR = Conv2dBnReluDescriptor()
