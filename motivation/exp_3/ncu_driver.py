"""
NCU Driver for Experiment 3: supports variable M dimension.

Usage:
    ncu ... python ncu_driver.py --bm 128 --bn 128 --bk 32 --warps 4 --stages 3 --M 256

Performs one warmup call (JIT compile) then one measured call.
NCU should use --launch-skip 1 --launch-count 1 to profile the second call.
"""

import argparse
import os
import sys

# Import kernel from exp_1
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exp_1"))
from benchmark import matmul_with_config, N, K

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bm", type=int, required=True)
    parser.add_argument("--bn", type=int, required=True)
    parser.add_argument("--bk", type=int, required=True)
    parser.add_argument("--warps", type=int, required=True)
    parser.add_argument("--stages", type=int, required=True)
    parser.add_argument("--M", type=int, default=256)
    args = parser.parse_args()

    M = args.M
    device = torch.device("cuda")
    a = torch.randn((M, K), device=device, dtype=torch.float16)
    b = torch.randn((K, N), device=device, dtype=torch.float16)

    config = {
        "BLOCK_SIZE_M": args.bm,
        "BLOCK_SIZE_N": args.bn,
        "BLOCK_SIZE_K": args.bk,
        "num_warps": args.warps,
        "num_stages": args.stages,
    }

    # Warmup: triggers Triton JIT compilation + first kernel launch
    _ = matmul_with_config(a, b, config)
    torch.cuda.synchronize()

    # Measured launch: NCU profiles this one (--launch-skip 1)
    _ = matmul_with_config(a, b, config)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
