"""
Adaptive autotuner for dynamic-shape GPU kernels.

Architecture: Score_final = Score_formula + Residual_ML
  - Score_formula: analytical 4-factor physics model
  - Residual_ML: LightGBM residual correction

Three-level online selection:
  Level 1: Decision table lookup (profiled shapes)
  Level 2: Formula + Residual ML (unseen shapes)
  Level 3: Pure formula (cold start fallback)

Unified kernel interface::

    from kernel.autotuner.interface import select_best, rank_all
    from kernel.tilelang.gemm.gemm_bias_act import DESCRIPTOR

    best_cfg, score = select_best(DESCRIPTOR, M=64, N=2304, K=768)
"""

from kernel.autotuner.interface import (  # noqa: F401
    TileLangKernelBase,
    configure_autotuner_cache,
    rank_all,
    select_best,
)
