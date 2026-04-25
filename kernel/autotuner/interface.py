"""
Unified kernel autotuner interface.

Each TileLang/Triton kernel implements a lightweight ``TileLangKernelBase``
subclass (the *descriptor*) that tells the autotuner how to build an
``OpSpec`` and what config search space to enumerate.  The generic
``select_best`` / ``rank_all`` functions then score every candidate via the
analytical cost model and return the winner -- no per-kernel eval script
boilerplate needed.

Typical usage inside a kernel file::

    class MyDescriptor(TileLangKernelBase):
        @property
        def name(self): return "gemm_bias_relu"
        def make_op_spec(self, M, N, K, **kw):
            return make_matmul_spec(M=M, N=N, K=K, primitives=["bias_add", "relu"])
        def get_raw_configs(self, M, K, **kw):
            return get_configs(M, K)

    DESCRIPTOR = MyDescriptor()

Then from any caller::

    from kernel.tilelang.gemm.gemm_bias_act import DESCRIPTOR
    from kernel.autotuner.interface import select_best

    best_cfg, score = select_best(DESCRIPTOR, M=64, N=2304, K=768)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from kernel.autotuner.cost_model import (
    OpSpec,
    TuneConfig,
    score_formula,
    score_formula_detailed,
)
from kernel.autotuner.gpu_spec import GPUSpec, get_gpu_spec


# ---------------------------------------------------------------------------
# Base class -- kernel authors subclass this
# ---------------------------------------------------------------------------


class TileLangKernelBase:
    """Base descriptor that every kernel subclasses.

    Subclasses **must** override:
      * ``name``
      * ``make_op_spec``
      * ``get_raw_configs``

    The remaining methods have sensible defaults that cover the common
    ``(block_M, block_N, block_K, num_stages, thread_num)`` config layout.
    Override only when necessary (e.g. split-k scoring adjustment).
    """

    # -- Must override -------------------------------------------------------

    @property
    def name(self) -> str:
        """Human-readable kernel name, e.g. ``'gemm_bias_relu'``."""
        raise NotImplementedError

    def make_op_spec(self, raw_config: Optional[Dict[str, Any]] = None,
                     **shape_kwargs: Any) -> OpSpec:
        """Build the cost-model ``OpSpec`` for a given shape.

        GEMM family:  receives ``M, N, K``.
        Conv family:   receives ``N, C, H, W, OC, KH, KW, stride, padding``.

        *raw_config* is optionally passed so that descriptors whose OpSpec
        depends on per-config parameters (e.g. split-k changes effective K)
        can read them.  Most descriptors ignore it.
        """
        raise NotImplementedError

    def get_raw_configs(self, **shape_kwargs: Any) -> List[Dict[str, Any]]:
        """Return the kernel's native config dicts.

        Each dict is passed directly as ``**kwargs`` to the kernel factory.
        """
        raise NotImplementedError

    # -- Defaults (override when needed) -------------------------------------

    def raw_config_to_tune_config(self, raw: Dict[str, Any]) -> TuneConfig:
        """Convert a raw kernel config dict to ``TuneConfig``.

        Default handles the common ``thread_num → num_warps`` mapping.
        """
        return TuneConfig(
            block_M=raw["block_M"],
            block_N=raw["block_N"],
            block_K=raw["block_K"],
            num_stages=raw["num_stages"],
            num_warps=raw["thread_num"] // 32,
        )

    def score_adjustment(
        self,
        raw: Dict[str, Any],
        base_score: float,
        **shape_kwargs: Any,
    ) -> float:
        """Kernel-specific score post-processing (e.g. split-k bonus).

        Default: return *base_score* unchanged.
        """
        return base_score

    def format_config(self, raw: Dict[str, Any]) -> str:
        """Human-readable one-liner for a config dict."""
        parts = [
            f"BM{raw['block_M']}",
            f"BN{raw['block_N']}",
            f"BK{raw['block_K']}",
        ]
        if "split_k" in raw:
            parts.append(f"SK{raw['split_k']}")
        parts.append(f"S{raw['num_stages']}")
        parts.append(f"T{raw['thread_num']}")
        return "_".join(parts)


# ---------------------------------------------------------------------------
# Pipeline functions
# ---------------------------------------------------------------------------


def select_best(
    descriptor: TileLangKernelBase,
    hw: Optional[GPUSpec] = None,
    **shape_kwargs: Any,
) -> Tuple[Dict[str, Any], float]:
    """Return the single best raw config and its score.

    Parameters
    ----------
    descriptor : TileLangKernelBase
        The kernel descriptor.
    hw : GPUSpec, optional
        GPU spec.  Auto-detected when *None*.
    **shape_kwargs
        Shape parameters forwarded to the descriptor
        (e.g. ``M=64, N=2304, K=768``).

    Returns
    -------
    (best_raw_config, best_score)
    """
    if hw is None:
        hw = get_gpu_spec()

    raw_configs = descriptor.get_raw_configs(**shape_kwargs)

    best_raw: Optional[Dict[str, Any]] = None
    best_score = -1.0

    for raw in raw_configs:
        op = descriptor.make_op_spec(raw_config=raw, **shape_kwargs)
        tune_cfg = descriptor.raw_config_to_tune_config(raw)
        base = score_formula(op, tune_cfg, hw)
        adjusted = descriptor.score_adjustment(raw, base, **shape_kwargs)
        if adjusted > best_score:
            best_score = adjusted
            best_raw = raw

    if best_raw is None:
        raise RuntimeError(
            f"No configs generated for {descriptor.name} with {shape_kwargs}"
        )
    return best_raw, best_score


def rank_all(
    descriptor: TileLangKernelBase,
    hw: Optional[GPUSpec] = None,
    detailed: bool = False,
    **shape_kwargs: Any,
) -> List[Dict[str, Any]]:
    """Rank every config for a shape, highest score first.

    Each entry is a dict with keys:

    * ``raw_config`` -- the kernel's native config dict
    * ``score`` -- adjusted float score
    * ``formatted`` -- human-readable config string

    When *detailed* is ``True``, an additional ``detail`` key holds the
    full ``score_formula_detailed`` breakdown (per-factor efficiencies and
    projection features).

    Parameters
    ----------
    descriptor : TileLangKernelBase
        The kernel descriptor.
    hw : GPUSpec, optional
        GPU spec.  Auto-detected when *None*.
    detailed : bool
        Include per-factor breakdown in each entry.
    **shape_kwargs
        Shape parameters forwarded to the descriptor.

    Returns
    -------
    List of result dicts, sorted by ``score`` descending.
    """
    if hw is None:
        hw = get_gpu_spec()

    raw_configs = descriptor.get_raw_configs(**shape_kwargs)

    results: List[Dict[str, Any]] = []
    for raw in raw_configs:
        op = descriptor.make_op_spec(raw_config=raw, **shape_kwargs)
        tune_cfg = descriptor.raw_config_to_tune_config(raw)
        if detailed:
            detail = score_formula_detailed(op, tune_cfg, hw)
            base = detail["score"]
        else:
            base = score_formula(op, tune_cfg, hw)
            detail = None

        adjusted = descriptor.score_adjustment(raw, base, **shape_kwargs)
        entry: Dict[str, Any] = {
            "raw_config": raw,
            "score": adjusted,
            "formatted": descriptor.format_config(raw),
        }
        if detailed:
            entry["detail"] = detail
        results.append(entry)

    results.sort(key=lambda e: e["score"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Shared helpers (formerly duplicated in every eval script)
# ---------------------------------------------------------------------------


def configure_autotuner_cache(cache_dir: str = ".tilelang_cache") -> None:
    """One-time TileLang cache/tmp setup.  Call before benchmarking."""
    import tilelang.env as tl_env
    from tilelang.autotuner import AutoTuner

    cache_root = Path(cache_dir).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / "tmp").mkdir(parents=True, exist_ok=True)
    tl_env.TILELANG_CACHE_DIR = str(cache_root)
    tl_env.TILELANG_TMP_DIR = str(cache_root / "tmp")
    tl_env.TILELANG_AUTO_TUNING_DISABLE_CACHE = "1"
    tl_env.disable_cache()
    AutoTuner.cache_dir = cache_root / "autotuner"


__all__ = [
    "TileLangKernelBase",
    "configure_autotuner_cache",
    "rank_all",
    "select_best",
]
