"""
Trait-driven analytical cost model.

This module is intentionally separate from `cost_model.py`.

The design goal is to support many operator families and fusion patterns
without adding operator-specific score patches. Instead of scoring a coarse
label such as "bias_relu" directly, the model scores a generic `OpSpec`
composed from:

1. `MainloopSpec`
   Describes the dominant tiled compute structure, for example GEMM or
   implicit-GEMM convolution.
2. `FusionPrimitive`
   Describes epilogue/reduction primitives such as bias add, ReLU, GELU,
   layernorm, or batchnorm with execution traits instead of operator names.
3. `ConfigProjection`
   Projects an `(OpSpec, TuneConfig, GPUSpec)` triple into reusable execution
   features such as tile bytes, outputs per thread, occupancy, and wave fill.

The resulting score is still a product of normalized factors:

    score =
        eff_mainloop
      * eff_memory
      * eff_parallel
      * eff_pipeline
      * eff_epilogue
      * eff_reduction

The factors are generic and consume traits, not hard-coded per-op formulas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

from kernel.autotuner.gpu_spec import GPUSpec


# ---------------------------------------------------------------------------
# Operator specification
# ---------------------------------------------------------------------------


ReductionAxis = Literal["none", "m", "n", "k", "spatial", "channel"]
MainloopKind = Literal["gemm", "implicit_gemm_conv"]


@dataclass(frozen=True)
class FusionPrimitive:
    """Execution-level primitive used in a fused epilogue or side reduction."""

    name: str
    flops_per_output: float
    bytes_per_output: float = 0.0
    reduction_axis: ReductionAxis = "none"
    extra_passes: int = 0
    sync_penalty: float = 0.0
    preferred_outputs_per_thread: float = 16.0
    preferred_tile_m: Optional[float] = None
    preferred_tile_n: Optional[float] = None
    smem_bytes_per_output: float = 0.0

    @property
    def has_reduction(self) -> bool:
        return self.reduction_axis != "none"

    @property
    def work_units(self) -> float:
        return max(1.0, self.flops_per_output + self.bytes_per_output)


@dataclass(frozen=True)
class MainloopSpec:
    """Main tiled compute structure shared by many operators."""

    kind: MainloopKind
    M: int
    N: int
    K: int

    input_elements: int
    weight_elements: int
    output_elements: int

    input_element_bytes: int = 2
    weight_element_bytes: int = 2
    output_element_bytes: int = 2
    accum_element_bytes: int = 4

    flops_per_fma: int = 2
    input_reuse_hint: float = 1.0
    weight_reuse_hint: float = 1.0
    spatial_reuse_hint: float = 1.0
    preferred_group_m: int = 8

    @property
    def mainloop_flops(self) -> int:
        return self.flops_per_fma * self.M * self.N * self.K

    @property
    def min_global_bytes(self) -> int:
        return (
            self.input_elements * self.input_element_bytes
            + self.weight_elements * self.weight_element_bytes
            + self.output_elements * self.output_element_bytes
        )


@dataclass(frozen=True)
class OpSpec:
    """Unified operator description consumed by the analytical model."""

    name: str
    mainloop: MainloopSpec
    epilogue: Tuple[FusionPrimitive, ...] = field(default_factory=tuple)

    @property
    def M(self) -> int:
        return self.mainloop.M

    @property
    def N(self) -> int:
        return self.mainloop.N

    @property
    def K(self) -> int:
        return self.mainloop.K

    @property
    def flops(self) -> float:
        epilogue_flops = self.M * self.N * sum(p.flops_per_output for p in self.epilogue)
        return float(self.mainloop.mainloop_flops + epilogue_flops)

    @property
    def bytes_accessed(self) -> float:
        epilogue_bytes = self.M * self.N * sum(p.bytes_per_output for p in self.epilogue)
        return float(self.mainloop.min_global_bytes + epilogue_bytes)

    @property
    def arithmetic_intensity(self) -> float:
        if self.bytes_accessed <= 0:
            return 0.0
        return self.flops / self.bytes_accessed

    @property
    def has_reduction_epilogue(self) -> bool:
        return any(p.has_reduction for p in self.epilogue)


@dataclass(frozen=True)
class TuneConfig:
    """Backend-agnostic candidate tile configuration."""

    block_M: int
    block_N: int
    block_K: int
    num_stages: int
    num_warps: int
    group_size_m: int = 8

    @property
    def threads(self) -> int:
        return self.num_warps * 32

    @property
    def shared_mem_bytes(self) -> int:
        a_tile = self.block_M * self.block_K * 2
        b_tile = self.block_K * self.block_N * 2
        return (a_tile + b_tile) * self.num_stages

    @property
    def regs_per_thread_estimate(self) -> int:
        acc_elements = (self.block_M * self.block_N) // max(1, self.threads)
        return acc_elements * 2 + 40


@dataclass(frozen=True)
class ConfigProjection:
    """Projected execution features reused by all score factors."""

    tiles_M: int
    tiles_N: int
    tiles_K: int
    grid_size: int
    tile_outputs: int
    outputs_per_thread: float
    mainloop_iterations: int

    mainloop_flops_per_tile: float
    epilogue_flops_per_tile: float
    total_flops_per_tile: float

    input_bytes_per_tile: float
    weight_bytes_per_tile: float
    output_bytes_per_tile: float
    epilogue_bytes_per_tile: float
    total_global_bytes_per_tile: float
    shared_mem_bytes_per_tile: float

    tile_arithmetic_intensity: float
    occupancy: float
    resident_blocks_per_sm: int
    wave_fill: float
    preferred_num_stages: int

    reduction_work: float
    reduction_passes: int
    epilogue_preferred_outputs: float


# ---------------------------------------------------------------------------
# Primitive library
# ---------------------------------------------------------------------------


BIAS_ADD = FusionPrimitive(
    name="bias_add",
    flops_per_output=1.0,
    bytes_per_output=0.0,
    preferred_outputs_per_thread=16.0,
)

RELU = FusionPrimitive(
    name="relu",
    flops_per_output=1.0,
    bytes_per_output=0.0,
    preferred_outputs_per_thread=16.0,
)

SILU = FusionPrimitive(
    name="silu",
    flops_per_output=4.0,
    bytes_per_output=0.0,
    preferred_outputs_per_thread=12.0,
)

GELU = FusionPrimitive(
    name="gelu",
    flops_per_output=8.0,
    bytes_per_output=0.0,
    preferred_outputs_per_thread=8.0,
)

ROW_LAYERNORM = FusionPrimitive(
    name="row_layernorm",
    flops_per_output=10.0,
    bytes_per_output=4.0,
    reduction_axis="n",
    extra_passes=1,
    sync_penalty=0.35,
    preferred_outputs_per_thread=8.0,
    preferred_tile_n=128.0,
)

BATCHNORM = FusionPrimitive(
    name="batchnorm",
    flops_per_output=6.0,
    bytes_per_output=2.0,
    reduction_axis="channel",
    extra_passes=1,
    sync_penalty=0.25,
    preferred_outputs_per_thread=12.0,
    preferred_tile_n=128.0,
)

PRIMITIVE_LIBRARY: Dict[str, FusionPrimitive] = {
    p.name: p
    for p in [BIAS_ADD, RELU, SILU, GELU, ROW_LAYERNORM, BATCHNORM]
}


def resolve_primitives(primitives: Iterable[str | FusionPrimitive]) -> Tuple[FusionPrimitive, ...]:
    resolved: List[FusionPrimitive] = []
    for primitive in primitives:
        if isinstance(primitive, FusionPrimitive):
            resolved.append(primitive)
        else:
            if primitive not in PRIMITIVE_LIBRARY:
                raise KeyError(f"Unknown fusion primitive: {primitive}")
            resolved.append(PRIMITIVE_LIBRARY[primitive])
    return tuple(resolved)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def make_matmul_spec(
    M: int,
    N: int,
    K: int,
    primitives: Sequence[str | FusionPrimitive] = (),
    *,
    input_element_bytes: int = 2,
    weight_element_bytes: int = 2,
    output_element_bytes: int = 2,
    name: Optional[str] = None,
) -> OpSpec:
    epilogue = resolve_primitives(primitives)
    mainloop = MainloopSpec(
        kind="gemm",
        M=M,
        N=N,
        K=K,
        input_elements=M * K,
        weight_elements=K * N,
        output_elements=M * N,
        input_element_bytes=input_element_bytes,
        weight_element_bytes=weight_element_bytes,
        output_element_bytes=output_element_bytes,
        input_reuse_hint=1.0,
        weight_reuse_hint=1.0,
        spatial_reuse_hint=1.0,
    )
    return OpSpec(name=name or "matmul", mainloop=mainloop, epilogue=epilogue)


def make_conv2d_spec(
    N: int,
    C: int,
    H: int,
    W: int,
    OC: int,
    KH: int,
    KW: int,
    *,
    stride: int = 1,
    padding: int = 0,
    primitives: Sequence[str | FusionPrimitive] = (),
    input_element_bytes: int = 2,
    weight_element_bytes: int = 2,
    output_element_bytes: int = 2,
    name: Optional[str] = None,
) -> OpSpec:
    OH = (H + 2 * padding - KH) // stride + 1
    OW = (W + 2 * padding - KW) // stride + 1

    gemm_M = N * OH * OW
    gemm_N = OC
    gemm_K = C * KH * KW

    spatial_overlap = max(1.0, (KH * KW) / max(1.0, float(stride * stride)))
    epilogue = resolve_primitives(primitives)
    mainloop = MainloopSpec(
        kind="implicit_gemm_conv",
        M=gemm_M,
        N=gemm_N,
        K=gemm_K,
        input_elements=N * C * H * W,
        weight_elements=OC * C * KH * KW,
        output_elements=N * OC * OH * OW,
        input_element_bytes=input_element_bytes,
        weight_element_bytes=weight_element_bytes,
        output_element_bytes=output_element_bytes,
        input_reuse_hint=math.sqrt(spatial_overlap),
        weight_reuse_hint=math.sqrt(max(1.0, spatial_overlap)),
        spatial_reuse_hint=spatial_overlap,
    )
    return OpSpec(name=name or "conv2d", mainloop=mainloop, epilogue=epilogue)


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------


_PREFERRED_OCCUPANCY = 0.5
_OCCUPANCY_SATURATION_EXPONENT = 0.15
_WAVE_FILL_EXPONENT = 0.35
_HIGH_WARP_COUNT_PENALTY = 0.9
_MEMORY_EFFICIENCY_EXPONENT = 0.75
_PIPELINE_FILL_EXPONENT = 0.5
_STAGE_MISMATCH_PENALTY = 0.5
_MAINLOOP_TARGET_OUTPUTS_PER_THREAD = 32.0
_MAINLOOP_MIN_OUTPUTS_PER_THREAD = 2.0
_MAINLOOP_OUTPUT_DISTANCE_PENALTY = 0.4
_EPILOGUE_DISTANCE_PENALTY = 0.5
_REDUCTION_DISTANCE_PENALTY = 0.3
_CONV_BLOCK_M_DISTANCE_PENALTY = 0.45
_CONV_BLOCK_N_DISTANCE_PENALTY = 0.6
_CONV_HIGH_WARP_PENALTY = 0.94
_CONV_PLAIN_SPATIAL_WORK_PENALTY = 0.18
_CONV_PLAIN_BLOCK_K_OVERSIZE_PENALTY = 0.45
_CONV_PLAIN_HIGH_WARP_PENALTY = 0.55


def _ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _resident_blocks_per_sm(config: TuneConfig, hw: GPUSpec) -> int:
    warps_per_block = max(1, config.num_warps)
    blocks_by_threads = hw.max_warps_per_sm // warps_per_block

    if config.shared_mem_bytes > 0:
        blocks_by_smem = hw.shared_mem_per_sm // config.shared_mem_bytes
    else:
        blocks_by_smem = hw.max_blocks_per_sm

    regs_per_block = config.regs_per_thread_estimate * config.threads
    if regs_per_block > 0:
        blocks_by_regs = hw.max_regs_per_sm // regs_per_block
    else:
        blocks_by_regs = hw.max_blocks_per_sm

    return max(0, min(blocks_by_threads, blocks_by_smem, blocks_by_regs, hw.max_blocks_per_sm))


def _estimate_occupancy(config: TuneConfig, hw: GPUSpec) -> float:
    blocks_per_sm = _resident_blocks_per_sm(config, hw)
    if hw.max_warps_per_sm <= 0:
        return 0.0
    return min(1.0, (blocks_per_sm * config.num_warps) / hw.max_warps_per_sm)


def _preferred_block_k(op: OpSpec) -> int:
    K = op.K
    if K >= 64:
        return 64
    if K >= 32:
        return 32
    return 16


def _warp_count_efficiency(config: TuneConfig) -> float:
    if config.num_warps <= 4:
        return 1.0
    penalty_steps = math.log2(config.num_warps / 4.0)
    return _HIGH_WARP_COUNT_PENALTY ** penalty_steps


def _weighted_preferred_outputs(op: OpSpec) -> float:
    if not op.epilogue:
        return 16.0
    total_weight = sum(p.work_units for p in op.epilogue)
    return sum(p.preferred_outputs_per_thread * p.work_units for p in op.epilogue) / total_weight


def _estimate_operand_reuse(op: OpSpec, config: TuneConfig, tiles_M: int) -> Tuple[float, float]:
    group_m = max(1, min(config.group_size_m, max(1, tiles_M)))

    if op.mainloop.kind == "gemm":
        input_reuse = 1.0
        weight_reuse = max(1.0, group_m ** 0.5)
        return input_reuse, weight_reuse

    input_reuse = max(1.0, op.mainloop.input_reuse_hint)
    weight_reuse = max(1.0, (group_m * op.mainloop.weight_reuse_hint) ** 0.5)
    return input_reuse, weight_reuse


def _preferred_num_stages(op: OpSpec, config: TuneConfig, tile_ai: float, hw: GPUSpec) -> int:
    ridge = hw.ridge_point_fp16
    memory_eff = 1.0 if ridge <= 0 else min(1.0, tile_ai / ridge)
    k_iters = max(1, _ceildiv(op.K, config.block_K))

    preferred = 2
    if op.mainloop.kind == "implicit_gemm_conv":
        preferred += 1
    if memory_eff < 0.9:
        preferred += 1
    if memory_eff < 0.5:
        preferred += 1
    if op.has_reduction_epilogue:
        preferred += 1

    return min(max(2, preferred), max(2, min(6, k_iters)))


def project_config(op: OpSpec, config: TuneConfig, hw: GPUSpec) -> ConfigProjection:
    tiles_M = _ceildiv(op.M, config.block_M)
    tiles_N = _ceildiv(op.N, config.block_N)
    tiles_K = _ceildiv(op.K, config.block_K)
    grid_size = tiles_M * tiles_N
    tile_outputs = config.block_M * config.block_N
    outputs_per_thread = tile_outputs / max(1, config.threads)

    input_reuse, weight_reuse = _estimate_operand_reuse(op, config, tiles_M)

    mainloop_flops = op.mainloop.flops_per_fma * config.block_M * config.block_N * op.K
    epilogue_flops = tile_outputs * sum(p.flops_per_output for p in op.epilogue)
    total_flops = mainloop_flops + epilogue_flops

    input_bytes = (op.K * config.block_M * op.mainloop.input_element_bytes) / input_reuse
    weight_bytes = (op.K * config.block_N * op.mainloop.weight_element_bytes) / weight_reuse
    output_bytes = tile_outputs * op.mainloop.output_element_bytes
    epilogue_bytes = tile_outputs * sum(p.bytes_per_output for p in op.epilogue)
    total_global_bytes = input_bytes + weight_bytes + output_bytes + epilogue_bytes
    shared_mem_bytes = config.shared_mem_bytes + tile_outputs * sum(p.smem_bytes_per_output for p in op.epilogue)

    if total_global_bytes <= 0:
        tile_ai = 0.0
    else:
        tile_ai = total_flops / total_global_bytes

    resident_blocks = max(1, _resident_blocks_per_sm(config, hw))
    resident_slots = max(1, hw.num_sm * resident_blocks)
    num_waves = max(1, _ceildiv(grid_size, resident_slots))
    wave_fill = grid_size / (num_waves * resident_slots)
    occupancy = _estimate_occupancy(config, hw)

    reduction_work = sum(p.work_units for p in op.epilogue if p.has_reduction)
    reduction_passes = sum(p.extra_passes for p in op.epilogue if p.has_reduction)
    preferred_outputs = _weighted_preferred_outputs(op)

    return ConfigProjection(
        tiles_M=tiles_M,
        tiles_N=tiles_N,
        tiles_K=tiles_K,
        grid_size=grid_size,
        tile_outputs=tile_outputs,
        outputs_per_thread=outputs_per_thread,
        mainloop_iterations=max(1, tiles_K),
        mainloop_flops_per_tile=mainloop_flops,
        epilogue_flops_per_tile=epilogue_flops,
        total_flops_per_tile=total_flops,
        input_bytes_per_tile=input_bytes,
        weight_bytes_per_tile=weight_bytes,
        output_bytes_per_tile=output_bytes,
        epilogue_bytes_per_tile=epilogue_bytes,
        total_global_bytes_per_tile=total_global_bytes,
        shared_mem_bytes_per_tile=shared_mem_bytes,
        tile_arithmetic_intensity=tile_ai,
        occupancy=occupancy,
        resident_blocks_per_sm=resident_blocks,
        wave_fill=wave_fill,
        preferred_num_stages=_preferred_num_stages(op, config, tile_ai, hw),
        reduction_work=reduction_work,
        reduction_passes=reduction_passes,
        epilogue_preferred_outputs=preferred_outputs,
    )


# ---------------------------------------------------------------------------
# Score factors
# ---------------------------------------------------------------------------


def eff_mainloop(op: OpSpec, config: TuneConfig, proj: ConfigProjection) -> float:
    actual_elements = (
        proj.tiles_M
        * config.block_M
        * proj.tiles_N
        * config.block_N
        * proj.tiles_K
        * config.block_K
    )
    useful_elements = op.M * op.N * op.K
    padding_eff = 1.0 if actual_elements == 0 else useful_elements / actual_elements

    # When the launch already exposes enough blocks to fill resident slots, the
    # mainloop can afford a higher outputs/thread target. When grid size or
    # occupancy are low, oversizing tiles should be penalized because it
    # directly reduces the number of schedulable blocks.
    launch_fill = min(1.0, proj.wave_fill * max(1, proj.resident_blocks_per_sm))
    occupancy_fill = min(1.0, proj.occupancy / max(_PREFERRED_OCCUPANCY, 1e-6))
    parallel_signal = math.sqrt(max(1e-6, launch_fill * occupancy_fill))
    target_outputs = (
        _MAINLOOP_MIN_OUTPUTS_PER_THREAD
        + (_MAINLOOP_TARGET_OUTPUTS_PER_THREAD - _MAINLOOP_MIN_OUTPUTS_PER_THREAD) * parallel_signal
    )
    outputs_ratio = max(1e-6, proj.outputs_per_thread / target_outputs)
    outputs_distance = abs(math.log2(outputs_ratio))
    thread_eff = 1.0 / (1.0 + _MAINLOOP_OUTPUT_DISTANCE_PENALTY * outputs_distance)

    k_eff = math.sqrt(min(1.0, config.block_K / _preferred_block_k(op)))
    return padding_eff * thread_eff * k_eff


def eff_memory(op: OpSpec, config: TuneConfig, proj: ConfigProjection, hw: GPUSpec) -> float:
    ridge = hw.ridge_point_fp16
    if ridge <= 0:
        return 1.0
    roofline_eff = min(1.0, proj.tile_arithmetic_intensity / ridge)

    smem_budget = max(1.0, float(hw.shared_mem_per_sm))
    smem_pressure = min(1.0, proj.shared_mem_bytes_per_tile / smem_budget)
    smem_eff = 1.0 / (1.0 + 0.25 * smem_pressure)

    return (roofline_eff ** _MEMORY_EFFICIENCY_EXPONENT) * smem_eff


def eff_parallel(op: OpSpec, config: TuneConfig, proj: ConfigProjection, hw: GPUSpec) -> float:
    if proj.grid_size == 0 or hw.num_sm == 0:
        return 1.0

    occupancy_eff = min(1.0, proj.occupancy / _PREFERRED_OCCUPANCY) ** _OCCUPANCY_SATURATION_EXPONENT
    wave_eff = proj.wave_fill ** _WAVE_FILL_EXPONENT
    warp_eff = _warp_count_efficiency(config)
    return occupancy_eff * wave_eff * warp_eff


def eff_pipeline(op: OpSpec, config: TuneConfig, proj: ConfigProjection, hw: GPUSpec) -> float:
    if config.num_stages <= 0 or proj.mainloop_iterations <= 0:
        return 1.0

    fill_eff = (
        proj.mainloop_iterations / (proj.mainloop_iterations + config.num_stages - 1)
    ) ** _PIPELINE_FILL_EXPONENT
    stage_match = 1.0 / (
        1.0 + _STAGE_MISMATCH_PENALTY * abs(config.num_stages - proj.preferred_num_stages)
    )
    return fill_eff * stage_match


def eff_implicit_conv(op: OpSpec, config: TuneConfig, proj: ConfigProjection, hw: GPUSpec) -> float:
    if op.mainloop.kind != "implicit_gemm_conv":
        return 1.0

    overlap = max(1.0, op.mainloop.spatial_reuse_hint)
    if overlap <= 1.0:
        return 1.0

    # Implicit-GEMM conv pays extra address-generation and gather cost. The
    # penalty matters most when the launch is already underfilled: in that
    # regime, oversizing tiles further reduces the number of schedulable
    # blocks and hurts latency more than a GEMM-like roofline estimate
    # predicts.
    overlap_signal = min(1.5, math.log2(overlap) / math.log2(9.0))
    resident_slots = max(1, hw.num_sm * max(1, proj.resident_blocks_per_sm))
    launch_util = min(1.0, proj.grid_size / resident_slots)
    underfill = 1.0 - launch_util

    # For plain spatial convolutions, a high-overlap implicit-GEMM mapping is
    # not as GEMM-like as the roofline/reuse estimate suggests. This remains
    # true even when the launch is fully occupied: large 3x3/5x5 convolutions
    # pay address-generation and input-gather costs that are mostly absent in a
    # dense GEMM. Keep this strongest for unfused conv2d; fusion-heavy epilogues
    # have their own benefits and should not inherit the same penalty.
    plain_conv = not op.epilogue
    spatial_work_eff = 1.0
    block_k_eff = 1.0
    plain_warp_eff = 1.0
    inferred_input_channels = op.K / overlap
    high_risk_plain_spatial = plain_conv and overlap >= 8.5 and (
        op.N >= 512 or inferred_input_channels <= 96.0
    )
    if high_risk_plain_spatial:
        work_signal = min(1.0, op.mainloop.mainloop_flops / 1.0e10)
        full_launch_signal = max(0.35, launch_util)
        spatial_work_eff = 1.0 / (
            1.0
            + _CONV_PLAIN_SPATIAL_WORK_PENALTY
            * overlap_signal
            * work_signal
            * full_launch_signal
        )

        if config.block_K > 32:
            block_k_eff = 1.0 / (
                1.0
                + _CONV_PLAIN_BLOCK_K_OVERSIZE_PENALTY
                * overlap_signal
                * work_signal
                * math.log2(config.block_K / 32.0)
            )

        if config.num_warps > 4:
            plain_warp_eff = _CONV_PLAIN_HIGH_WARP_PENALTY ** (
                work_signal * math.log2(config.num_warps / 4.0)
            )

    if underfill <= 0.0:
        return spatial_work_eff * block_k_eff * plain_warp_eff

    preferred_block_n = 64.0 if overlap >= 4.0 and underfill >= 0.2 else 128.0
    preferred_block_m = 32.0 if overlap >= 4.0 and underfill >= 0.5 else 64.0

    m_distance = abs(math.log2(config.block_M / preferred_block_m))
    n_distance = abs(math.log2(config.block_N / preferred_block_n))

    m_eff = 1.0 / (
        1.0 + _CONV_BLOCK_M_DISTANCE_PENALTY * overlap_signal * underfill * m_distance
    )
    n_eff = 1.0 / (
        1.0 + _CONV_BLOCK_N_DISTANCE_PENALTY * overlap_signal * underfill * n_distance
    )

    if overlap >= 4.0 and underfill >= 0.2 and config.num_warps > 4:
        warp_eff = _CONV_HIGH_WARP_PENALTY ** math.log2(config.num_warps / 4.0)
    else:
        warp_eff = 1.0

    stage_eff = 1.0
    if overlap > 1.0 and config.num_stages > 1:
        ridge = hw.ridge_point_fp16
        if ridge > 0:
            memory_pressure = 1.0 - min(1.0, proj.tile_arithmetic_intensity / ridge)
        else:
            memory_pressure = 0.0
        stage_eff = 1.0 / (
            1.0 + 0.25 * (config.num_stages - 1) * overlap_signal * memory_pressure
        )

    return m_eff * n_eff * warp_eff * stage_eff * spatial_work_eff * block_k_eff * plain_warp_eff


def _primitive_efficiency(primitive: FusionPrimitive, config: TuneConfig, proj: ConfigProjection) -> float:
    outputs_ratio = max(1e-6, proj.outputs_per_thread / primitive.preferred_outputs_per_thread)
    distance = abs(math.log2(outputs_ratio))
    eff = 1.0 / (1.0 + _EPILOGUE_DISTANCE_PENALTY * distance)

    if primitive.preferred_tile_m:
        m_distance = abs(math.log2(config.block_M / primitive.preferred_tile_m))
        eff *= 1.0 / (1.0 + 0.2 * m_distance)
    if primitive.preferred_tile_n:
        n_distance = abs(math.log2(config.block_N / primitive.preferred_tile_n))
        eff *= 1.0 / (1.0 + 0.2 * n_distance)
    return eff


def eff_epilogue(op: OpSpec, config: TuneConfig, proj: ConfigProjection) -> float:
    if not op.epilogue:
        return 1.0

    total_weight = sum(p.work_units for p in op.epilogue)
    weighted_log_sum = 0.0
    for primitive in op.epilogue:
        primitive_eff = _primitive_efficiency(primitive, config, proj)
        weight = primitive.work_units / total_weight
        weighted_log_sum += weight * math.log(max(1e-9, primitive_eff))

    epilogue_share = proj.epilogue_flops_per_tile / max(1.0, proj.total_flops_per_tile)
    epilogue_strength = min(1.0, 0.25 + 4.0 * epilogue_share)
    if any(p.has_reduction or p.bytes_per_output > 0.0 or p.sync_penalty > 0.0 for p in op.epilogue):
        epilogue_strength = max(epilogue_strength, 0.65)

    return math.exp(weighted_log_sum * epilogue_strength)


def eff_reduction(op: OpSpec, config: TuneConfig, proj: ConfigProjection) -> float:
    reduction_primitives = [p for p in op.epilogue if p.has_reduction]
    if not reduction_primitives:
        return 1.0

    penalties = []
    for primitive in reduction_primitives:
        axis_extent = config.block_N if primitive.reduction_axis in {"n", "channel"} else config.block_M
        preferred_extent = primitive.preferred_tile_n if primitive.reduction_axis in {"n", "channel"} else primitive.preferred_tile_m
        if preferred_extent is None:
            preferred_extent = 128.0 if primitive.reduction_axis in {"n", "channel"} else 64.0

        extent_distance = abs(math.log2(axis_extent / preferred_extent))
        pass_penalty = 1.0 / (1.0 + 0.2 * primitive.extra_passes)
        sync_penalty = 1.0 / (1.0 + primitive.sync_penalty)
        axis_penalty = 1.0 / (1.0 + _REDUCTION_DISTANCE_PENALTY * extent_distance)
        penalties.append(pass_penalty * sync_penalty * axis_penalty)

    product = 1.0
    for penalty in penalties:
        product *= penalty
    return product ** (1.0 / len(penalties))


# ---------------------------------------------------------------------------
# Combined score
# ---------------------------------------------------------------------------


def score_formula(op: OpSpec, config: TuneConfig, hw: GPUSpec) -> float:
    proj = project_config(op, config, hw)
    return (
        eff_mainloop(op, config, proj)
        * eff_memory(op, config, proj, hw)
        * eff_parallel(op, config, proj, hw)
        * eff_pipeline(op, config, proj, hw)
        * eff_implicit_conv(op, config, proj, hw)
        * eff_epilogue(op, config, proj)
        * eff_reduction(op, config, proj)
    )


def score_formula_detailed(op: OpSpec, config: TuneConfig, hw: GPUSpec) -> dict:
    proj = project_config(op, config, hw)
    emain = eff_mainloop(op, config, proj)
    emem = eff_memory(op, config, proj, hw)
    epar = eff_parallel(op, config, proj, hw)
    epipe = eff_pipeline(op, config, proj, hw)
    econv = eff_implicit_conv(op, config, proj, hw)
    eepi = eff_epilogue(op, config, proj)
    ered = eff_reduction(op, config, proj)
    score = emain * emem * epar * epipe * econv * eepi * ered

    return {
        "score": score,
        "eff_mainloop": emain,
        "eff_memory": emem,
        "eff_parallel": epar,
        "eff_pipeline": epipe,
        "eff_implicit_conv": econv,
        "eff_epilogue": eepi,
        "eff_reduction": ered,
        "projection": {
            "tiles_M": proj.tiles_M,
            "tiles_N": proj.tiles_N,
            "tiles_K": proj.tiles_K,
            "grid_size": proj.grid_size,
            "tile_outputs": proj.tile_outputs,
            "outputs_per_thread": proj.outputs_per_thread,
            "mainloop_iterations": proj.mainloop_iterations,
            "tile_arithmetic_intensity": proj.tile_arithmetic_intensity,
            "occupancy": proj.occupancy,
            "resident_blocks_per_sm": proj.resident_blocks_per_sm,
            "wave_fill": proj.wave_fill,
            "preferred_num_stages": proj.preferred_num_stages,
            "input_bytes_per_tile": proj.input_bytes_per_tile,
            "weight_bytes_per_tile": proj.weight_bytes_per_tile,
            "output_bytes_per_tile": proj.output_bytes_per_tile,
            "epilogue_bytes_per_tile": proj.epilogue_bytes_per_tile,
            "shared_mem_bytes_per_tile": proj.shared_mem_bytes_per_tile,
            "reduction_passes": proj.reduction_passes,
            "epilogue_preferred_outputs": proj.epilogue_preferred_outputs,
        },
    }


def score_configs(op: OpSpec, configs: Sequence[TuneConfig], hw: GPUSpec) -> List[float]:
    return [score_formula(op, cfg, hw) for cfg in configs]


def select_best_config(op: OpSpec, configs: Sequence[TuneConfig], hw: GPUSpec) -> Tuple[int, TuneConfig, float]:
    scores = score_configs(op, configs, hw)
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return best_idx, configs[best_idx], scores[best_idx]


__all__ = [
    "BATCHNORM",
    "BIAS_ADD",
    "ConfigProjection",
    "FusionPrimitive",
    "GELU",
    "MainloopSpec",
    "OpSpec",
    "PRIMITIVE_LIBRARY",
    "RELU",
    "ROW_LAYERNORM",
    "SILU",
    "TuneConfig",
    "eff_epilogue",
    "eff_mainloop",
    "eff_memory",
    "eff_parallel",
    "eff_pipeline",
    "eff_implicit_conv",
    "eff_reduction",
    "make_conv2d_spec",
    "make_matmul_spec",
    "project_config",
    "resolve_primitives",
    "score_configs",
    "score_formula",
    "score_formula_detailed",
    "select_best_config",
]
