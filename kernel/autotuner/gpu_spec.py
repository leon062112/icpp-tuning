"""
GPU hardware specification detection and modeling.

Detects GPU capabilities at runtime and provides a structured representation
used by the cost model for hardware-aware performance prediction.
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class GPUSpec:
    """Hardware specification of a GPU device.

    All fields are populated from runtime queries. Used as input to the
    analytical cost model (Eff_memory, Eff_parallel, etc.).
    """
    # Identity
    name: str
    compute_capability: tuple  # (major, minor)

    # Compute resources
    num_sm: int               # Number of streaming multiprocessors
    max_threads_per_sm: int   # Max resident threads per SM
    max_warps_per_sm: int     # Max resident warps per SM
    warp_size: int            # Threads per warp (32 for NVIDIA)
    max_threads_per_block: int
    max_regs_per_block: int   # 32-bit registers per block
    max_regs_per_sm: int      # 32-bit registers per SM

    # Memory hierarchy
    shared_mem_per_block: int    # bytes, configurable max
    shared_mem_per_sm: int       # bytes, total per SM
    l2_cache_size: int           # bytes
    global_mem_size: int         # bytes

    # Bandwidth & throughput
    mem_bandwidth_gbps: float       # GB/s, theoretical peak
    peak_fp16_tflops: float         # TFLOPS, tensor core peak
    peak_fp32_tflops: float         # TFLOPS, FP32 CUDA core peak

    # Derived
    @property
    def ridge_point_fp16(self) -> float:
        """Roofline ridge point: ops/byte to saturate compute (FP16 tensor core)."""
        if self.mem_bandwidth_gbps <= 0:
            return float('inf')
        return (self.peak_fp16_tflops * 1e3) / self.mem_bandwidth_gbps  # FLOP/Byte

    @property
    def ridge_point_fp32(self) -> float:
        """Roofline ridge point for FP32 CUDA cores."""
        if self.mem_bandwidth_gbps <= 0:
            return float('inf')
        return (self.peak_fp32_tflops * 1e3) / self.mem_bandwidth_gbps

    @property
    def max_blocks_per_sm(self) -> int:
        """Maximum concurrent blocks per SM (architecture-dependent)."""
        major = self.compute_capability[0]
        if major >= 8:  # Ampere+
            return 32
        elif major >= 7:  # Volta/Turing
            return 32
        else:
            return 16


_KNOWN_GPUS = {
    "A100": GPUSpec(
        name="NVIDIA A100",
        compute_capability=(8, 0),
        num_sm=108,
        max_threads_per_sm=2048,
        max_warps_per_sm=64,
        warp_size=32,
        max_threads_per_block=1024,
        max_regs_per_block=65536,
        max_regs_per_sm=65536,
        shared_mem_per_block=163840,  # 160KB configurable
        shared_mem_per_sm=167936,     # 164KB total
        l2_cache_size=40 * 1024 * 1024,  # 40MB
        global_mem_size=80 * 1024**3,    # 80GB
        mem_bandwidth_gbps=2039.0,
        peak_fp16_tflops=312.0,
        peak_fp32_tflops=19.5,
    ),
    "A800": GPUSpec(
        name="NVIDIA A800",
        compute_capability=(8, 0),
        num_sm=108,
        max_threads_per_sm=2048,
        max_warps_per_sm=64,
        warp_size=32,
        max_threads_per_block=1024,
        max_regs_per_block=65536,
        max_regs_per_sm=65536,
        shared_mem_per_block=163840,
        shared_mem_per_sm=167936,
        l2_cache_size=40 * 1024 * 1024,
        global_mem_size=80 * 1024**3,
        mem_bandwidth_gbps=2039.0,
        peak_fp16_tflops=312.0,
        peak_fp32_tflops=19.5,
    ),
    "H100": GPUSpec(
        name="NVIDIA H100",
        compute_capability=(9, 0),
        num_sm=132,
        max_threads_per_sm=2048,
        max_warps_per_sm=64,
        warp_size=32,
        max_threads_per_block=1024,
        max_regs_per_block=65536,
        max_regs_per_sm=65536,
        shared_mem_per_block=228 * 1024,  # 228KB
        shared_mem_per_sm=228 * 1024,
        l2_cache_size=50 * 1024 * 1024,
        global_mem_size=80 * 1024**3,
        mem_bandwidth_gbps=3352.0,
        peak_fp16_tflops=989.0,
        peak_fp32_tflops=67.0,
    ),
    "V100": GPUSpec(
        name="NVIDIA V100",
        compute_capability=(7, 0),
        num_sm=80,
        max_threads_per_sm=2048,
        max_warps_per_sm=64,
        warp_size=32,
        max_threads_per_block=1024,
        max_regs_per_block=65536,
        max_regs_per_sm=65536,
        shared_mem_per_block=98304,  # 96KB
        shared_mem_per_sm=98304,
        l2_cache_size=6 * 1024 * 1024,
        global_mem_size=32 * 1024**3,
        mem_bandwidth_gbps=900.0,
        peak_fp16_tflops=125.0,
        peak_fp32_tflops=15.7,
    ),
    "4090": GPUSpec(
        name="NVIDIA RTX 4090",
        compute_capability=(8, 9),
        num_sm=128,
        max_threads_per_sm=1536,
        max_warps_per_sm=48,
        warp_size=32,
        max_threads_per_block=1024,
        max_regs_per_block=65536,
        max_regs_per_sm=65536,
        shared_mem_per_block=100 * 1024,
        shared_mem_per_sm=100 * 1024,
        l2_cache_size=72 * 1024 * 1024,
        global_mem_size=24 * 1024**3,
        mem_bandwidth_gbps=1008.0,
        peak_fp16_tflops=330.0,
        peak_fp32_tflops=82.6,
    ),
}


def _match_known_gpu(device_name: str) -> Optional[GPUSpec]:
    """Try to match device name against known GPU database."""
    name_upper = device_name.upper()
    for key, spec in _KNOWN_GPUS.items():
        if key.upper() in name_upper:
            return spec
    return None


def detect_gpu(device_id: int = 0) -> GPUSpec:
    """Detect GPU hardware specification from the current CUDA device.

    Queries torch.cuda properties at runtime. Falls back to known GPU
    database if certain fields (bandwidth, TFLOPS) cannot be queried.

    Args:
        device_id: CUDA device index.

    Returns:
        GPUSpec with all fields populated.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    props = torch.cuda.get_device_properties(device_id)
    device_name = props.name
    cc = (props.major, props.minor)

    # Try to match known GPU for bandwidth/TFLOPS (not queryable via torch)
    known = _match_known_gpu(device_name)

    # Runtime-queryable properties
    num_sm = props.multi_processor_count
    max_threads_per_sm = props.max_threads_per_multi_processor
    max_warps_per_sm = max_threads_per_sm // 32
    max_threads_per_block = getattr(props, 'max_threads_per_block', 1024)
    max_regs_per_block = props.regs_per_multiprocessor  # per SM actually
    shared_mem_per_block = getattr(props, 'shared_memory_per_block_optin', getattr(props, 'max_shared_memory_per_block_optin', 49152))
    shared_mem_per_sm = getattr(props, 'shared_memory_per_multiprocessor', shared_mem_per_block)  # approximate
    global_mem_size = props.total_memory
    l2_cache_size = props.l2_cache_size if hasattr(props, 'l2_cache_size') else (
        known.l2_cache_size if known else 40 * 1024 * 1024
    )

    # Bandwidth and TFLOPS: use known DB or estimate
    if known:
        mem_bandwidth_gbps = known.mem_bandwidth_gbps
        peak_fp16_tflops = known.peak_fp16_tflops
        peak_fp32_tflops = known.peak_fp32_tflops
        # Use known smem/regs if runtime query seems off
        if shared_mem_per_block == 0:
            shared_mem_per_block = known.shared_mem_per_block
        shared_mem_per_sm = known.shared_mem_per_sm
        max_regs_per_sm = known.max_regs_per_sm
    else:
        # Rough estimation from clock speed
        clock_ghz = props.clock_rate / 1e6  # kHz -> GHz
        # Estimate: 2 FMA ops × cores_per_sm × num_sm × clock
        cores_per_sm = _estimate_cuda_cores_per_sm(cc)
        peak_fp32_tflops = 2 * cores_per_sm * num_sm * clock_ghz / 1e3
        # FP16 tensor core: roughly 8x-16x FP32 for Ampere+
        tc_multiplier = 16 if cc[0] >= 8 else 8
        peak_fp16_tflops = peak_fp32_tflops * tc_multiplier
        # Memory bandwidth from bus width (not always available)
        mem_bandwidth_gbps = _estimate_bandwidth(props)
        max_regs_per_sm = max_regs_per_block
        shared_mem_per_sm = shared_mem_per_block

    return GPUSpec(
        name=device_name,
        compute_capability=cc,
        num_sm=num_sm,
        max_threads_per_sm=max_threads_per_sm,
        max_warps_per_sm=max_warps_per_sm,
        warp_size=32,
        max_threads_per_block=max_threads_per_block,
        max_regs_per_block=max_regs_per_block,
        max_regs_per_sm=max_regs_per_sm,
        shared_mem_per_block=shared_mem_per_block,
        shared_mem_per_sm=shared_mem_per_sm,
        l2_cache_size=l2_cache_size,
        global_mem_size=global_mem_size,
        mem_bandwidth_gbps=mem_bandwidth_gbps,
        peak_fp16_tflops=peak_fp16_tflops,
        peak_fp32_tflops=peak_fp32_tflops,
    )


def _estimate_cuda_cores_per_sm(cc: tuple) -> int:
    major = cc[0]
    if major >= 9:      # Hopper
        return 128
    elif major >= 8:    # Ampere / Ada
        return 128
    elif major >= 7:    # Volta / Turing
        return 64
    elif major >= 6:    # Pascal
        return 128 if cc[1] == 1 else 64
    else:
        return 128


def _estimate_bandwidth(props) -> float:
    # memory_clock_rate is in kHz, memory_bus_width in bits
    if hasattr(props, 'memory_clock_rate') and hasattr(props, 'memory_bus_width'):
        clock_ghz = props.memory_clock_rate / 1e6
        bus_bytes = props.memory_bus_width / 8
        # DDR: effective rate = 2x clock
        return 2 * clock_ghz * bus_bytes
    return 1000.0  # conservative default

_cached_spec: Optional[GPUSpec] = None


def get_gpu_spec(device_id: int = 0, force_refresh: bool = False) -> GPUSpec:
    """Get cached GPU spec (singleton per process)."""
    global _cached_spec
    if _cached_spec is None or force_refresh:
        _cached_spec = detect_gpu(device_id)
    return _cached_spec
