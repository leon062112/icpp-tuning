"""
Configuration search space generation with LCM marginal-stratified sampling
and hardware-aware pruning.

Three initialization modes:
  1. Cold Start: Prune first, then LCM marginal sampling within valid configs
  2. Structure-aware Cold Start: Stratified sampling over known valid configs
  3. Warm Start: Performance DB top-K + LCM exploration supplement

Hardware pruning levels:
  L1: Shared memory constraint
  L2: Register pressure constraint
  L3: Minimum occupancy (occupancy >= threshold)

Key design principle: "Prune first, then sample" ensures LCM's marginal
uniformity guarantee is preserved in the final output.
"""

import math
import random
from typing import Dict, List, Optional

from kernel.autotuner.gpu_spec import GPUSpec
from kernel.autotuner.cost_model import TuneConfig


# ---------------------------------------------------------------------------
# Backend-specific parameter value pools
# ---------------------------------------------------------------------------

# Triton: block dims must be power-of-2 >= 16
_TRITON_MATMUL_PARAMS = {
    "block_M":     [32, 64, 128],
    "block_N":     [32, 64, 128, 256],
    "block_K":     [32, 64],
    "num_stages":  [2, 3, 4, 5],
    "num_warps":   [2, 4, 8],
}

# TileLang: block dims must be multiples of 16 (MMA constraint)
# Non-pow2 block_M (48, 80, 96, 112) use FullCol warp policy
_TILELANG_MATMUL_PARAMS = {
    "block_M":     [32, 48, 64, 80, 96, 112, 128],
    "block_N":     [64, 128, 256],
    "block_K":     [32, 64],
    "num_stages":  [2, 3, 4],
    "num_warps":   [4, 8],
}

# Conv2d Triton: power-of-2 block dims
_TRITON_CONV2D_PARAMS = {
    "block_M":     [32, 64, 128],
    "block_N":     [32, 64, 128],
    "block_K":     [16, 32, 64],
    "num_stages":  [2, 3, 4, 5],
    "num_warps":   [2, 4, 8],
}

# Conv2d TileLang: multiples of 16
_TILELANG_CONV2D_PARAMS = {
    "block_M":     [32, 48, 64, 96, 128],
    "block_N":     [32, 64, 128],
    "block_K":     [16, 32, 64],
    "num_stages":  [2, 3, 4],
    "num_warps":   [4, 8],
}


def _get_param_pools(op_type: str, backend: str) -> Dict[str, List[int]]:
    """Get parameter value pools for given op_type and backend."""
    if op_type == "matmul":
        if backend == "triton":
            return _TRITON_MATMUL_PARAMS.copy()
        return _TILELANG_MATMUL_PARAMS.copy()
    elif op_type == "conv2d":
        if backend == "triton":
            return _TRITON_CONV2D_PARAMS.copy()
        return _TILELANG_CONV2D_PARAMS.copy()
    raise ValueError(f"Unsupported op_type: {op_type}")


# ---------------------------------------------------------------------------
# LCM Marginal-Stratified Sampling (Cold Start Core)
# ---------------------------------------------------------------------------

def _lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b)


def _multi_lcm(values: List[int]) -> int:
    result = values[0]
    for v in values[1:]:
        result = _lcm(result, v)
    return result


def lcm_marginal_sample(
    param_pools: Dict[str, List[int]],
    n_samples: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[Dict[str, int]]:
    """LCM marginal-stratified sampling for unbiased config generation.

    Each parameter value appears with equal frequency across the sample set,
    ensuring unbiased marginal coverage without Cartesian product explosion.

    Algorithm:
      1. Compute LCM of all parameter pool sizes
      2. Replicate each pool to length = LCM (uniform frequency)
      3. Independently shuffle each pool
      4. Zip by index to form configs

    Args:
        param_pools: {param_name: [possible_values]}
        n_samples: Number of configs to generate. Default = LCM.
        seed: Random seed for reproducibility.

    Returns:
        List of config dicts with guaranteed marginal uniformity.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    param_names = list(param_pools.keys())
    pool_sizes = [len(param_pools[name]) for name in param_names]

    # Step 1: LCM of all pool sizes
    L = _multi_lcm(pool_sizes)

    # Default: generate LCM configs (one full cycle of uniform coverage)
    if n_samples is None:
        n_samples = L

    # Step 2: Replicate each pool to length = max(L, n_samples)
    target_len = max(L, n_samples)
    expanded_pools = {}
    for name in param_names:
        pool = param_pools[name]
        # Repeat pool to cover target length, each value appears equally
        reps = math.ceil(target_len / len(pool))
        expanded = (pool * reps)[:target_len]
        expanded_pools[name] = expanded

    # Step 3: Independent random shuffle
    for name in param_names:
        rng.shuffle(expanded_pools[name])

    # Step 4: Zip by index
    configs = []
    for i in range(n_samples):
        cfg_dict = {name: expanded_pools[name][i] for name in param_names}
        configs.append(cfg_dict)

    return configs


def _dict_to_tuneconfig(d: Dict[str, int]) -> TuneConfig:
    """Convert a config dict to TuneConfig."""
    return TuneConfig(
        block_M=d["block_M"],
        block_N=d["block_N"],
        block_K=d["block_K"],
        num_stages=d["num_stages"],
        num_warps=d["num_warps"],
    )


# ---------------------------------------------------------------------------
# Full Cartesian enumeration (exhaustive, for offline profiling)
# ---------------------------------------------------------------------------

def generate_exhaustive_configs(
    op_type: str = "matmul",
    backend: str = "tilelang",
) -> List[TuneConfig]:
    """Generate full Cartesian product of config space.

    Used for offline profiling where we want complete coverage.
    Apply hardware pruning after this.
    """
    pools = _get_param_pools(op_type, backend)
    configs = []
    for bm in pools["block_M"]:
        for bn in pools["block_N"]:
            for bk in pools["block_K"]:
                for ns in pools["num_stages"]:
                    for nw in pools["num_warps"]:
                        cfg = TuneConfig(
                            block_M=bm, block_N=bn, block_K=bk,
                            num_stages=ns, num_warps=nw,
                        )
                        if cfg.is_valid_for_backend(backend):
                            configs.append(cfg)
    return configs


# ---------------------------------------------------------------------------
# Occupancy estimation (used by pruning and cost model)
# ---------------------------------------------------------------------------

# Default minimum occupancy threshold for pruning.
# Configs below this are unlikely to be performant.
MIN_OCCUPANCY_THRESHOLD = 0.2


def estimate_occupancy(config: TuneConfig, hw: GPUSpec) -> float:
    """Estimate theoretical occupancy (fraction of max warps active per SM).

    Considers thread, shared memory, and register limitations.
    """
    warps_per_block = config.num_warps
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

    blocks_per_sm = min(blocks_by_threads, blocks_by_smem, blocks_by_regs, hw.max_blocks_per_sm)
    blocks_per_sm = max(blocks_per_sm, 0)

    active_warps = blocks_per_sm * warps_per_block
    occupancy = active_warps / hw.max_warps_per_sm if hw.max_warps_per_sm > 0 else 0.0
    return min(1.0, occupancy)


# ---------------------------------------------------------------------------
# Three-level hardware pruning
# ---------------------------------------------------------------------------

def prune_shared_memory(configs: List[TuneConfig], hw: GPUSpec) -> List[TuneConfig]:
    """L1: Remove configs exceeding shared memory capacity."""
    max_smem = hw.shared_mem_per_block
    return [c for c in configs if c.shared_mem_bytes <= max_smem]


def prune_register_pressure(configs: List[TuneConfig], hw: GPUSpec) -> List[TuneConfig]:
    """L2: Remove configs with excessive register pressure."""
    max_regs = hw.max_regs_per_block
    return [c for c in configs if c.regs_per_thread_estimate * c.threads <= max_regs]


def prune_minimum_occupancy(
    configs: List[TuneConfig],
    hw: GPUSpec,
    min_occupancy: float = MIN_OCCUPANCY_THRESHOLD,
) -> List[TuneConfig]:
    """L3: Remove configs with insufficient occupancy.

    Uses estimate_occupancy() to compute theoretical occupancy and filters
    configs below the threshold. This is stricter than just requiring >=1
    block per SM — low-occupancy configs are almost never performant.

    Args:
        configs: Candidate configurations.
        hw: GPU hardware spec.
        min_occupancy: Minimum occupancy fraction (default 0.2 = 20%).
    """
    result = []
    for c in configs:
        if c.threads > hw.max_threads_per_block:
            continue
        if c.threads > hw.max_threads_per_sm:
            continue
        if c.shared_mem_bytes > hw.shared_mem_per_sm:
            continue
        if estimate_occupancy(c, hw) < min_occupancy:
            continue
        result.append(c)
    return result


def prune_configs(configs: List[TuneConfig], hw: GPUSpec) -> List[TuneConfig]:
    """Apply all three pruning levels sequentially."""
    c1 = prune_shared_memory(configs, hw)
    c2 = prune_register_pressure(c1, hw)
    c3 = prune_minimum_occupancy(c2, hw)
    return c3


# ---------------------------------------------------------------------------
# Three initialization modes
# ---------------------------------------------------------------------------

def _stratified_sample_from_valid(
    valid_configs: List[TuneConfig],
    n_samples: Optional[int] = None,
    seed: int = 42,
) -> List[TuneConfig]:
    """Stratified sampling within a valid config set, preserving marginal uniformity.

    Strategy:
      1. Extract actual parameter value pools from the valid set
      2. Use LCM to determine the natural sample size (or use n_samples)
      3. For each LCM-generated combination, find the closest match in the
         valid set using a greedy dimension-priority assignment
      4. Fallback: if n_samples exceeds what LCM matching can provide,
         supplement with random samples from uncovered configs

    This avoids the "generate then reject" anti-pattern: we directly select
    from the valid set while maximizing marginal uniformity.

    Args:
        valid_configs: Already-pruned valid configurations.
        n_samples: Target number of output configs. Default = LCM of pool sizes
            (capped at len(valid_configs)).
        seed: Random seed.

    Returns:
        Subset of valid_configs with approximately uniform marginal coverage.
    """
    if not valid_configs:
        return []

    rng = random.Random(seed)

    # Extract observed parameter value pools
    pools = {
        "block_M": sorted(set(c.block_M for c in valid_configs)),
        "block_N": sorted(set(c.block_N for c in valid_configs)),
        "block_K": sorted(set(c.block_K for c in valid_configs)),
        "num_stages": sorted(set(c.num_stages for c in valid_configs)),
        "num_warps": sorted(set(c.num_warps for c in valid_configs)),
    }

    pool_sizes = [len(v) for v in pools.values()]
    lcm_size = _multi_lcm(pool_sizes)

    # Determine target count.
    # Use at least 2*LCM or 30 configs (whichever is larger) to ensure
    # sufficient coverage, especially for backends with small LCM (e.g.,
    # Triton LCM=12 covers only 4.7% of the exhaustive space).
    min_reasonable = max(2 * lcm_size, 30)
    if n_samples is None:
        n_samples = min(min_reasonable, len(valid_configs))
    else:
        n_samples = min(n_samples, len(valid_configs))

    if n_samples >= len(valid_configs):
        # Want all configs — just shuffle for randomness
        result = list(valid_configs)
        rng.shuffle(result)
        return result

    # Build index: group configs by block_M for stratified picking
    # Strategy: pick one config per "stratum" (block_M bucket), cycling through
    # all parameter values as evenly as possible.
    from collections import defaultdict

    # Group by the primary tiling dimension (block_M) for stratification
    by_block_m: Dict[int, List[TuneConfig]] = defaultdict(list)
    for c in valid_configs:
        by_block_m[c.block_M].append(c)

    # Shuffle within each bucket
    for key in by_block_m:
        rng.shuffle(by_block_m[key])

    # Round-robin across block_M values to ensure each block_M appears equally
    block_m_values = sorted(by_block_m.keys())

    selected = []
    selected_keys = set()

    # Phase 1: Round-robin stratified pick across block_M
    # Within each block_M, pick configs that maximize coverage of other dims
    bucket_cursors = {bm: 0 for bm in block_m_values}

    while len(selected) < n_samples:
        made_progress = False
        for bm in block_m_values:
            if len(selected) >= n_samples:
                break
            bucket = by_block_m[bm]
            cursor = bucket_cursors[bm]
            # Find next unused config in this bucket
            while cursor < len(bucket):
                c = bucket[cursor]
                key = (c.block_M, c.block_N, c.block_K, c.num_stages, c.num_warps)
                cursor += 1
                if key not in selected_keys:
                    selected.append(c)
                    selected_keys.add(key)
                    made_progress = True
                    break
            bucket_cursors[bm] = cursor

        if not made_progress:
            break

    # Phase 2: If still short, fill from remaining configs
    if len(selected) < n_samples:
        remaining = [
            c for c in valid_configs
            if (c.block_M, c.block_N, c.block_K, c.num_stages, c.num_warps) not in selected_keys
        ]
        rng.shuffle(remaining)
        for c in remaining:
            if len(selected) >= n_samples:
                break
            selected.append(c)

    return selected

def cold_start_configs(
    op_type: str,
    backend: str,
    hw: GPUSpec,
    n_samples: Optional[int] = None,
    seed: int = 42,
) -> List[TuneConfig]:
    """Cold Start: Prune first, then LCM marginal sampling within valid configs.

    Strategy: "prune first, sample second" to preserve LCM's marginal
    uniformity guarantee in the final output.

    Algorithm:
      1. Generate exhaustive configs + apply hardware pruning → valid set
      2. Extract actual parameter value pools from valid configs
      3. Apply LCM sampling restricted to the valid set

    This ensures the final output has unbiased marginal coverage across
    all parameter dimensions that actually survive hardware constraints.

    Args:
        op_type: "matmul" or "conv2d"
        backend: "triton" or "tilelang"
        hw: GPU spec for pruning.
        n_samples: Desired number of output configs. Default = LCM of valid
            parameter pool sizes.
        seed: Random seed for reproducibility.

    Returns:
        List of valid TuneConfigs with uniform marginal distribution
        (uniformity preserved post-pruning).
    """
    # Step 1: Get the full valid config set (exhaustive + prune)
    all_valid = generate_exhaustive_configs(op_type, backend)
    all_valid = prune_configs(all_valid, hw)

    if not all_valid:
        return []

    # Step 2: Apply stratified sampling within the valid set
    return _stratified_sample_from_valid(all_valid, n_samples=n_samples, seed=seed)


def structure_aware_cold_start_configs(
    valid_configs: List[TuneConfig],
    hw: GPUSpec,
    n_samples: Optional[int] = None,
    seed: int = 42,
) -> List[TuneConfig]:
    """Structure-aware Cold Start: Stratified sampling over known valid configs.

    When valid_config.yaml exists but no performance data. Applies stratified
    sampling directly over the valid config set to guarantee approximate
    marginal uniformity without rejection.

    Unlike the old approach (LCM sample → reject invalid), this directly
    operates on the valid set, so no samples are wasted.

    Args:
        valid_configs: Pre-verified valid configurations.
        hw: GPU spec for additional pruning (e.g., different GPU).
        n_samples: Number to sample. Default = LCM of observed pool sizes
            (capped at len(valid_configs)).
        seed: Random seed.

    Returns:
        Subset of valid_configs with approximately uniform marginal coverage.
    """
    if not valid_configs:
        return []

    # Apply hardware pruning (valid_configs may come from a different GPU)
    pruned = prune_configs(valid_configs, hw)
    if not pruned:
        return []

    return _stratified_sample_from_valid(pruned, n_samples=n_samples, seed=seed)


def warm_start_configs(
    top_k_configs: List[TuneConfig],
    op_type: str,
    backend: str,
    hw: GPUSpec,
    n_explore: Optional[int] = None,
    seed: int = 42,
) -> List[TuneConfig]:
    """Warm Start: Top-K exploitation + LCM exploration supplement.

    When Performance DB exists (same operator, different shapes).
    Combines high-performance candidates with unbiased exploration.

    Args:
        top_k_configs: Top-K configs from Performance DB (exploitation seeds).
        op_type: "matmul" or "conv2d"
        backend: "triton" or "tilelang"
        hw: GPU spec for pruning.
        n_explore: Number of exploration configs to add. Default = len(top_k).
        seed: Random seed.

    Returns:
        Merged list: [exploitation seeds (first)] + [exploration seeds].
    """
    if n_explore is None:
        n_explore = max(len(top_k_configs), 10)

    # Exploitation: prune top-K to ensure hardware validity
    exploit = prune_configs(top_k_configs, hw)

    # Exploration: LCM sample to fill gaps
    explore = cold_start_configs(
        op_type, backend, hw, n_samples=n_explore * 2, seed=seed
    )

    # Merge: exploitation first (higher priority), then exploration
    seen = set(
        (c.block_M, c.block_N, c.block_K, c.num_stages, c.num_warps)
        for c in exploit
    )

    merged = list(exploit)
    for c in explore:
        key = (c.block_M, c.block_N, c.block_K, c.num_stages, c.num_warps)
        if key not in seen:
            seen.add(key)
            merged.append(c)
            if len(merged) >= len(exploit) + n_explore:
                break

    return merged


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

def generate_search_space(
    op_type: str,
    hw: GPUSpec,
    backend: str = "tilelang",
    mode: str = "exhaustive",
    n_samples: Optional[int] = None,
    valid_configs: Optional[List[TuneConfig]] = None,
    top_k_configs: Optional[List[TuneConfig]] = None,
    seed: int = 42,
) -> List[TuneConfig]:
    """Generate pruned search space with specified initialization mode.

    Args:
        op_type: "matmul" or "conv2d"
        hw: GPU hardware spec for pruning.
        backend: "triton" or "tilelang"
        mode: Initialization mode:
            "exhaustive" — full Cartesian product (for offline profiling)
            "cold" — LCM marginal sampling (no prior knowledge)
            "structure" — LCM within known valid configs
            "warm" — Top-K + LCM exploration
        n_samples: Number of configs for sampling modes.
        valid_configs: Required for mode="structure".
        top_k_configs: Required for mode="warm".
        seed: Random seed for sampling modes.

    Returns:
        List of valid TuneConfigs after hardware pruning.
    """
    if mode == "exhaustive":
        full = generate_exhaustive_configs(op_type, backend)
        return prune_configs(full, hw)

    elif mode == "cold":
        return cold_start_configs(op_type, backend, hw, n_samples, seed)

    elif mode == "structure":
        if valid_configs is None:
            raise ValueError("mode='structure' requires valid_configs")
        return structure_aware_cold_start_configs(valid_configs, hw, n_samples, seed)

    elif mode == "warm":
        if top_k_configs is None:
            raise ValueError("mode='warm' requires top_k_configs")
        return warm_start_configs(top_k_configs, op_type, backend, hw, n_samples, seed)

    else:
        raise ValueError(f"Unsupported mode: {mode}")


# ---------------------------------------------------------------------------
# Shape sampling for offline profiling
# ---------------------------------------------------------------------------

def generate_matmul_sample_shapes(
    M_range: tuple = (1, 4096),
    N_values: Optional[List[int]] = None,
    K_values: Optional[List[int]] = None,
) -> List[dict]:
    """Generate representative M sample points for offline profiling.

    Strategy: dense at small M (wave count changes rapidly),
    sparser at large M (behavior more stable).
    """
    if N_values is None:
        N_values = [768, 2304, 4096]
    if K_values is None:
        K_values = [768, 2304, 4096]

    M_samples = set()
    for m in range(1, 17):
        M_samples.add(m)
    for m in range(16, 129, 16):
        M_samples.add(m)
    for m in range(128, 1025, 64):
        M_samples.add(m)
    for m in range(1024, min(M_range[1] + 1, 4097), 256):
        M_samples.add(m)

    M_samples = sorted(m for m in M_samples if M_range[0] <= m <= M_range[1])

    shapes = []
    for m in M_samples:
        for n in N_values:
            for k in K_values:
                shapes.append({"M": m, "N": n, "K": k})
    return shapes


def generate_conv2d_sample_shapes(
    N_values: Optional[List[int]] = None,
    HW_values: Optional[List[int]] = None,
    C_values: Optional[List[int]] = None,
    OC_values: Optional[List[int]] = None,
) -> List[dict]:
    """Generate representative conv2d shapes for offline profiling."""
    if N_values is None:
        N_values = [1, 2, 4, 8, 16, 32]
    if HW_values is None:
        HW_values = [7, 14, 28, 56, 112, 224]
    if C_values is None:
        C_values = [64, 128, 256, 512]
    if OC_values is None:
        OC_values = [64, 128, 256, 512]

    shapes = []
    for n in N_values:
        for hw in HW_values:
            for c in C_values:
                for oc in OC_values:
                    shapes.append({
                        "N": n, "C": c, "H": hw, "W": hw,
                        "OC": oc, "KH": 3, "KW": 3,
                        "stride": 1, "padding": 1,
                    })
    return shapes


# ---------------------------------------------------------------------------
# Summary / diagnostics
# ---------------------------------------------------------------------------

def search_space_summary(
    op_type: str,
    hw: GPUSpec,
    backend: str = "tilelang",
    mode: str = "exhaustive",
    **kwargs,
) -> dict:
    """Return search space statistics for diagnostics."""
    full = generate_exhaustive_configs(op_type, backend)
    after_l1 = prune_shared_memory(full, hw)
    after_l2 = prune_register_pressure(after_l1, hw)
    after_l3 = prune_minimum_occupancy(after_l2, hw)

    # Also generate with the requested mode for comparison
    mode_configs = generate_search_space(
        op_type, hw, backend, mode=mode, **kwargs
    )

    # Occupancy distribution of final pruned set
    occupancies = [estimate_occupancy(c, hw) for c in after_l3]
    avg_occ = sum(occupancies) / len(occupancies) if occupancies else 0

    stats = {
        "op_type": op_type,
        "backend": backend,
        "mode": mode,
        "exhaustive_full": len(full),
        "after_smem_prune": len(after_l1),
        "after_reg_prune": len(after_l2),
        "after_occupancy_prune": len(after_l3),
        "min_occupancy_threshold": MIN_OCCUPANCY_THRESHOLD,
        "avg_occupancy": round(avg_occ, 3),
        "mode_output_size": len(mode_configs),
        "pruning_ratio": round(1 - len(after_l3) / len(full), 3) if full else 0,
    }
    return stats
