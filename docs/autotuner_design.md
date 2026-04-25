# 面向动态形状的自动调优方案设计 (v8)

## 设计原则

**离线深度调优 + 在线轻量选择**，在线选优核心公式：

```
Score_final(S, C, HW) = Score_formula(S, C, HW)
```

即 **"Trait-Driven 七因子乘积解析公式"** 直接评分选优。公式基于 GPU 物理规律（Roofline、occupancy、wave 填充、pipeline 流水线等），通过乘积式组合实现"木桶效应"——任一因子的瓶颈直接拉低总分。

---

## 支持算子列表

| 算子 | 动态维度 | 融合类型 |
|:--|:--|:--|
| matmul | M, N, K | 单算子 |
| matmul + bias + act (ReLU/GELU/SiLU) | M, N, K | 计算融合 |
| matmul + bias + layernorm | M, N, K | 计算+归约融合 |
| conv2d | N, H, W | 单算子 |
| conv2d + relu | N, H, W | 计算融合 |
| conv2d + batchnorm | N, H, W | 计算+归约融合 |

---

## Trait-Driven 算子描述体系

> 实现: `kernel/autotuner/cost_model_v2.py`

为了在同一套评分框架下统一支持 matmul/conv2d 全系列融合算子，v2 Cost Model 引入了 **Trait-Driven** 设计：不再针对每种算子编写专用评分逻辑，而是将算子分解为可组合的执行特征（Trait），评分因子消费特征而非硬编码算子名称。

### 三层抽象

```
OpSpec（算子描述）
 ├── MainloopSpec        — 主循环计算结构（GEMM / implicit-GEMM Conv）
 │     ├── kind: "gemm" | "implicit_gemm_conv"
 │     ├── M, N, K                — 等价 GEMM 维度
 │     ├── input/weight/output_elements — 实际访存元素数
 │     ├── element_bytes          — 各操作数字节宽度
 │     ├── input/weight/spatial_reuse_hint — 数据复用提示（Conv 空间重叠）
 │     └── preferred_group_m      — L2 tiling group 偏好
 │
 └── epilogue: Tuple[FusionPrimitive, ...]  — 融合尾部原语序列
       └── FusionPrimitive
             ├── name                    — 名称标识
             ├── flops_per_output        — 每输出元素 FLOPs
             ├── bytes_per_output        — 每输出元素额外访存字节
             ├── reduction_axis          — 归约轴: "none"|"m"|"n"|"k"|"spatial"|"channel"
             ├── extra_passes            — 额外遍历次数（如 layernorm 的两遍扫描）
             ├── sync_penalty            — 同步开销系数
             ├── preferred_outputs_per_thread — 最优每线程输出数
             ├── preferred_tile_m/n      — 偏好 tile 尺寸
             └── smem_bytes_per_output   — 每输出元素 shared memory 开销
```

### 内置原语库 (Primitive Library)

| 原语 | flops/output | bytes/output | reduction_axis | extra_passes | sync_penalty |
|:--|:--|:--|:--|:--|:--|
| `bias_add` | 1 | 0 | none | 0 | 0 |
| `relu` | 1 | 0 | none | 0 | 0 |
| `silu` | 4 | 0 | none | 0 | 0 |
| `gelu` | 8 | 0 | none | 0 | 0 |
| `row_layernorm` | 10 | 4 | n | 1 | 0.35 |
| `batchnorm` | 6 | 2 | channel | 1 | 0.25 |

**组合示例**：
- `matmul + bias + relu` → `OpSpec(mainloop=gemm, epilogue=(bias_add, relu))`
- `conv2d + batchnorm` → `OpSpec(mainloop=implicit_gemm_conv, epilogue=(batchnorm,))`

### ConfigProjection（执行特征提取）

给定一组 `(算子描述, Tile配置, GPU规格)`，预先计算出该配置在该硬件上的各项执行指标（如 tile 访存量、occupancy、wave 填充率等），供后续所有评分因子直接使用，避免重复计算：

| 特征类别 | 字段 | 说明 |
|:--|:--|:--|
| **Tile 结构** | tiles_M/N/K, grid_size, tile_outputs, outputs_per_thread | Tile 网格与线程负载 |
| **计算量** | mainloop/epilogue/total_flops_per_tile | 每 tile FLOPs |
| **访存量** | input/weight/output/epilogue_bytes_per_tile, total_global_bytes | 考虑操作数复用的实际访存 |
| **Shared Mem** | shared_mem_bytes_per_tile | 含 epilogue smem 开销 |
| **性能模型** | tile_arithmetic_intensity, occupancy, resident_blocks_per_sm | Roofline / 资源占用 |
| **Wave 分析** | wave_fill | GPU 波次填充率 |
| **Pipeline** | preferred_num_stages | 基于算术强度 (Arithmetic Intensity) 和算子特征的推荐 stage 数 |
| **Epilogue** | reduction_work/passes, epilogue_preferred_outputs | 归约与尾部负载特征 |

**操作数复用建模**：
- GEMM: weight_reuse ∝ sqrt(group_m)，input 无复用
- Conv2d: input_reuse = sqrt(spatial_overlap)，weight_reuse ∝ sqrt(group_m × weight_reuse_hint)

---

## 核心公式

### 解析物理评分 (Score_formula)

```
Score_formula = eff_mainloop
             × eff_memory
             × eff_parallel
             × eff_pipeline
             × eff_implicit_conv    (仅 conv2d 生效，GEMM 恒为 1)
             × eff_epilogue         (仅有融合尾部时生效)
             × eff_reduction        (仅有归约原语时生效)
```

选择 `C* = argmax_C Score_formula(S, C, HW)`。

#### 七因子详细公式

| 因子 | 含义 | 核心公式 |
|:--|:--|:--|
| `eff_mainloop` | 主循环效率：padding + 线程负载 + block_K | padding_eff × thread_eff × k_eff |
| `eff_memory` | 访存效率：Roofline + Shared Memory 压力 | roofline_eff^0.75 × smem_eff |
| `eff_parallel` | 并行效率：occupancy + wave 填充 + warp 数 | occupancy_eff × wave_eff × warp_eff |
| `eff_pipeline` | 流水线效率：fill 率 + stage 匹配度 | fill_eff × stage_match |
| `eff_implicit_conv` | Conv 特有：空间重叠下的 tile/warp 偏好 | m_eff × n_eff × warp_eff (或 1.0) |
| `eff_epilogue` | 融合尾部效率：加权几何平均各原语效率 | exp(Σ wᵢ × log(prim_effᵢ)) |
| `eff_reduction` | 归约效率：轴匹配 + pass 惩罚 + 同步开销 | Π(axis_eff × pass_eff × sync_eff)^(1/n) |

各因子 ∈ (0, 1]，乘积 ∈ (0, 1]。

#### eff_mainloop 细节

```
padding_eff = (M×N×K) / (ceil(M/BM)×BM × ceil(N/BN)×BN × ceil(K/BK)×BK)

# 动态目标: 并行度越高 → 允许更大 outputs/thread
parallel_signal = sqrt(launch_fill × occupancy_fill)
target_outputs = 2 + (32 - 2) × parallel_signal
outputs_ratio = outputs_per_thread / target_outputs
thread_eff = 1 / (1 + 0.4 × |log₂(outputs_ratio)|)

k_eff = sqrt(min(1, block_K / preferred_block_K))
```

**关键改进**：`thread_eff` 不再使用固定目标值，而是根据 wave 填充率和 occupancy 动态调整——当并行度充足时容忍更大 tile（更高 outputs/thread），当 SM 利用不足时惩罚过大 tile。

#### eff_memory 细节

```
roofline_eff = min(1, tile_AI / ridge_point)
smem_pressure = shared_mem_bytes / max_shared_mem_per_sm
smem_eff = 1 / (1 + 0.25 × smem_pressure)

eff_memory = roofline_eff^0.75 × smem_eff
```

#### eff_parallel 细节

```
occupancy_eff = min(1, occupancy / 0.5)^0.15    # 饱和指数：50% 以上收益递减
wave_eff = wave_fill^0.35
warp_eff = 0.9^max(0, log₂(num_warps/4))        # >4 warps 逐级惩罚

eff_parallel = occupancy_eff × wave_eff × warp_eff
```

#### eff_pipeline 细节

```
fill_eff = (K_iters / (K_iters + num_stages - 1))^0.5
stage_match = 1 / (1 + 0.5 × |num_stages - preferred_stages|)

# preferred_stages 动态计算：
#   基础 = 2; conv → +1; memory_eff < 0.9 → +1; < 0.5 → +1; 有归约 → +1
#   上限 = min(6, K_iters)
```

#### eff_implicit_conv 细节（仅 conv2d 生效）

```
# 仅在 underfill > 0（SM 未填满）时生效
overlap_signal = min(1.5, log₂(spatial_overlap) / log₂(9))
underfill = 1 - grid_size / resident_slots

# 高空间重叠 + 高 underfill → 偏好小 tile
preferred_BN = 64 if overlap≥4 ∧ underfill≥0.2 else 128
preferred_BM = 32 if overlap≥4 ∧ underfill≥0.5 else 64

m_eff = 1 / (1 + 0.45 × overlap_signal × underfill × |log₂(BM/pref_BM)|)
n_eff = 1 / (1 + 0.6  × overlap_signal × underfill × |log₂(BN/pref_BN)|)
warp_eff = 0.94^log₂(num_warps/4)  (当 overlap≥4 ∧ underfill≥0.2 ∧ warps>4)
```

#### eff_epilogue 细节

对每个融合原语计算效率，取加权几何平均（权重 = work_units = max(1, flops + bytes)）：

```
prim_eff(p) = 1 / (1 + 0.5 × |log₂(outputs_per_thread / p.preferred_outputs)|)
            × (tile_m 偏好匹配) × (tile_n 偏好匹配)

eff_epilogue = exp(Σ (wᵢ/W) × log(prim_effᵢ))   # 加权几何平均
```

#### eff_reduction 细节

对每个归约原语计算：

```
axis_penalty = 1 / (1 + 0.3 × |log₂(axis_extent / preferred_extent)|)
pass_penalty = 1 / (1 + 0.2 × extra_passes)
sync_penalty = 1 / (1 + sync_penalty_coeff)

eff_reduction = Π(axis × pass × sync)^(1/n)   # 几何平均
```

#### 模型常量一览

| 常量 | 值 | 含义 |
|:--|:--|:--|
| `_PREFERRED_OCCUPANCY` | 0.5 | occupancy 达标阈值 |
| `_OCCUPANCY_SATURATION_EXPONENT` | 0.15 | occupancy 饱和指数 |
| `_WAVE_FILL_EXPONENT` | 0.35 | wave 填充惩罚强度 |
| `_HIGH_WARP_COUNT_PENALTY` | 0.9 | 每倍增 warp 的衰减系数 |
| `_MEMORY_EFFICIENCY_EXPONENT` | 0.75 | roofline 效率指数 |
| `_PIPELINE_FILL_EXPONENT` | 0.5 | pipeline fill 指数 |
| `_STAGE_MISMATCH_PENALTY` | 0.5 | stage 不匹配惩罚 |
| `_MAINLOOP_TARGET_OUTPUTS_PER_THREAD` | 32.0 | 最大每线程输出目标 |
| `_MAINLOOP_MIN_OUTPUTS_PER_THREAD` | 2.0 | 最小每线程输出目标 |
| `_MAINLOOP_OUTPUT_DISTANCE_PENALTY` | 0.4 | 输出偏离惩罚 |
| `_EPILOGUE_DISTANCE_PENALTY` | 0.5 | 尾部效率偏离惩罚 |
| `_REDUCTION_DISTANCE_PENALTY` | 0.3 | 归约轴偏离惩罚 |
| `_CONV_BLOCK_M_DISTANCE_PENALTY` | 0.45 | Conv block_M 偏离惩罚 |
| `_CONV_BLOCK_N_DISTANCE_PENALTY` | 0.6 | Conv block_N 偏离惩罚 |
| `_CONV_HIGH_WARP_PENALTY` | 0.94 | Conv 高 warp 数惩罚 |

---

## 系统架构

```
┌───────────────────────────────────────────────────────────────────────┐
│                        离线阶段 (Offline)                               │
│                                                                         │
│  Step 1: 搜索空间生成 + 硬件检测                      (~1s)             │
│      - GPU 规格检测 (GPUSpec)                                           │
│      - Exhaustive 枚举 + 三层硬件剪枝 (含 occupancy ≥ 20%)              │
│      - TileLang matmul: 252 → 86 configs                              │
│      - Triton matmul:   288 → 104 configs                             │
│      - TileLang conv2d: 270 → 183 configs                             │
│      - Triton conv2d:   324 → 168 configs                             │
│  Step 2: 分层采样 Profiling                          (~17min)           │
│  Step 3: 决策表构建                                   (~1s)              │
│  Step 4: Kernel 预编译                                (~2min)            │
│                                                                         │
│  总计: ~20 min (一次性, per device per op_type)                          │
└───────────────────────────────────────────────┬───────────────────────┘
                                                │ 输出: PerformanceDB
                                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│                        在线阶段 (Online)                               │
│                                                                         │
│   Input: shape (M, N, K) or (N, H, W)                                  │
│                                                                         │
│   Level 1: 查决策表 (离线 profiling 过的 shape 区间)                      │
│     → 命中: 直接返回 oracle config       精度 ~99%   开销 < 1μs         │
│                                                                         │
│   Level 2: 解析公式选优 (决策表未命中的 shape)                            │
│     → for each config:                                                  │
│         Score = Score_formula(七因子乘积)                                │
│     → 选 argmax                          精度 ~85%   开销 ~2μs          │
│                                                                         │
│   Output: compiled kernel + config                                      │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 搜索空间初始化方案

### 设计目标

为搜索算法提供高质量的初始配置集，满足：
- **有效性**：覆盖可运行配置
- **无偏性**：避免参数分布偏置
- **泛化性**：支持跨尺寸、跨算子、跨平台
- **后端感知**：Triton (pow2) 与 TileLang (mul16) 约束自动适配

### 核心设计原则："先剪枝，后采样"

> **关键改进**：为保证 LCM 的边缘均匀性保证在最终输出中仍然成立，
> 采用"先做 exhaustive + 硬件剪枝得到 valid 集合 → 再在 valid 集合内做分层采样"的策略。
>
> 旧方案"先 LCM 采样 → 后剪枝"会因非均匀过滤破坏原始分布，
> 导致某些参数值（如大 block_K、高 num_stages）在最终输出中严重欠采样。

### 后端参数约束

**Matmul:**

| 参数 | Triton 约束 | TileLang 约束 |
|:--|:--|:--|
| block_M | {32, 64, 128} (pow2) | {32, 48, 64, 80, 96, 112, 128} (mul16) |
| block_N | {32, 64, 128, 256} (pow2) | {64, 128, 256} (mul16) |
| block_K | {32, 64} (pow2) | {32, 64} (mul16) |
| num_stages | {2, 3, 4, 5} | {2, 3, 4} |
| num_warps | {2, 4, 8} | {4, 8} |
| 额外参数 | GROUP_SIZE_M (L2 tiling) | FullCol policy (非pow2 block_M) |

**Conv2d:**

| 参数 | Triton 约束 | TileLang 约束 |
|:--|:--|:--|
| block_M | {32, 64, 128} (pow2) | {32, 48, 64, 96, 128} (mul16) |
| block_N | {32, 64, 128} (pow2) | {32, 64, 128} (mul16) |
| block_K | {16, 32, 64} (pow2) | {16, 32, 64} (mul16) |
| num_stages | {2, 3, 4, 5} | {2, 3, 4} |
| num_warps | {2, 4, 8} | {4, 8} |

统一数据结构 `TuneConfig` 通过 `.to_triton_dict()` / `.to_tilelang_dict()` 转换为各自后端的启动参数。

### 三种初始化模式

#### 模式 1：完全冷启动 (Cold Start) — 先剪枝后分层采样

无任何先验信息时，保证无偏覆盖：

```
算法：Prune-First Stratified Sampling

Step 1: 生成 exhaustive 全笛卡尔积
  TileLang matmul: 7×3×2×3×2 = 252 configs
  Triton matmul:   3×4×2×4×3 = 288 configs

Step 2: 三层硬件剪枝 → 得到 valid 集合
  TileLang: 252 → 86 valid configs
  Triton:   288 → 104 valid configs

Step 3: 在 valid 集合内做分层采样
  - 按 block_M 分桶 → 桶内 shuffle → round-robin 轮询
  - 保证各 block_M 值出现频率尽可能均等
  - 默认采样量: max(2 × LCM, 30) (保证足够覆盖)
  - TileLang 输出 ~84 configs, Triton 输出 ~30 configs
```

**性质**：
- 无偏性：在硬件可行集合内，各参数值的边缘分布近似均匀
- 真实性：最终分布忠实反映硬件约束（大 tile 占比小是因为硬件不支持，而非采样偏差）
- 高效性：Triton 从旧方案的 12 configs 提升到 30 configs（覆盖率 4.7% → 28.8%）

**适用**：新平台、新算子，从 0 构建搜索空间。

#### 模式 2：结构冷启动 (Structure-aware Cold Start) — 分层采样

已有 `valid_configs` 但无性能排序时，直接在 valid 集合上做分层采样：

```python
def structure_aware_cold_start_configs(valid_configs, hw, n_samples, seed):
    # 1. 对 valid_configs 做硬件剪枝（可能换了 GPU）
    pruned = prune_configs(valid_configs, hw)

    # 2. 在 pruned 集合内做分层采样
    #    按 block_M 分桶 → round-robin → 近似均匀
    return _stratified_sample_from_valid(pruned, n_samples, seed)
```

**改进点**（对比旧方案）：
- 旧方案：LCM 生成候选 → rejection sampling → 稀疏集合下大量丢弃、分布扭曲
- 新方案：直接在 valid 集合上操作，零浪费，分布可控

**适用**：已知哪些 config 能编译通过，但不知道性能排序。

#### 模式 3：热启动 (Warm Start)

有 Performance DB 时，融合利用种子 + 探索种子：

```
初始种子 = [利用种子 (Exploitation)] + [探索种子 (Exploration)]
              ↓                           ↓
         Top-K from DB               冷启动分层采样补充
         (高性能候选)                 (覆盖未探索区域)
```

分为两类：
- **同算子泛化**：同一 kernel，不同 problem size → Top-K 迁移
- **跨算子弱泛化**：不同算子间 config 迁移 (不保证有效性)

### 三层硬件剪枝 (所有模式通用)

无论何种初始化模式，生成后均执行：

```
L1: shared_mem(config) <= hw.max_shared_mem_per_block        (消除内存溢出)
L2: regs(config) × threads <= hw.max_regs_per_block          (消除寄存器溢出)
L3: estimate_occupancy(config, hw) >= MIN_OCCUPANCY (20%)    (剔除低效配置)
```

**L3 改进说明**：
- 旧方案仅要求"能调度至少 1 block/SM"（即 occupancy > 0%）
- 新方案要求 occupancy ≥ 20%，剔除明显低效的配置
- 使用 `estimate_occupancy()` 综合考虑 thread、shared memory、register 三重限制
- 效果：TileLang matmul 从 229 → 86 configs（剔除 143 个低效配置）

### 分层采样核心算法 (`_stratified_sample_from_valid`)

```python
def _stratified_sample_from_valid(valid_configs, n_samples, seed):
    """在 valid config 集合内做分层采样。

    Algorithm:
      1. 按 block_M 分桶（主分层维度）
      2. 桶内随机 shuffle
      3. Round-robin 轮询：依次从每个 block_M 桶中取 1 个 config
      4. 若目标数量仍不够，从剩余 configs 随机补充

    性质:
      - block_M 各值出现次数差 ≤ 1（接近完美均匀）
      - 桶内 shuffle 保证其他维度 (block_N, block_K 等) 的弱均匀
      - 对 n=42 (TileLang LCM): block_M 分布 {32:7, 48:7, 64:6, 80:6, 96:6, 112:5, 128:5}
        其中 96/112/128 仅 5-6 个是因为硬件 valid 集合本身只有这么多

    默认采样量:
      n_samples = min(max(2 × LCM, 30), len(valid_configs))
      → TileLang: min(max(84, 30), 86) = 84
      → Triton:   min(max(24, 30), 104) = 30
    """
```

### 各模式对比

| 模式 | 输入要求 | 输出规模 (A100) | 覆盖度 | 适用场景 |
|:--|:--|:--|:--|:--|
| Cold (分层采样) | 无 | ~84 (TL) / ~30 (Triton) | valid 集合内近似均匀 | 新平台/新算子 |
| Structure | valid_configs | ~N (可指定) | 已验证空间内分层均匀 | 有可行性信息 |
| Warm | Performance DB | Top-K + 探索 | 高性能区 + 补充 | 有历史数据 |
| Exhaustive | 无 | ~86 (TL) / ~104 (Triton) | 完全覆盖 | 离线 profiling |

---

## 离线阶段详细设计

### Step 1: 搜索空间生成 + 硬件检测

```python
def generate_offline_search_space(op_type: str, backend: str, hw: GPUSpec) -> List[TuneConfig]:
    """离线阶段使用 exhaustive 模式获取完整候选集。

    TileLang matmul: 7×3×2×3×2 = 252 → 剪枝后 86
    Triton matmul:   3×4×2×4×3 = 288 → 剪枝后 104
    TileLang conv2d: 5×3×3×3×2 = 270 → 剪枝后 183
    Triton conv2d:   3×3×3×4×3 = 324 → 剪枝后 168

    三层剪枝:
      L1: shared_mem(config) <= hw.max_shared_mem_per_block
      L2: regs(config) × threads <= hw.max_regs_per_block
      L3: estimate_occupancy(config, hw) >= 0.2  (20% 最低 occupancy)
    """
```

### Step 2: 分层采样 Profiling

```python
def build_sampling_plan(op_type, shape_ranges, hw) -> List[Dict]:
    """按 wave count 分层，每层选代表性点。

    Matmul: 22 M点 × 9 (N,K)组合 = 198 shapes
    Conv2d: 6 HW点 × 6 N点 = 36 shapes
    """

def run_profiling(shapes, configs, ...) -> ProfilingResults:
    """并行编译 + benchmark。

    优化: 动态形状 kernel 每个 config 编译一次, 复用于所有 shape。
    总实测: ~10,000 次, 总耗时 ~17min。
    """
```

### Step 3: 决策表构建

```python
def build_decision_table(profiling: ProfilingResults) -> DecisionTable:
    """按 (N,K) 分组, M 轴区间划分, 每区间记录 oracle config。"""
```

### Step 4: Kernel 预编译

```python
def precompile_top_configs(configs, decision_table, ...) -> Dict[int, object]:
    """预编译决策表涉及的 5-10 个 config 对应的动态 kernel。"""
```

---

## 在线阶段详细设计

```python
class AdaptiveTuner:
    """统一自适应调优器: 查表 → 公式选优。"""

    def __init__(self, op_type: str, backend: str = "tilelang"):
        self.op_type = op_type
        self.backend = backend
        self.hw = detect_gpu()
        self.db: Optional[PerformanceDB] = None
        self.kernel_cache: Dict[int, object] = {}
        self._load_offline_artifacts()

    def select(self, shape: OpShape) -> SelectResult:
        """两级递进选优。"""

        # Level 1: 决策表
        if self.db and self.db.decision_table:
            idx = self.db.decision_table.lookup(shape.to_dict())
            if idx is not None:
                return self._make_result(idx, level="table")

        # Level 2: 解析公式选优
        idx = self._formula_select(shape)
        return self._make_result(idx, level="formula")

    def _formula_select(self, shape: OpShape) -> int:
        """七因子解析公式选优。"""
        configs = self.db.config_pool if self.db else cold_start_configs(self.op_type, self.backend, self.hw)

        # 构建 OpSpec
        op = make_op_spec(self.op_type, shape)

        # 评分并选优
        scores = [score_formula(op, cfg, self.hw) for cfg in configs]
        return int(np.argmax(scores))
```

---

## 精度预期

| 选优策略 | Top-1 准确率 | 性能比 (vs oracle) | 适用场景 |
|:--|:--|:--|:--|
| Level 1: 决策表 | ~95-100% | 0.97-1.00 | profiling 覆盖的 shape |
| Level 2: 七因子公式 | ~55-65% | 0.82-0.88 | 未见 shape / 冷启动 |
| 综合 (L1+L2) | — | **0.90-0.95** | 典型场景 |

---

## 完整使用流程

### 离线构建

```bash
python -m kernel.autotuner.offline_builder \
    --op matmul \
    --backend tilelang \
    --M-range 1 4096 \
    --N-values 768 2304 4096 \
    --K-values 768 2304 4096

# 输出:
#   ~/.cache/icpp-tuning/matmul_tilelang_perf_db.json  (决策表 + config pool)
```

### 在线推理

```python
tuner = AdaptiveTuner("matmul", backend="tilelang")
# 自动加载 perf_db

# Case 1: profiling 覆盖过的 shape → 查表命中
shape = OpShape(op_type="matmul", M=64, N=2304, K=768, fusion="bias_relu")
result = tuner.select(shape)
# result.level = "table", result.config = oracle 最优

# Case 2: 未见过的 shape → 公式选优
shape = OpShape(op_type="matmul", M=1337, N=3072, K=1024)
result = tuner.select(shape)
# result.level = "formula"
# Score = score_formula(op, cfg, hw) → 七因子乘积

# Case 3: 冷启动 → 纯公式 (无离线 DB)
tuner_cold = AdaptiveTuner("matmul")
result = tuner_cold.select(shape)
# result.level = "formula"
```

---

## 模块清单与实施路线

### Phase 1: 基础设施 + Cost Model
1. `kernel/autotuner/gpu_spec.py` — GPU 规格检测
2. `kernel/autotuner/cost_model_v2.py` — Trait-Driven 七因子公式 + score_formula()
3. `kernel/autotuner/search_space.py` — 配置空间 + 三层剪枝

### Phase 2: 离线 Profiling 引擎
4. `kernel/autotuner/offline_profiler.py` — 分层采样 + 并行 profiling
5. `kernel/autotuner/performance_db.py` — 决策表 + 知识库

### Phase 3: 在线引擎 + 集成
6. `kernel/autotuner/offline_builder.py` — 完整离线 pipeline
7. `kernel/autotuner/tuner.py` — 两级递进在线选优
8. 集成到现有 matmul_bias_act.py 等模板

### Phase 4: 算子模板补充
9. `kernel/tilelang/conv2d.py` — Conv2d 动态形状
10. `kernel/tilelang/conv2d_fused.py` — Conv2d+ReLU, Conv2d+BN
11. `kernel/tilelang/matmul_bias_layernorm.py` — Matmul+Bias+LN

### Phase 5: 验证
12. 纯公式精度 baseline (~85%)
13. 决策表命中精度 (~99%)
14. 在线开销: 查表<1μs, 公式选优<2μs
15. 离线构建时间 <20min
16. 跨设备: 换 GPU 后公式自动适配 (GPUSpec 参数化)

---

## 创新点总结

| # | 创新点 | 说明 |
|:--|:--|:--|
| 1 | **Trait-Driven 算子描述体系** | 以 FusionPrimitive 执行特征替代算子名称匹配，同一评分框架统一处理 GEMM/Conv 全系列融合算子 |
| 2 | **乘积式七因子物理公式** | 木桶效应，无超参数拟合，完全基于 GPU 物理规律（Roofline、occupancy、wave、pipeline、空间重叠、融合负载、归约开销） |
| 3 | **动态线程负载目标** | thread_eff 根据 wave 填充和 occupancy 自适应调整，而非固定阈值 |
| 4 | **两级递进选优** | 查表(~99%) → 公式(~85%)，覆盖所有场景，无额外依赖 |
| 5 | **"先剪枝后采样"搜索空间初始化** | 先做 exhaustive + 硬件剪枝得到 valid 集合，再在 valid 集合内做分层采样，保证边缘均匀性不被非均匀过滤破坏；三模式递进 (冷/结构/热启动) |
| 6 | **统一 6 种融合算子** | 同一框架处理 matmul/conv2d 全系列 |
| 7 | **后端感知搜索空间** | Triton (pow2) / TileLang (mul16) 自动适配，统一 TuneConfig 数据结构 |
| 8 | **离线分层采样** | 按 wave count 分层，最小采样量获得最大覆盖 |
| 9 | **跨设备零成本迁移** | 公式通过 GPUSpec 参数化，换 GPU 后自动适配，无需重训任何模型 |

### 一句话概括

> **"以 Trait-Driven 七因子乘积解析公式建模 GPU 物理规律（Roofline、occupancy、wave 填充、pipeline、Conv 空间重叠、融合尾部负载、归约开销），通过'先剪枝后分层采样'实现无偏搜索空间初始化，通过'查表→公式'两级递进策略实现动态形状下 matmul/conv2d 全系列融合算子的自适应高性能调优。"**
