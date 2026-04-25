# A Trait-Driven Analytical Cost Model for Tile Configuration Ranking

## 摘要

本文总结当前实现 [kernel/autotuner/cost_model_v2.py](/denghaodong/code/icpp-tuning/kernel/autotuner/cost_model_v2.py) 的整体设计思路。该模型面向 GPU 张量程序的 tile configuration 排序问题，采用一种 trait-driven 的解析式框架，以避免为不同融合算子编写大量 operator-specific 的启发式规则。模型以 `OpSpec`、`TuneConfig` 和 `GPUSpec` 为输入，先将算子、候选配置与硬件特征统一投影到一组中间执行特征，再通过主循环、内存、并行、流水线、epilogue 与 reduction 六个效率项的乘积进行综合评分。其目标不是直接精确预测 kernel latency，而是在低 profiling 开销下，为 autotuner 提供可解释、跨算子复用、且具有一定硬件感知能力的候选排序。

## 1. 引言

GPU kernel autotuning 的核心问题之一，是如何在一个通常较大的离散搜索空间中，快速找到一组高质量候选配置。完全依赖 runtime profiling 的方法虽然准确，但代价高，尤其是在动态 shape、融合算子和多硬件后端并存时，搜索成本会迅速放大。另一方面，若仅依赖针对单一算子的手工经验规则，则难以推广到新的融合模式和新的 operator family。

当前 `cost_model_v2` 的设计出发点，是在两者之间取得折中。它不使用历史样本训练出的黑盒回归器，也不直接为诸如 `gemm+bias+relu` 或 `gemm+bias+layernorm` 这样的具体算子组合写专用公式，而是尝试把算子的执行结构分解为通用语义单元，并在这一抽象层面上构建一个硬件感知的解析模型。换言之，该模型希望回答的问题不是“这个具体算子在某张卡上跑多少毫秒”，而是“在一组候选 tile 中，哪类 tile 更符合当前工作负载与硬件的执行特征”。

## 2. 问题定义

给定一个算子描述 `OpSpec`、一个候选配置 `TuneConfig` 以及一个 GPU 硬件描述 `GPUSpec`，模型输出一个标量分数：

```text
score = f(OpSpec, TuneConfig, GPUSpec)
```

其中分数越高，表示该配置越可能具有更好的运行性能。需要强调的是，该分数不是 latency 的直接估计值，而是一个相对排序信号。它服务于两类场景：

1. 在全量 autotune 之前进行预排序，缩小 profiling 范围。
2. 对候选配置的优劣给出可解释的结构化分析。

因此，模型首先追求的是排序质量和解释能力，其次才是与真实 latency 的数值逼近。

## 3. 方法总览

整个模型可以概括为两个阶段：

1. **执行特征投影阶段**
   将 `(OpSpec, TuneConfig, GPUSpec)` 投影到一组统一的中间执行特征 `ConfigProjection`。
2. **效率乘积评分阶段**
   基于中间特征计算六个归一化效率项，并将其相乘得到最终分数。

形式化地，当前实现采用如下分解：

```text
score =
    eff_mainloop
  * eff_memory
  * eff_parallel
  * eff_pipeline
  * eff_epilogue
  * eff_reduction
```

这样的设计有两个直接好处。其一，不同算子族可以复用相同的投影特征和评分框架。其二，评分结果可被拆解为多个具备物理含义的因子，便于定位排序偏差来自哪里。

## 4. 统一抽象

### 4.1 主循环抽象：`MainloopSpec`

`MainloopSpec` 描述算子的主体计算结构。当前实现支持至少两类主循环：

- `gemm`
- `implicit_gemm_conv`

它包含的问题规模与数据规模信息主要有：

- `M, N, K`
- 输入、权重与输出元素数
- 输入、权重、输出与累加类型的字节数
- `flops_per_fma`
- 输入复用、权重复用与空间复用提示

从功能上看，`MainloopSpec` 提供了两类信息：一类是理论计算量，另一类是最小访存量。这些信息为后续算访比、padding 和复用估计提供基础。

### 4.2 融合 primitive 抽象：`FusionPrimitive`

融合算子的 epilogue 部分在当前模型中被拆成若干原子 primitive，每个 primitive 描述自己的执行属性，而不依赖上层算子名称。关键属性包括：

- `flops_per_output`
- `bytes_per_output`
- `reduction_axis`
- `extra_passes`
- `sync_penalty`
- `preferred_outputs_per_thread`
- `preferred_tile_m`
- `preferred_tile_n`

例如：

- `BIAS_ADD` 与 `RELU` 表示轻量 pointwise primitive；
- `SILU` 与 `GELU` 表示更重的逐元素非线性；
- `ROW_LAYERNORM` 与 `BATCHNORM` 表示带 reduction 的 epilogue。

这种表示方式使模型可以在不硬编码具体融合算子名字的情况下，对 epilogue 的额外开销进行建模。

### 4.3 配置抽象：`TuneConfig`

`TuneConfig` 负责描述一个候选 tile 配置，主要字段包括：

- `block_M`
- `block_N`
- `block_K`
- `num_stages`
- `num_warps`
- `group_size_m`

此外，当前实现还从这些字段中推导两个重要资源估计：

- `shared_mem_bytes`
- `regs_per_thread_estimate`

这使得后续模型能够从配置本身推断其对 occupancy、并发 block 数和共享内存压力的影响。

### 4.4 硬件抽象：`GPUSpec`

`GPUSpec` 提供硬件容量与峰值能力，包括：

- `num_sm`
- `max_warps_per_sm`
- `max_regs_per_sm`
- `shared_mem_per_sm`
- `mem_bandwidth_gbps`
- `peak_fp16_tflops`

在此基础上，模型进一步使用 `ridge_point_fp16` 等派生量，将 roofline 相关信息显式纳入评分。

## 5. 执行特征投影

`project_config()` 是当前模型的核心中间层。它不直接评分，而是将输入映射为一组后续评分项复用的中间执行特征。主要特征包括：

- `tiles_M, tiles_N, tiles_K`
- `grid_size`
- `tile_outputs`
- `outputs_per_thread`
- `mainloop_iterations`
- `tile_arithmetic_intensity`
- `occupancy`
- `resident_blocks_per_sm`
- `wave_fill`
- `preferred_num_stages`
- `shared_mem_bytes_per_tile`

这些特征可分为四类：

1. **几何特征**
   例如 `tiles_M`、`tiles_N`、`grid_size`，反映问题如何被 tile 切分。
2. **资源特征**
   例如 `shared_mem_bytes_per_tile`、`resident_blocks_per_sm`、`occupancy`，反映配置对硬件容量的占用情况。
3. **算访特征**
   例如 `tile_arithmetic_intensity`，反映该 tile 更偏算力受限还是带宽受限。
4. **调度特征**
   例如 `wave_fill`、`preferred_num_stages`，反映 block 波次利用率与流水线阶段偏好。

这一投影层的意义在于，把算子差异、配置差异与硬件差异统一折叠为“执行特征差异”，从而让评分公式本身保持相对稳定。

## 6. 评分函数

### 6.1 主循环效率 `eff_mainloop`

`eff_mainloop` 用于度量当前 tile 是否适合主循环执行，主要包含三项：

- `padding_eff`
- `thread_eff`
- `k_eff`

其中 `padding_eff` 度量 tile 覆盖范围中真正有效计算所占比例；`k_eff` 则反映 `block_K` 是否接近模型偏好的 K 分块。

当前版本中最关键的是 `thread_eff` 的设计。早期直觉通常会用固定目标值来约束 `outputs_per_thread`，但这一做法在小 `grid_size` 或低 occupancy 的场景下容易高估大 tile。当前实现将目标输出粒度设为一个**动态值**，其大小由：

- `wave_fill`
- `resident_blocks_per_sm`
- `occupancy`

共同决定。于是当 launch 已经能够较好填满 GPU 时，模型允许更大的 `outputs_per_thread`；而当并行度本身不足时，模型会惩罚进一步增大 tile 导致的 block 数下降。该设计试图把“线程粒度偏好”从静态常数改写为“受并行饱和度调制的动态偏好”。

### 6.2 内存效率 `eff_memory`

`eff_memory` 从两个角度刻画内存相关代价：

1. **roofline 效率**
   使用 `tile_arithmetic_intensity / ridge_point_fp16` 近似判断当前 tile 是否具备足够高的算访比。
2. **共享内存压力**
   用 tile 的共享内存占用相对于单个 SM 共享内存预算的比例来进行惩罚。

因此，`eff_memory` 本质上是一个解析近似：它不模拟真实的 cache 行为，也不重建精确带宽模型，而是通过算访比和共享内存压力给出一阶判断。

### 6.3 并行效率 `eff_parallel`

`eff_parallel` 评价 GPU 是否被充分喂饱。它由三部分组成：

- `occupancy_eff`
- `wave_eff`
- `warp_eff`

其中 `occupancy_eff` 关注单 block 的资源占用是否抑制了并发；`wave_eff` 关注 grid 是否产生严重尾波浪费；`warp_eff` 则对过高的 warp 数给出额外惩罚。该项体现的是“从 SM 调度视角观察该配置是否合理”。

### 6.4 流水线效率 `eff_pipeline`

`eff_pipeline` 用于刻画 pipeline stage 设置是否合适。当前实现由两部分构成：

- `fill_eff`
  用于度量在给定主循环迭代数下，过多 stage 带来的填充与排空损失。
- `stage_match`
  用于度量实际 `num_stages` 与 `preferred_num_stages` 的偏差。

其中 `preferred_num_stages` 是一个由算访比、主循环类型和 reduction 情况推导出来的解析偏好值。

### 6.5 Epilogue 效率 `eff_epilogue`

`eff_epilogue` 用于度量融合 epilogue 的线程粒度和 tile 形状是否合理。每个 primitive 会根据自身偏好对当前配置给出一个局部效率值，多个 primitive 再以几何平均的方式合并。

这种几何平均设计意味着：只要某个 primitive 与当前 tile 明显不匹配，整体 epilogue 效率就会受到实质性拖累，从而避免单个 primitive 的高分掩盖另一个 primitive 的显著不匹配。

### 6.6 Reduction 效率 `eff_reduction`

`eff_reduction` 专门用于处理带 reduction 的 epilogue primitive。它显式考虑：

- reduction 轴长度是否接近偏好
- `extra_passes`
- `sync_penalty`

这使得 layernorm、batchnorm 一类融合 epilogue 的额外代价能够被独立表达，而不被简单地并入 pointwise epilogue 中。

## 7. 模型的结构性特点

### 7.1 Trait-driven，而非 operator-specific

当前模型最大的结构特征，在于它不直接依赖算子名字，而是依赖：

- 主循环类型
- primitive 的算术/访存/同步 trait
- 配置的资源占用 trait
- 硬件资源 trait

因此，它试图学习的不是“某个算子模板的经验公式”，而是“哪些执行 trait 会使某类 tile 更适合某类硬件和工作负载”。这也是其跨算子复用能力的来源。

### 7.2 可解释性

由于最终分数可以拆成 `eff_mainloop` 到 `eff_reduction` 六项，因此模型不仅能排序，还能解释。例如，当某一配置排序靠前时，可以进一步判断其优势究竟来自：

- 更好的 tile 覆盖率；
- 更高的算访比；
- 更高的 occupancy；
- 更匹配的 epilogue 粒度；
- 或更低的 reduction 同步代价。

这也是为何当前仓库配套了 `score_formula_detailed()` 和详细 CSV 导出能力。

### 7.3 面向排序而非面向绝对预测

该模型的直接输出是相对评分，不是绝对 latency。换言之，它的目标是“把更好的配置排到更前面”，而不是“精确估计每个配置的毫秒数”。这一定位决定了它更适合作为 pre-ranker 或 reranker，而非完整替代 runtime profiling 的 oracle 模型。

## 8. 经验观察：以 `gemm_bias_act` 为例

在当前仓库的测试中，`cost_model_v2` 已经能够在较低开销下为 `gemm_bias_act` 提供有意义的预排序。经验上，它可以稳定避开一批明显较差的配置，并把搜索范围收缩到更小的候选集合。

近期对 `eff_mainloop` 中 `outputs_per_thread` 的动态化处理进一步改善了部分 shape 上的实际性能。这说明，在当前模型中，主循环粒度与并行饱和度之间的耦合，确实是影响排序质量的关键因素之一。

不过也应看到，模型仍不能稳定命中 oracle top-1，尤其在某些特定形状下，排序偏差依然存在。这表明当前解析项虽然已经抓住了一部分主要因素，但仍未完整覆盖 backend lowering、指令粒度、访存合并以及 epilogue 真实开销等细节。

## 9. 局限性

当前模型存在以下限制：

1. **不是 latency predictor**
   分数只适合排序，不应被解释为真实执行时间。
2. **资源估计仍是近似**
   寄存器与共享内存占用没有直接来自编译后端的精确信息。
3. **对 backend 细节建模不足**
   当前尚未显式表示指令形状、vectorization、访存合并、cache 行为和 tensor core fragment 约束。
4. **因子之间存在耦合**
   `eff_mainloop`、`eff_parallel` 与 `eff_epilogue` 在某些场景下会共同影响同一类配置偏好，可能导致系统性偏差。
5. **对融合 epilogue 的表达仍较粗**
   简单的 primitive trait 尚不足以完整刻画所有复杂融合模式。

因此，当前模型更适合作为“低成本、可解释、可迁移的候选排序器”，而非最终的真实性能 oracle。

## 10. 结论

当前 [kernel/autotuner/cost_model_v2.py](/denghaodong/code/icpp-tuning/kernel/autotuner/cost_model_v2.py) 可以被理解为一种面向 GPU tile 配置选择的 trait-driven 解析型成本模型。它通过将算子、配置与硬件统一抽象为执行特征，并用六个具有物理含义的效率项进行组合，构建了一套在不同 operator family 间可复用、在工程上可解释、在调优流程中低成本可落地的排序框架。

其核心贡献不在于给出一个精确 latency 数值，而在于提供一套结构化的、可扩展的性能归因方法：对于一个候选配置为何更好，模型能够从主循环、内存、并行、流水线、epilogue 和 reduction 六个维度给出明确解释。这使得它既能服务于 autotuning，也能服务于后续的模型修正与性能分析。
