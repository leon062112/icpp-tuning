1. 实验目标
尝试理清下列问题
 Q1：Dynamic Shape 影响autotuning的稳定性
  不同 shape 下最优 autotune config 往往不一致
Q2：Autotuning 参数为何显著影响性能
  BLOCK size / warps / stages 如何作用于底层执行
Q3：Dynamic Shape 是否改变性能瓶颈？
  同一 config 在不同 shape 下是否呈现不同 bottleneck
揭示“tuning参数 → 硬件执行→性能”的因果关系，并说明该关系在动态shape下是不稳定的。
2. 实验设置
2.1 Kernel 选择
- GEMM（compute + memory tradeoff）
- GEMM + Bias + ReLU (optional)
2.2 Autotune 参数空间
BLOCK_M ∈ {64, 128, 256}
BLOCK_N ∈ {64, 128, 256}
BLOCK_K ∈ {32, 64}
num_warps ∈ {2, 4, 8}
num_stages ∈ {1, 2, 3, 4}
2.3 Shape
固定 N=2304, K=768（对应LLM hidden→intermediate投影）, 
只变动态维度 M：M = 24/43/62/81/100/128/192/256
2.4 NCU Profiling
暂时无法在飞书文档外展示此内容
3. 实验设计
实验 1：Static Tuning 的性能损失量化
最优config是否shape-dependent？Static tuning 性能损失
方法：
1. 对 5 个 shape × ~20 个 config，全量测 latency（共 ~100 次 benchmark）
2. 对每个 shape 找到 per-shape best config
3. 选一个 "global best"（在最大 shape S5 上的最优 config）作为 static baseline
4. 计算 static baseline 在每个 shape 上相对 per-shape best 的性能损失
输出：
Figure 1: Shape × Config Latency Heatmap
- X 轴：config 编号（按 S5 上的 latency 排序）
- Y 轴：5 个 shape
- 颜色：latency（归一化到每个 shape 的最优值）
- 标注每个 shape 的 best config
Figure 2: Static Tuning Performance Loss
- X 轴：5 个 shape
- 柱状图：per-shape best latency vs static baseline latency
- 标注性能损失百分比
预期结论：
- 不同 shape 的最优 config 不同（heatmap 中最优位置不在同一列）
- Static tuning 在小 shape 上损失可达 X%（量化数字）
实验 2：Tuning 参数 → 硬件行为的因果分析
BLOCK_SIZE / num_warps / num_stages 各自如何影响硬件底层执行？
方法：
固定 1 个 shape（S3, M=256，中等规模，兼具并行度和访存压力），用控制变量法：
2a. Tile Size 影响（BLOCK_M ∈ {32, 64, 128}，其余固定）
- NCU 采集：Occupancy, L2 Hit Rate, Latency
- 预期：大 tile → L2 hit rate 上升（tile 内数据复用增加），Occupancy 下降（每 SM 能容纳的 block 数减少）
2b. num_warps 影响（warps ∈ {2, 4, 8}，其余固定）
- NCU 采集：Occupancy, Tensor Pipe Util, Stall Not Selected, Latency
- 预期：更多 warps → Occupancy 上升，但 Stall Not Selected 也上升（warp 间调度竞争）
2c. num_stages 影响（stages ∈ {2, 3, 4, 5}，其余固定）
- NCU 采集：Stall Memory Dependency, DRAM Throughput, Latency
- 预期：更多 stages → Stall Memory Dependency 下降（prefetch 隐藏访存延迟），但过多 stages → 寄存器压力导致 Occupancy 反降
共 10 个 config × 1 个 shape = 10 次 NCU profiling。
输出：
Figure 3: Parameter → Hardware Behavior (Controlled Experiments)
- 子图 (a)：X=BLOCK_M, Y 双轴=L2 Hit Rate / Occupancy
- 子图 (b)：X=num_warps, Y 双轴=Occupancy / Stall Not Selected
- 子图 (c)：X=num_stages, Y 双轴=Stall Memory Dep / DRAM Throughput
每个子图附带 latency 柱状图（或用次坐标轴）
预期结论：
- Tuning 参数通过 并行度、数据复用、pipeline 深度 三条路径影响性能
- 每条路径存在 trade-off（如大 tile 提升复用但降低并行度），不存在"万能最优"
实验 3：Dynamic Shape 改变硬件瓶颈（核心）
回答：为什么同一 config 在不同 shape 下性能差异大？瓶颈如何迁移？
方法： 
从实验 1 中选 2 个代表性 config：
- Config-A：大 tile（BLOCK_M=128, BLOCK_N=256, warps=8），适合大 shape
- Config-B：小 tile（BLOCK_M=32, BLOCK_N=64, warps=2），适合小 shape
对这 2 个 config × 5 个 shape = 10 个组合做 NCU profiling。
输出：
Figure 4: Hardware Bottleneck Migration across Shapes
- X 轴：5 个 shape（S1→S5）
- 4 条折线（Occupancy, L2 Hit Rate, DRAM Throughput, Tensor Util）
- Config-A 实线 / Config-B 虚线
预期结论：
- S1 (M=16)：Occupancy 极低（tile 数不够填满 SM），性能受限于 parallelism
- S5 (M=4096)：DRAM throughput 接近峰值，性能受限于 memory bandwidth
- Config-A 在小 shape 下 occupancy 更差（大 tile → 更少 block），在大 shape 下 L2 hit rate 更高（大 tile → 数据复用好）
- 实验 2 揭示了参数的静态 trade-off，本实验证明 shape 变化使 trade-off 的最优解发生偏移

4. 最终结论
The optimal tuning configuration is shape-dependent（实验 1, Figure 1-2） — Static autotuning incurs up to X% performance loss under dynamic shapes.
1. Tuning parameters affect performance through distinct hardware mechanisms（实验 2, Figure 3） — Tile size governs data reuse vs. parallelism; warps control occupancy vs. scheduling contention; pipeline stages trade memory latency hiding for register pressure. Each mechanism has inherent trade-offs with no universally optimal setting.
2. Dynamic shapes shift the hardware bottleneck, invalidating static trade-offs（实验 3, Figure 4） — The same configuration transitions from parallelism-bound (small M) to memory-bound (large M), causing the optimal balance point of Exp.2's trade-offs to move with shape.
3. Implication（引出方法） — This motivates a unified framework that co-optimizes fusion strategy, kernel template, and tuning parameters under dynamic shapes.
