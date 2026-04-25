# Unified Kernel Autotuner Interface

## Goal
每个 kernel 添加 ~10 行 descriptor，调用 `select_best(DESCRIPTOR, M=64, N=2304, K=768)` 即可获取最优配置。

## New File: `kernel/autotuner/interface.py`

**TileLangKernelBase** — 基类，kernel 只需覆写 3 个方法:
- `name` — kernel 名称
- `make_op_spec(**shape_kwargs) -> OpSpec` — 构建算子描述
- `get_raw_configs(**shape_kwargs) -> List[dict]` — 搜索空间

默认实现: `raw_config_to_tune_config`(thread_num//32→num_warps), `score_adjustment`(返回原分), `format_config`

**select_best(descriptor, hw, **shape_kwargs)** → (best_raw_config, score)
**rank_all(descriptor, hw, detailed, **shape_kwargs)** → 排序列表
**configure_autotuner_cache(cache_dir)** — 共用缓存配置

## Kernel Descriptors (每个 kernel 底部添加)

| Kernel | make_op_spec | 特殊处理 |
|---|---|---|
| gemm.py | `make_matmul_spec(M, N, K//split_k)` | `score_adjustment`: split_k bonus/penalty |
| gemm_bias_act.py | `make_matmul_spec(M, N, K, ["bias_add", "relu"])` | 无 |
| gemm_bias_layernorm.py | `make_matmul_spec(M, N, K, ["bias_add", "row_layernorm"])` | 无 |
| conv2d.py | `make_conv2d_spec(...)` | 无 |
| conv2d_relu.py | `make_conv2d_spec(..., ["relu"])` | 无 |
| conv2d_bn.py | `make_conv2d_spec(..., ["batchnorm"])` | 无 |
| conv2d_bn_relu.py | `make_conv2d_spec(..., ["batchnorm", "relu"])` | 无 |
| conv_bn_add_relu.py | `make_conv2d_spec(..., ["batchnorm", "relu"])` | 无 |

## 使用示例

```python
from kernel.tilelang.gemm.gemm_bias_act import DESCRIPTOR
from kernel.autotuner.interface import select_best

best_config, score = select_best(DESCRIPTOR, M=64, N=2304, K=768)
# best_config = {"block_M": 32, "block_N": 128, "block_K": 32, "num_stages": 2, "thread_num": 128}
```

## Eval Scripts 简化
3 个 eval 脚本 (~960行) → 调用 rank_all() 的薄封装，删除重复的 config class/_make_hw_spec/configure_autotuner_cache

## Implementation Order
1. 创建 `kernel/autotuner/interface.py`
2. GEMM 系列 kernel 添加 DESCRIPTOR (gemm.py, gemm_bias_act.py, gemm_bias_layernorm.py)
3. Conv 系列 kernel 添加 DESCRIPTOR (5 个 conv kernel)
4. 简化 3 个 eval 脚本
5. 更新 `kernel/autotuner/__init__.py` 导出

## Verification
- `select_best(DESCRIPTOR, M=64, N=2304, K=768)` 与原 rank_configs()[0] 结果一致
- gemm split_k 的 score_adjustment 行为与原 score_tilelang_config() 一致
- 简化后的 eval 脚本 dry-run 输出与原版一致
