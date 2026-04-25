# GEMM Cost Model v2 Runtime Evaluation

- Hardware: NVIDIA A100-SXM4-40GB, CUDA device 6
- Shapes: `N=2304`, `K=768`, `M=1..128 step 5` (26 shapes per op)
- Selector: formula top-1 from `kernel/autotuner/cost_model_v2.py`
- Baseline: each script's PyTorch/torch baseline benchmark
- Raw logs: `evaluation/gemm_formula_full.log`, `evaluation/gemm_bias_act_formula_full.log`, `evaluation/gemm_bias_layernorm_formula_full.log`
- Parsed CSV: `evaluation/gemm_three_ops_formula_summary.csv`

## Summary

| op | shapes | min speedup | median | mean | max | slower than torch | near baseline (<1.1x) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `gemm` | 26 | 1.9590 | 3.2853 | 3.2229 | 5.2733 | 0 | 0 |
| `gemm_bias_act` | 26 | 1.3966 | 1.5526 | 1.5709 | 1.8008 | 0 | 0 |
| `gemm_bias_layernorm` | 26 | 1.0065 | 1.4162 | 1.4959 | 2.1189 | 0 | 2 |

## Worst Shapes

### gemm

| M | config | formula ms | torch ms | speedup |
|---:|---|---:|---:|---:|
| 106 | `BM16_BN128_BK64_SK2_S4_T128` | 0.0235 | 0.0461 | 1.9590 |
| 126 | `BM16_BN128_BK64_SK2_S4_T128` | 0.0242 | 0.0480 | 1.9781 |
| 121 | `BM16_BN128_BK64_SK2_S4_T128` | 0.0247 | 0.0503 | 2.0411 |
| 101 | `BM16_BN128_BK64_SK2_S4_T128` | 0.0227 | 0.0479 | 2.1116 |
| 1 | `BM16_BN128_BK64_SK2_S4_T256` | 0.0141 | 0.0315 | 2.2339 |

### gemm_bias_act

| M | config | formula ms | torch ms | speedup |
|---:|---|---:|---:|---:|
| 121 | `BM16_BN256_BK64_S2_T256` | 0.0222 | 0.0311 | 1.3966 |
| 81 | `BM16_BN256_BK64_S2_T256` | 0.0218 | 0.0307 | 1.4081 |
| 71 | `BM16_BN256_BK64_S2_T256` | 0.0216 | 0.0306 | 1.4178 |
| 116 | `BM16_BN256_BK64_S2_T256` | 0.0222 | 0.0315 | 1.4191 |
| 91 | `BM16_BN256_BK64_S2_T256` | 0.0212 | 0.0311 | 1.4656 |

### gemm_bias_layernorm

| M | config | formula ms | torch ms | speedup |
|---:|---|---:|---:|---:|
| 1 | `BM16_BN128_BK64_S2_T256` | 0.0487 | 0.0490 | 1.0065 |
| 51 | `BM32_BN128_BK64_S2_T256` | 0.0477 | 0.0520 | 1.0897 |
| 71 | `BM16_BN128_BK64_S2_T128` | 0.0488 | 0.0565 | 1.1573 |
| 36 | `BM16_BN128_BK64_S2_T128` | 0.0460 | 0.0540 | 1.1750 |
| 6 | `BM16_BN128_BK64_S2_T256` | 0.0481 | 0.0586 | 1.2171 |

## Findings

- No tested shape is slower than the torch baseline for any of the three operators.
- Plain GEMM has the strongest margin: minimum speedup is still about 1.96x, and median speedup is about 3.29x.
- GEMM+Bias+ReLU is consistently faster but has a narrower margin, mostly because the torch baseline latency is already around 0.03 ms while the fused TileLang kernel stays around 0.02 ms.
- GEMM+Bias+LayerNorm is the most fragile case: `M=1` is effectively tied with torch at 1.0065x, and `M=51` is only 1.0897x. The fused path launches separate GEMM and LayerNorm kernels, so for tiny/irregular M the extra launch and reduction overhead consumes most of the fusion benefit.
- The worst GEMM points are at larger M such as 106/126 because the selected split-K config adds atomic accumulation overhead while torch matmul remains highly optimized; nevertheless, they remain nearly 2x faster in this sweep.
