# DREAMPlace 风格 GPU 解析式 Placer 计划书

> 创建日期：2026-04-19
> 维护人：jiangxiaoyi2004@gmail.com
> 工作目录：`team_trash_Workspace/dreamplace/`

---

## 1. 背景与立项动机

### 1.1 比赛与团队分工

- 参赛比赛：**Partcl / Hudson River Trading Macro Placement Challenge 2026**，提交截止 2026-05-21。
- 团队内 SA GPU 路线由其他成员维护（`team_trash_Workspace/sa_gpu/`），当前 17 IBM 平均 proxy ≈ 1.5347，比 RePlAce baseline（1.4578）差 5.3%。
- 本人选择**第二条独立路线**：实现一个 DREAMPlace 风格的 GPU 解析式 placer，作为团队的另一个候选算法 / SA 的 warm start。

### 1.2 为什么选 DREAMPlace 路线

- **算法范式互补**：SA 是离散随机搜索；解析法是连续梯度优化，信息利用率高、收敛快。
- **硬件契合**：评测机为 RTX 6000 Ada 48 GB，PyTorch 上写解析法可直接享 GPU 加速。
- **学术 SOTA**：DREAMPlace = GPU 上的 RePlAce，是当前 placement 学术界主流框架。
- **与 SA 解耦**：完全不依赖 SA 团队代码（SA 的 `team_trash_Workspace/sa_gpu/` 与 `team_trash_Workspace/RePlAce/` 都不能依赖），失败也不会影响队友。

### 1.3 边界

| 项 | 范围 |
|---|---|
| Tier | 仅 Tier 1（17 IBM proxy） |
| 可依赖 | `macro_place/` SDK、`external/MacroPlacement/`、PyPI 第三方包 |
| 不可依赖 | `team_trash_Workspace/sa_gpu/`、`team_trash_Workspace/RePlAce/` |
| 运行模式 | HPC + sbatch，每次提交前需向用户确认 sbatch 脚本 |
| 单 benchmark 时间预算 v1 | < 10 分钟 |

---

## 2. 数学建模

### 2.1 决策变量

设有 N 个 macro（hard + soft），每个有 (x, y) 中心坐标。整体决策变量：

```
x ∈ R^(2N)
```

- Fixed macros 的位置作为常量，通过 grad mask 保证不更新
- Soft macros 参与 global placement（让 density 场更准），但**不做 legalization**（允许重叠）
- Hard macros 参与 global placement + legalization + detail placement

### 2.2 目标函数

```
L(x) = WL_WA(x) + λ · D_eDensity(x)
```

v1 **不包含 congestion 项**，留作 v2。

### 2.3 Wirelength：Weighted Average (WA)

对每个 net 的 pin 集合 {x_i}，用 softmax 加权代替不可导的 max/min：

```
~max(x; γ) = Σ x_i · exp(x_i/γ) / Σ exp(x_i/γ)
~min(x; γ) = Σ x_i · exp(-x_i/γ) / Σ exp(-x_i/γ)
```

总 WL：

```
WL(x) = Σ_e w_e · [~max_e(x) - ~min_e(x) + ~max_e(y) - ~min_e(y)]
```

数值稳定化：用 log-sum-exp trick，先减 max 再 exp。

**v1 简化**：所有 pin 都视为 macro 中心（不抽 pin offset）。

### 2.4 Density：eDensity（电势能）

把每个 macro 当电荷，求解 Poisson 方程得电势场。

**Step 1**：把 macro 涂到 G×G 网格得密度场 ρ（用 bell-shape 软相交，对 macro 中心可导）

**Step 2**：解 Poisson 方程 `-∇²ψ = ρ - ρ̄`，用 2D DCT 在频域：

```
ρ̂ = DCT2(ρ - ρ̄)
ψ̂[u,v] = ρ̂[u,v] / (k_u² + k_v²)
ψ = IDCT2(ψ̂)
```

**Step 3**：电势能

```
D(x) = ½ · Σ ρ_{r,c} · ψ_{r,c} · A_grid
```

**Grid 自适应**：

```
G = clamp(ceil(canvas / min_macro_size) * 2, 64, 512)
```

### 2.5 调度

| 量 | 起始 | 终止 | 调度策略 |
|---|---|---|---|
| γ (WA 平滑) | grid_w × 8 | grid_w × 0.5 | 几何衰减，每 50 步 ×0.95 |
| λ (density 权重) | 让 WL 与 D 初始 1:1 | 上限 100× 初值 | 几何增长，每 50 步 ×1.05 |
| lr (Adam) | canvas_w × 0.01 | 不变 | 固定 |

---

## 3. 算法栈总览

| 阶段 | 方法 | 说明 |
|---|---|---|
| 初始解 | `benchmark.macro_positions` | 比赛自带 hand-crafted 起点，便于和 SA 对照 |
| Loss | WL_WA + λ·D_eDensity | 不含 congestion |
| Pin 模型 | macro 中心 | v1 简化，不抽 pin offset |
| 变量 | hard + soft 一起 | fixed 用 grad mask |
| 优化器 | Adam | PyTorch 原生，开箱即用 |
| 终止 | 固定 1500 步 + 9 分钟时间兜底 | 提前进入 legalize |
| Legalization | Greedy spiral search | hard only，soft 不动 |
| Detail | 30 行格点 hill-climb | ~1000 trials |
| 精度 | float32 | |

---

## 4. 工程实现

### 4.1 目录结构（计划）

```
team_trash_Workspace/dreamplace/
├── PLAN_zh.md                 # 本文件
├── README.md                  # 简短运行手册（M0 时建）
├── placer.py                  # DreamPlacer 入口（评测器加载点）
├── torch_loss.py              # WA wirelength
├── torch_density.py           # eDensity (painting + Poisson DCT)
├── global_place.py            # 训练循环 + λ/γ 调度
├── legalize.py                # Spiral search legalization
├── detail.py                  # Hill-climb detail placement
├── benchmark_context.py       # 从 Benchmark 构造 net/pin 张量
├── tests/
│   └── test_smoke.py          # ibm01 smoke test
└── results/                   # 运行产物（JSONL 日志、可视化等）
```

### 4.2 Placer 入口契约

```python
import torch
from macro_place.benchmark import Benchmark

class DreamPlacer:
    def __init__(self): ...   # 读 env var 配置
    def place(self, benchmark: Benchmark) -> torch.Tensor:  # [num_macros, 2]
        ...
```

### 4.3 环境变量

| 名称 | 默认值 | 说明 |
|---|---|---|
| DP_ITERS | 1500 | 全局优化最大迭代步 |
| DP_GRID | auto | density 网格大小（auto = 自适应） |
| DP_LR | auto | Adam learning rate（auto = canvas_w × 0.01） |
| DP_LAMBDA_INIT | auto | λ 初值（auto = 让 WL/D ≈ 1:1） |
| DP_LAMBDA_MULT | 1.05 | λ 几何增长率 |
| DP_LAMBDA_STEP | 50 | λ 每多少步增长 |
| DP_TIME_BUDGET | 540 | 全局阶段秒数硬上限（9 分钟） |
| DP_DEVICE | auto | cuda / cpu |
| DP_LOG_PATH | None | 若设置则写 JSONL |

### 4.4 日志格式（JSONL）

每行一个 record，对齐 SA 团队 `full_cuda_after_legalize.jsonl` 风格：

```json
{
  "name": "ibm01",
  "stage": "global",
  "step": 100,
  "wl": 0.082,
  "density": 0.91,
  "overflow": 0.12,
  "lambda": 0.5,
  "gamma": 0.34,
  "lr": 0.23,
  "elapsed": 12.4
}
```

最终一行：

```json
{
  "name": "ibm01",
  "stage": "final",
  "proxy_cost": 1.31,
  "wirelength": 0.09,
  "density": 0.92,
  "congestion": 1.5,
  "overlaps": 0,
  "valid": true,
  "runtime": 245.6
}
```

### 4.5 测试策略

仅一个 smoke test：

- 文件：`tests/test_smoke.py`
- 内容：跑 ibm01（最小 grid + 少 iters，~60s）
- 断言：
  - 输出 shape == `(num_macros, 2)`
  - hard macro `overlap_count == 0`
  - `proxy < 2.0`（仅验流程，不评分）

### 4.6 开发顺序

1. **ibm01** 单 benchmark 跑通
2. **ibm01 / ibm04 / ibm09** 三个对照（小、中、密三档）
3. **17 个全跑** sbatch 出最终报告

---

## 5. 里程碑

| # | 里程碑 | 预期交付 | 完成判据 |
|---|---|---|---|
| **M0** | 工程骨架 | `dreamplace/` 目录 + 占位 placer.py | `uv run evaluate dreamplace/placer.py -b ibm01` 不报错（即使输出 = 初始解） |
| **M1** | WA wirelength | `torch_loss.py` | 单元测试：随机 placement，梯度方向把强连接 macro 朝彼此推 |
| **M2** | eDensity | `torch_density.py` | 单元测试：单 macro 居中电势能为 0；两 macro 重叠时梯度互斥 |
| **M3** | 训练循环 | `global_place.py` | ibm01 跑完无报错，loss 单调下降趋势，输出可视化 |
| **M4** | Legalization | `legalize.py` | ibm01 legalize 后 `compute_overlap_metrics` 报 overlap_count==0 |
| **M5** | Detail placement | `detail.py` | hill-climb 后 proxy 严格 ≤ legalize 后 proxy |
| **M6** | Pipeline 串接 | `placer.py` | 完整 place(benchmark) 跑通，env var 可控 |
| **M7** | Smoke test | `tests/test_smoke.py` | `uv run pytest team_trash_Workspace/dreamplace/tests/ -v` 通过 |
| **M8** | ibm01 调通 | sbatch 跑 ibm01 | proxy < 1.5，零重叠，10 min 内 |
| **M9** | 3-bench 对照 | ibm01/04/09 三个结果 | 三个都 ≤ SA baseline 对应值 |
| **M10** | 17 全跑 + 报告 | JSONL + 对照表 + 决策 | 出报告，决定是否进入 v2 |

每个里程碑预计工作量 0.5–1.5 天，整体 v1 预计 2 周可见全 17 结果。

---

## 6. 已知风险与对策

| 风险 | 概率 | 影响 | 对策 |
|---|---|---|---|
| WA 的 exp 数值溢出 | 高 | nan loss | log-sum-exp 稳定化 |
| Painting 不可导导致梯度为 0 | 中 | 优化卡死 | 用 bell-shape 软相交，不用硬相交 |
| Poisson DCT 漏掉减均值 | 中 | ψ 无解 | 解前显式减 ρ̄ |
| λ 初值太大 → cells 还没拓扑就被推散 | 高 | WL 上不去 | 初值用"WL/D=1:1"自动定，几何缓增 |
| Fixed macro 没掩掉梯度 | 中 | 输出违规 | 训练前后都做 mask + restore |
| Legalize 损失大 | 高 | 比 RePlAce 差 | v2 升级 Tetris / network flow |
| 没有 pin offset 影响 WL 准确度 | 中 | 分数低估约 5% | v2 抽 pin offset |
| 没有 congestion 项 | 高 | congestion 大头未被优化 | v2 加 grid-based congestion 近似 |
| HPC sbatch 排队 / GPU 申请失败 | 低 | 进度延误 | 本地小 benchmark 先验，大 benchmark 排队等 |

---

## 7. v2 候选改进（M10 后视情况启动）

本节为计划阶段的初步候选。**v1 运行后的实际 TODO 列表见 §10**（按 17 benchmark 实测结果重排序）。

按优先级排序：

1. **Pin offset** —— 从 `PlacementCost` 抽 pin 偏移，WL 更准
2. **Congestion 可微项** —— grid-based 路由需求 + ReLU(超载) + top-k
3. **Legalization 升级** —— Tetris / Abacus 替代 spiral search
4. **Nesterov 优化器** —— 替代 Adam，匹配 DREAMPlace 论文
5. **Macro-aware density** —— 大 macro 单独建电势项
6. **Quadratic 初始解** —— 解 `Lx = b` 替代 `initial.plc` 起点
7. **联合后处理** —— 把结果送给 SA 团队当 warm start

---

## 8. 决策日志

| 日期 | 决策 | 理由 |
|---|---|---|
| 2026-04-19 | 选 DREAMPlace 路线，不做 SA / B*-tree / RL | SA 已有队友；DREAMPlace 是 SOTA + GPU 友好 |
| 2026-04-19 | v1 不含 congestion | 易把 WL 项搅乱，先把 WL+density 跑稳 |
| 2026-04-19 | v1 用 macro 中心代替 pin offset | 简化实现，预计影响 ~5% |
| 2026-04-19 | v1 用 Adam 而非 Nesterov | PyTorch 原生，省去手写 |
| 2026-04-19 | v1 用 spiral search legalization | 100 行可成，与 SA 团队思路一致便于对照 |
| 2026-04-19 | soft macros 一起进 global，不进 legalize | density 场更准；soft 允许重叠 |
| 2026-04-19 | 仅 1 个 smoke test | 算法验收以分数为准，单元功能测试边写边 print 验 |
| 2026-04-20 | v1 开 `torch.use_deterministic_algorithms(True)` | 后续调参需要可复现；会消除 GPU 内 atomic 抖动，但不消除 CPU/GPU 差异 |
| 2026-04-20 | v1 不做 float64 升级 | CPU/GPU 数值差约 0.6%，绝对值仍优于 RePlAce；代价是内存 2×、慢 30% |

---

## 9. v1 结果（2026-04-20）

sbatch job 33417034，A100-SXM4-40GB，deterministic=on，total runtime 3642s (60.7 min)。
- 完整 log：`/scratch/users/k2366837/logs/dreamplace-m10-33417034.out`
- JSONL：`team_trash_Workspace/dreamplace/results/m10_33417034.jsonl`

| Benchmark | DreamPlace | SA baseline | RePlAce baseline | vs SA | vs RePlAce | Overlaps |
|---|---|---|---|---|---|---|
| ibm01 | 0.9718 | 1.3166 | 0.9976 | +26.2% | +2.6% | 0 |
| ibm02 | 1.6820 | 1.9072 | 1.8370 | +11.8% | +8.4% | 0 |
| ibm03 | 1.1684 | 1.7401 | 1.3222 | +32.9% | +11.6% | 0 |
| ibm04 | 1.2009 | 1.5037 | 1.3024 | +20.1% | +7.8% | 0 |
| ibm06 | 1.3974 | 2.5057 | 1.6187 | +44.2% | +13.7% | 0 |
| ibm07 | 1.3264 | 2.0229 | 1.4633 | +34.4% | +9.4% | 0 |
| ibm08 | 1.3867 | 1.9239 | 1.4285 | +27.9% | +2.9% | 0 |
| ibm09 | 0.9454 | 1.3875 | 1.1194 | +31.9% | +15.5% | 0 |
| ibm10 | 1.2011 | 2.1108 | 1.5009 | +43.1% | +20.0% | 0 |
| ibm11 | 1.0134 | 1.7111 | 1.1774 | +40.8% | +13.9% | 0 |
| ibm12 | 1.6013 | 2.8261 | 1.7261 | +43.3% | +7.2% | 0 |
| ibm13 | 1.1517 | 1.9141 | 1.3355 | +39.8% | +13.8% | 0 |
| ibm14 | 1.4167 | 2.2750 | 1.5436 | +37.7% | +8.2% | 0 |
| ibm15 | 1.3991 | 2.3000 | 1.5159 | +39.2% | +7.7% | 0 |
| ibm16 | 1.3249 | 2.2337 | 1.4780 | +40.7% | +10.4% | 0 |
| ibm17 | 1.5555 | 3.6726 | 1.6446 | +57.6% | +5.4% | 0 |
| ibm18 | 1.7209 | 2.7755 | 1.7722 | +38.0% | +2.9% | 0 |
| **AVG** | **1.3214** | **2.1251** | **1.4578** | **+37.8%** | **+9.4%** | **0** |

关键事实：
- **17/17 VALID**（零重叠）
- **17/17 击败 RePlAce baseline**，最大 +20.0% (ibm10)，最小 +2.6% (ibm01)
- 相对 SA baseline 领先 37.8%
- 平均单 benchmark 3.5 min，最大单 benchmark ~8.8 min (ibm17，525s)
- WL 全部 < 0.09（稳），density 0.51–0.62（稳），**congestion 是大头瓶颈**（10/17 cong > 1.9）

---

## 10. v2 TODO（按 v1 实测重排）

1. **Congestion 可微项** — v1 最大短板；10/17 benchmark 的 cong > 1.9，ibm02/06/10/12/14/17 cong > 2
2. **Detail placement 加速** — 单次 `compute_proxy_cost` ~1.5s，detail 120s 只跑 ~80 trials；换成自研 torch surrogate 可到 10ms/trial（100× trials）
3. **Pin offset** — 从 `PlacementCost` 抽 pin 偏移替代 macro 中心，预计 WL 再降 ~5%
4. **Legalization 升级** — spiral search 换成 Tetris / Abacus / min-cost flow，减小 legalize 退化
5. **Nesterov 优化器** — 替代 Adam，匹配 DREAMPlace 原论文，可能少跑一半步数
6. **float64 消除 CPU/GPU 差异** — CPU 0.9668 vs GPU 0.9730（deterministic 后）仍差 0.6%；若要 CPU/GPU 完全一致，全程 float64
7. **Warm start 互通** — 把 `global + legalize` 结果存成 .pt，给 SA 团队当 warm start 试 hybrid 方案
