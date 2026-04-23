# DreamPlace GPU

这个目录是面向当前 challenge API 的 DreamPlace 风格宏块布局实现。它不是直接搬运上游 DREAMPlace，而是基于仓库里的 `Benchmark` / `PlacementCost` 接口，重写了一套可在当前数据格式上运行的 PyTorch 流程。

当前实现重点是：

- analytical placement
- soft macro relaxation
- official `compute_proxy_cost` 复排和复核
- 动态调度器（GPU primary + opportunistic backfill）
- 可恢复 pause/checkpoint
- NG45 结果保存和 ORFS 联动验证

当前代码入口都要求使用 `uv`。不要直接用裸 `python` / `pip`。

## 目录结构

| 路径 | 作用 |
|------|------|
| `team_trash_Workspace/dreamplace_gpu/placer.py` | challenge placer 入口，解析 `DP_*` 环境变量 |
| `team_trash_Workspace/dreamplace_gpu/optimizer.py` | 主优化流程：analytical、soft relax、official rerank、local refine |
| `team_trash_Workspace/dreamplace_gpu/parallel_runner.py` | 单 benchmark / 小批量运行器，支持保存 JSONL 和串接 ORFS |
| `team_trash_Workspace/dreamplace_gpu/global_scheduler.py` | 全局调度器，支持 GPU backfill、pause/resume、checkpoint |
| `team_trash_Workspace/dreamplace_gpu/benchmark_context.py` | 构建 net / routing 上下文并缓存 PLC |
| `team_trash_Workspace/dreamplace_gpu/trace_utils.py` | 过程 trace、GIF、placement snapshot |
| `scripts/evaluate_with_orfs.py` | 用官方 ORFS 路径验证 placement，输出 WNS/TNS/Area |
| `scripts/generate_macro_placement_tcl.py` | 将 placement 转成 ORFS / OpenROAD 可用的 `macros.tcl` |
| `scripts/render_saved_placement.py` | 将已保存的 placement `.pt` 渲染成 PNG |

## 当前默认行为

### 1. 默认保存最终 placement

每个 benchmark 跑完后默认保存：

```text
output/dreamplace_gpu/placements/<benchmark>.pt
output/dreamplace_gpu/placements/<benchmark>.csv
```

- `.pt`：`[num_macros, 2]` 的中心坐标 tensor
- `.csv`：hard macro 坐标、尺寸、fixed 标记

相关开关：

```text
DP_PLACEMENT_DIR                 # 修改输出目录
DP_SAVE_FINAL_PLACEMENT=0        # 关闭最终 placement 保存
```

### 2. 中间 placement 默认不保存

需要时手动打开：

```bash
DP_SAVE_INTERMEDIATE_PLACEMENTS=1
```

输出目录：

```text
output/traces/dreamplace_gpu/<benchmark>/placements/
```

### 3. ORFS 默认优先复用预综合网表

当前默认逻辑是：

- `parallel_runner.py` / `global_scheduler.py` 在 ORFS 验证时默认设置  
  `DP_RUNNER_ORFS_SKIP_SYNTHESIS=1`
- `scripts/evaluate_with_orfs.py` 默认也优先走 `--skip-synthesis`

这表示：

- 默认不重新跑 Yosys synthesis
- 直接复用已有预综合网表来评估你的宏布局
- 更适合比较“布局本身”带来的 WNS/TNS/Area 变化

如果你明确想跑完整 flow，可显式关闭：

```bash
DP_RUNNER_ORFS_SKIP_SYNTHESIS=0
```

## 运行方式

### 单个 benchmark

```bash
DP_DEVICE=cuda:0 \
uv run python team_trash_Workspace/dreamplace_gpu/parallel_runner.py \
  --benchmarks ibm01 \
  --out team_trash_Workspace/dreamplace_gpu/results/ibm01.jsonl
```

### IBM 全量

```bash
DP_DEVICE=cuda:0 \
uv run python team_trash_Workspace/dreamplace_gpu/parallel_runner.py \
  --all \
  --out team_trash_Workspace/dreamplace_gpu/results/ibm_full.jsonl
```

IBM 这组主要看：

- `proxy_cost`
- `overlaps`
- `runtime`

它不走 ORFS，也不输出 `WNS/TNS/Area`。

### NG45 全量 + ORFS 验证

```bash
DP_DEVICE=cuda:0 \
DP_RESOURCE_MODE=throughput \
DP_OFFICIAL_FINAL_ONLY=1 \
DP_RUNNER_ISOLATE=1 \
DP_RUNNER_ORFS_VALIDATE=1 \
DP_RUNNER_ORFS_SKIP_SYNTHESIS=1 \
uv run python team_trash_Workspace/dreamplace_gpu/global_scheduler.py \
  --ng45 \
  --orfs-validate \
  --orfs-root ../OpenROAD-flow-scripts \
  --out team_trash_Workspace/dreamplace_gpu/results/ng45_orfs.jsonl \
  --work-dir team_trash_Workspace/dreamplace_gpu/results/ng45_orfs_work \
  --max-gpu-jobs 1 \
  --max-gpu-burst-jobs 2 \
  --schedule large-first
```

这条命令会：

1. 跑 DreamPlace placement
2. 默认保存 placement `.pt/.csv`
3. 在同一条命令里继续调用 ORFS
4. 把结果写回同一个 JSONL

### 只对已有 placement 跑 ORFS

```bash
uv run python scripts/evaluate_with_orfs.py \
  --benchmark ariane136_ng45 \
  --orfs-root ../OpenROAD-flow-scripts \
  --placement output/dreamplace_gpu/placements/ariane136_ng45.pt
```

默认就是预综合网表模式。

## 调度器说明

`global_scheduler.py` 现在不是简单串行队列，而是一个偏“可抢占”的资源调度器：

- `1` 个 primary GPU job
- GPU 空闲时可启动 backfill job
- primary 需要资源时，会请求 backfill worker 暂停
- 暂停时优先写 checkpoint，再恢复

默认 checkpoint backend 是：

```text
shm
```

即 Linux 下优先走：

```text
/dev/shm/dreamplace_gpu/
```

相关变量：

```text
DP_CHECKPOINT_BACKEND=shm|disk
DP_CHECKPOINT_PATH
DP_PAUSE_REQUEST_PATH
```

调度器相关参数：

```text
--max-gpu-jobs
--max-gpu-burst-jobs
--gpu-idle-util-threshold
--gpu-backfill-free-mib
--gpu-evict-util-threshold
--backfill-grace-seconds
--pause-timeout-seconds
```

## 当前已知状态

### 稳定部分

- IBM / NG45 placement 主流程能跑
- 最终 placement 默认能保存
- NG45 placement 能渲染成静态 PNG
- ORFS 桥接脚本已经补齐：
  - 支持现成 OpenROAD 配置目录
  - 支持从 `scripts/OpenROAD/*.tar.gz` 自动解包
  - 支持最小 fallback ORFS design 生成
  - `generate_macro_placement_tcl.py` 已增加 direct-instance fallback，不再只支持 SRAM `macro_mem[K]` 命名

### 仍在收敛的问题

- NG45 的 ORFS 还没有完全稳定
- 当前一轮远端 `presynth` 全量里：
  - `ariane133_ng45` / `ariane136_ng45` 已进入 `orfs_validate`
  - `mempool_tile_ng45` / `nvdla_ng45` 已拿到 DreamPlace proxy cost，但 ORFS 返回 `code 2`
- 所以当前 README 不再承诺“NG45 默认一定能稳定产出完整 WNS/TNS/Area”

换句话说：

- DreamPlace placement：当前可用
- placement 保存 / 渲染：当前可用
- ORFS 联动：当前在修通中，可跑，但需要继续排查 benchmark-specific ORFS 失败

## 结果文件格式

`parallel_runner.py` / `global_scheduler.py` 输出 JSONL，每行一个 benchmark。

常见字段：

```json
{
  "name": "ariane136_ng45",
  "proxy_cost": 0.8316795826,
  "wirelength": 0.0519,
  "density": 0.8200,
  "congestion": 0.7396,
  "overlaps": 0,
  "valid": false,
  "runtime": 34.82,
  "placement_paths": {
    "placement_pt": "output/dreamplace_gpu/placements/ariane136_ng45.pt",
    "placement_csv": "output/dreamplace_gpu/placements/ariane136_ng45.csv"
  },
  "orfs": {
    "wns": ...,
    "tns": ...,
    "area": ...
  }
}
```

如果 ORFS 失败，`orfs` 里会是：

```json
{
  "error": "ORFS flow failed with code 2"
}
```

## 渲染 placement 图片

当前仓库已经支持把保存下来的 `.pt` 直接渲染成 PNG。

### 单张图

```bash
uv run python scripts/render_saved_placement.py \
  --placement output/dreamplace_gpu/placements/ariane136_ng45.pt \
  --benchmark ariane136_ng45 \
  --out output/rendered_placements/ariane136_ng45.png
```

输出图包含三栏：

1. placement
2. density heatmap
3. congestion heatmap

### 当前已经渲染好的示例

本地已有：

```text
output/rendered_placements/ariane133_ng45.png
output/rendered_placements/ariane136_ng45.png
output/rendered_placements/mempool_tile_ng45.png
output/rendered_placements/nvdla_ng45.png
```

## Trace / GIF

如果要保留优化过程图：

```bash
TRACE_PLACEMENT=1 \
TRACE_EVERY=10 \
TRACE_DIR=output/traces \
uv run evaluate team_trash_Workspace/dreamplace_gpu/placer.py -b ibm01
```

会输出：

```text
output/traces/dreamplace_gpu/<benchmark>/dreamplace_gpu_<benchmark>.gif
```

如果要保留中间 PNG frame：

```bash
TRACE_KEEP_FRAMES=1
```

## 常用环境变量

```text
DP_DEVICE
DP_RESOURCE_MODE
DP_OFFICIAL_FINAL_ONLY
DP_ANALYTICAL_ITERS
DP_RECIPES
DP_SEEDS
DP_TOP_K_CANDIDATES
DP_MAX_GPU_BATCH_CANDIDATES
DP_SOFT_RELAX_ITERS
DP_SOFT_RELAX_LR_SCALES
DP_LOCAL_REFINE_TRIALS
DP_LOG_PROXY_CALIBRATION
DP_SAVE_FINAL_PLACEMENT
DP_SAVE_INTERMEDIATE_PLACEMENTS
DP_RUNNER_ORFS_VALIDATE
DP_RUNNER_ORFS_SKIP_SYNTHESIS
DP_RUNNER_ORFS_NO_DOCKER
DP_CHECKPOINT_BACKEND
```

## 建议的使用方式

如果你的目标是优化布局本身，而不是重新评估综合质量，建议分两层：

1. 日常调参
   - 跑 DreamPlace
   - 默认保存 placement
   - 用预综合网表模式验证 ORFS

2. 最终确认
   - 固定 placement
   - 再决定是否关闭 `skip-synthesis` 跑完整 flow

这样能把“布局变化”和“综合变化”分开看，结果更容易解释。
