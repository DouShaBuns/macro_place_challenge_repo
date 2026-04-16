# GPU Batched SA Placer

本实现完全放在 `team_trash_Workspace/sa_gpu` 下，不修改主项目 package。

## 使用官方 Evaluator

```powershell
$env:UV_CACHE_DIR='D:\workspace\partcl-macro-place-challenge\.uv-cache'
uv run evaluate team_trash_Workspace/sa_gpu/placer.py -b ibm01
uv run evaluate team_trash_Workspace/sa_gpu/placer.py --all
```

官方 loader 会按文件路径加载 `placer.py`。`SAGPUPlacer` 直接定义在该文件内，以满足 loader 的 `attr.__module__ == path.stem` 检查。

## 并行 Runner

```powershell
uv run python team_trash_Workspace/sa_gpu/parallel_runner.py --all --seeds 1 2 3 4
uv run python team_trash_Workspace/sa_gpu/parallel_runner.py --benchmarks ibm01 --seeds 1 2 --candidate-batch 8 --iters 5
```

runner 使用 `ProcessPoolExecutor` 做 benchmark 级并行。seed 维度在每个 worker 内部作为 torch batch 处理，避免为多个 seed 创建多个互相抢 GPU 的进程。

## 环境参数

```powershell
$env:SA_GPU_ITERS='120'
$env:SA_GPU_CANDIDATE_BATCH='32'
$env:SA_GPU_SEEDS='42,43,44,45,46,47'
$env:SA_GPU_DEVICE='cuda'
$env:SA_GPU_WARM_START_PATH='team_trash_Workspace/sa_gpu/results/ibm01_placement.pt'
```

## Warm-start 接口

SA 默认仍从 benchmark 自带的 `initial.plc` 合法化后启动。为了后续接入 RePlAce 或其他初始解生成器，`SAGPUPlacer` 现在支持 warm-start provider：

```python
placer = SAGPUPlacer(warm_start=my_method)
```

`my_method` 可以是 callable，也可以实现 `generate(benchmark)` 或 `place(benchmark)`，返回 `[num_macros, 2]` 的完整 placement，或只返回 `[num_hard_macros, 2]` 的 hard macro placement。SA 会在开始退火前统一补齐 soft macro、恢复 fixed macro、legalize 和 clamp。

也可以直接从 torch 保存的 placement 启动：

```powershell
uv run python team_trash_Workspace/sa_gpu/parallel_runner.py --benchmarks ibm01 --warm-start-path path\to\placement.pt
```

## Cost Model

搜索 proxy 使用 torch batched 计算：

- 通过重新加载的 `PlacementCost` net/pin 映射计算 pin-level HPWL
- hard macro overlap 和 boundary legality
- macro/grid 矩形相交计算 density
- source-to-sink L-routing congestion 近似，包含 macro blockage 和 smoothing

最终报告分数仍使用仓库官方 `compute_proxy_cost`。

## 实现与仿真记录

日期：2026-04-14。

已实现模块：

- `placer.py`：官方 loader 入口，`SAGPUPlacer` 直接定义在文件内。
- `benchmark_context.py`：重新加载 `PlacementCost`，提取 pin、net、port、grid、routing tensor。
- `torch_objective.py`：batched torch proxy evaluator，覆盖 wirelength、overlap、density、congestion。
- `sa_optimizer.py`：多 seed simulated annealing，每个 seed 每轮生成 batched candidates。
- `legalize.py`：初始 legalization、最终 legalization、canvas clamp、fallback shelf packing。
- `parallel_runner.py`：自定义多 benchmark runner，支持 device 分配和 JSONL 输出。
- `tests/`：覆盖 evaluator shape、overlap/legalization、官方 loader 兼容性、runner smoke 行为。

实现和仿真中遇到的问题：

- 手动安装 CUDA wheel 后，普通 `uv run` 仍会按旧 `uv.lock` 同步回 CPU torch。需要用 PyTorch CUDA index 重新生成 lockfile，保证正常 `uv run` 解析到 CUDA torch。
- CPU 全量运行过慢，早期全量 CPU run 在完成前被停止。后续全量验证必须使用 CUDA。
- 第一版 CUDA congestion 使用 `[batch, routing_pair, row, col]` 的 dense 4D tensor，在大 IBM case 和小显存 GPU 上风险较高。现在 routing demand 按 512 条 routing pair 分块累加。
- runner 最初只在全部 benchmark 完成后写结果；如果后面的 benchmark 失败，前面已完成结果会丢失。现在改为增量写 JSONL，并记录单个 benchmark 的 error。
- 早期 torch-side legality 与官方 overlap metrics 不完全一致，出现过 `validate_placement` 通过但 `compute_proxy_cost` overlap count 非零的情况。现在最终返回前强制再做一次 legalization。
- fallback shelf packer 最初没有把 fixed hard macros 当作已占用障碍。现在 fixed macros 会先作为 placed obstacles 参与避让。

已验证环境：

```text
torch = 2.11.0+cu128
torch.version.cuda = 12.8
torch.cuda.is_available() = True
GPU = NVIDIA GeForce RTX 4060 Laptop GPU
```

验证命令：

```powershell
$env:UV_CACHE_DIR='D:\workspace\partcl-macro-place-challenge\.uv-cache'
uv run --extra dev pytest team_trash_Workspace/sa_gpu/tests
uv run python team_trash_Workspace/sa_gpu/parallel_runner.py --all --seeds 1 2 3 4 --candidate-batch 16 --iters 80 --devices cuda:0 --out team_trash_Workspace/sa_gpu/results/full_cuda_after_legalize.jsonl
```

smoke tests 结果：

```text
3 passed
```

最终 legalization 后的 IBM 全量结果：

```text
average proxy cost = 1.5347
valid benchmarks   = 17 / 17
overlap count      = 每个 benchmark 都是 0
total runtime      = 1608 s
average runtime    = 94.6 s / benchmark
```

逐 benchmark 结果：

| Benchmark | Proxy | Wirelength | Density | Congestion | Overlaps | Runtime |
|-----------|------:|-----------:|--------:|-----------:|---------:|--------:|
| ibm01 | 1.2935 | 0.0916 | 0.9177 | 1.4862 | 0 | 23.84s |
| ibm02 | 1.6383 | 0.0809 | 0.7925 | 2.3223 | 0 | 22.36s |
| ibm03 | 1.4498 | 0.0866 | 0.8272 | 1.8993 | 0 | 20.45s |
| ibm04 | 1.4670 | 0.0771 | 0.8652 | 1.9147 | 0 | 20.68s |
| ibm06 | 1.8138 | 0.0672 | 0.8547 | 2.6384 | 0 | 23.77s |
| ibm07 | 1.5154 | 0.0681 | 0.8510 | 2.0436 | 0 | 32.32s |
| ibm08 | 1.5406 | 0.0722 | 0.8814 | 2.0553 | 0 | 41.69s |
| ibm09 | 1.1460 | 0.0610 | 0.8659 | 1.3041 | 0 | 34.42s |
| ibm10 | 1.4427 | 0.0715 | 0.7523 | 1.9900 | 0 | 143.60s |
| ibm11 | 1.2704 | 0.0575 | 0.9033 | 1.5224 | 0 | 54.28s |
| ibm12 | 1.7091 | 0.0626 | 0.8244 | 2.4686 | 0 | 137.22s |
| ibm13 | 1.4531 | 0.0566 | 0.9277 | 1.8654 | 0 | 128.34s |
| ibm14 | 1.6230 | 0.0540 | 0.9731 | 2.1650 | 0 | 189.79s |
| ibm15 | 1.6114 | 0.0601 | 0.9415 | 2.1611 | 0 | 142.84s |
| ibm16 | 1.5727 | 0.0518 | 0.8680 | 2.1739 | 0 | 187.24s |
| ibm17 | 1.7510 | 0.0555 | 0.9508 | 2.4403 | 0 | 264.05s |
| ibm18 | 1.7928 | 0.0542 | 1.0435 | 2.4336 | 0 | 141.06s |

按仓库 README leaderboard 口径解读：

- 当前分数比 SA baseline 平均值好约 27.8%。
- 17 个 benchmark 中有 3 个赢过 RePlAce baseline：`ibm02`、`ibm10`、`ibm12`；但平均值仍比 RePlAce 差约 5.3%。
- 当前平均值比 README 第 8 名公开分数 `1.4568` 差约 5.4%，比第 7 名 `1.4403` 差约 6.6%。
- 实际状态：已经 zero-overlap，且明显强于 SA baseline；但距离公开 leaderboard cutoff 还有差距。

下一步优化目标：

- 改善 final legalization 后的 placement 质量，因为 legality 已稳定，但 legalization 会轻微拉高 proxy。
- 将简单 L-routing congestion proxy 替换为更接近官方 3-pin 和 multi-pin branch 行为的近似。
- 加入 post-legalization local refinement，在保持 zero overlap 的前提下降低 HPWL 和 congestion。
- 按 benchmark family 调整 SA move mix、temperature schedule 和 candidate batch size。
