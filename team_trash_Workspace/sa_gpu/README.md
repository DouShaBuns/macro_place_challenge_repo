# GPU Batched SA Placer

This implementation is self-contained under `team_trash_Workspace/sa_gpu` and does not modify the main project package.

## Run With Official Evaluator

```powershell
$env:UV_CACHE_DIR='D:\workspace\partcl-macro-place-challenge\.uv-cache'
uv run evaluate team_trash_Workspace/sa_gpu/placer.py -b ibm01
uv run evaluate team_trash_Workspace/sa_gpu/placer.py --all
```

The official evaluator loads `placer.py` by path. `SAGPUPlacer` is defined directly in that file so it satisfies the loader's `attr.__module__ == path.stem` check.

## Parallel Runner

```powershell
uv run python team_trash_Workspace/sa_gpu/parallel_runner.py --all --seeds 1 2 3 4
uv run python team_trash_Workspace/sa_gpu/parallel_runner.py --benchmarks ibm01 --seeds 1 2 --candidate-batch 8 --iters 5
```

The runner parallelizes across benchmarks with `ProcessPoolExecutor`. Seeds are handled as a torch batch inside each worker so one process does not spawn many competing GPU contexts.

## Environment Knobs

```powershell
$env:SA_GPU_ITERS='120'
$env:SA_GPU_CANDIDATE_BATCH='32'
$env:SA_GPU_SEEDS='42,43,44,45,46,47'
$env:SA_GPU_DEVICE='cuda'
```

## Optimization Trace GIF

Tracing is disabled by default. Enable it with environment variables before
running the normal evaluator:

```powershell
$env:TRACE_PLACEMENT='1'
$env:TRACE_EVERY='10'
$env:TRACE_DIR='output/traces'
$env:TRACE_FPS='8'
uv run evaluate team_trash_Workspace/sa_gpu/placer.py -b ibm01
```

This writes:

```text
output/traces/sa_gpu/ibm01/sa_gpu_ibm01.gif
```

The GIF records the initial legalized placement, periodic best placements from
the annealing loop, and the final legalized placement. Set
`TRACE_KEEP_FRAMES=1` to keep the intermediate PNG frames.

## Cost Model

The search proxy uses torch batched computations:

- pin-level HPWL using reloaded `PlacementCost` net/pin mappings
- hard macro overlap and boundary legality
- grid density from macro/grid rectangle intersections
- source-to-sink L-routing congestion approximation with macro blockage and smoothing

Final reported scores still come from the repository's official `compute_proxy_cost`.

## Implementation And Simulation Log

Date: 2026-04-14.

Implemented modules:

- `placer.py`: official-loader entry point with `SAGPUPlacer` defined directly in the file.
- `benchmark_context.py`: reloads `PlacementCost` and extracts pin, net, port, grid, and routing tensors.
- `torch_objective.py`: batched torch proxy evaluator for wirelength, overlap, density, and congestion.
- `sa_optimizer.py`: multi-seed simulated annealing with batched candidates per seed.
- `legalize.py`: initial and final legalization, canvas clamp, and fallback shelf packing.
- `parallel_runner.py`: custom multi-benchmark runner with device assignment and JSONL output.
- `tests/`: smoke tests for evaluator shape, overlap/legalization behavior, loader compatibility, and runner behavior.

Problems found during implementation and simulation:

- `uv run` initially synchronized the environment back to CPU torch even after installing a CUDA wheel manually. The lockfile had to be regenerated with the PyTorch CUDA index so normal `uv run` resolves CUDA torch.
- A full CPU run was too slow for useful iteration and was stopped before completion. CUDA execution was required for full-benchmark testing.
- The first full CUDA run used a dense 4D congestion tensor over `[batch, routing_pair, row, col]`, which is risky on large IBM cases and small GPUs. Congestion routing demand is now accumulated in chunks of 512 routing pairs.
- The original runner wrote results only after all benchmarks completed. If a late benchmark failed, earlier results were lost. The runner now writes JSONL incrementally and records per-benchmark errors.
- Torch-side legality and the official overlap metrics were not identical in early runs. Some placements passed `validate_placement` but still had nonzero `compute_proxy_cost` overlap count. The final result now always runs an additional legalization pass before returning.
- The fallback shelf packer originally ignored fixed hard macros as occupied regions. It now treats fixed macros as already placed obstacles.

Environment verified:

```text
torch = 2.11.0+cu128
torch.version.cuda = 12.8
torch.cuda.is_available() = True
GPU = NVIDIA GeForce RTX 4060 Laptop GPU
```

Validation commands:

```powershell
$env:UV_CACHE_DIR='D:\workspace\partcl-macro-place-challenge\.uv-cache'
uv run --extra dev pytest team_trash_Workspace/sa_gpu/tests
uv run python team_trash_Workspace/sa_gpu/parallel_runner.py --all --seeds 1 2 3 4 --candidate-batch 16 --iters 80 --devices cuda:0 --out team_trash_Workspace/sa_gpu/results/full_cuda_after_legalize.jsonl
```

Smoke tests passed:

```text
3 passed
```

Full IBM result after final legalization:

```text
average proxy cost = 1.5347
valid benchmarks   = 17 / 17
overlap count      = 0 on every benchmark
total runtime      = 1608 s
average runtime    = 94.6 s / benchmark
```

Per-benchmark results:

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

## Full IBM Rerun 2026-04-16

Environment:

```text
torch = 2.11.0+cu128
torch.version.cuda = 12.8
torch.cuda.is_available() = True
GPU = NVIDIA GeForce RTX 4060 Laptop GPU
```

Validation commands:

```powershell
uv run --extra dev pytest team_trash_Workspace/sa_gpu/tests
uv run python team_trash_Workspace/sa_gpu/parallel_runner.py --all --seeds 1 2 3 4 --candidate-batch 16 --iters 80 --devices cuda:0 --out team_trash_Workspace/sa_gpu/results/full_cuda_rerun_20260416_142548.jsonl
```

Smoke tests passed:

```text
3 passed
```

Full IBM rerun result:

```text
result file       = team_trash_Workspace/sa_gpu/results/full_cuda_rerun_20260416_142548.jsonl
average proxy     = 1.5332
valid benchmarks  = 17 / 17
overlap count     = 0 on every benchmark
total runtime     = 2436.57 s
average runtime   = 143.33 s / benchmark
```

Per-benchmark results:

| Benchmark | Proxy | Wirelength | Density | Congestion | Overlaps | Runtime |
|-----------|------:|-----------:|--------:|-----------:|---------:|--------:|
| ibm01 | 1.2608 | 0.0891 | 0.9176 | 1.4259 | 0 | 27.83s |
| ibm02 | 1.6383 | 0.0809 | 0.7925 | 2.3223 | 0 | 25.92s |
| ibm03 | 1.4498 | 0.0866 | 0.8272 | 1.8993 | 0 | 24.81s |
| ibm04 | 1.4670 | 0.0771 | 0.8652 | 1.9147 | 0 | 26.81s |
| ibm06 | 1.8138 | 0.0672 | 0.8547 | 2.6384 | 0 | 31.37s |
| ibm07 | 1.5154 | 0.0681 | 0.8510 | 2.0436 | 0 | 43.33s |
| ibm08 | 1.5406 | 0.0722 | 0.8814 | 2.0553 | 0 | 56.35s |
| ibm09 | 1.1453 | 0.0608 | 0.8654 | 1.3037 | 0 | 46.52s |
| ibm10 | 1.4427 | 0.0715 | 0.7523 | 1.9900 | 0 | 254.71s |
| ibm11 | 1.2704 | 0.0575 | 0.9033 | 1.5224 | 0 | 79.77s |
| ibm12 | 1.7083 | 0.0625 | 0.8247 | 2.4669 | 0 | 510.47s |
| ibm13 | 1.4615 | 0.0568 | 0.9280 | 1.8815 | 0 | 344.24s |
| ibm14 | 1.6230 | 0.0540 | 0.9731 | 2.1650 | 0 | 379.55s |
| ibm15 | 1.6112 | 0.0601 | 0.9400 | 2.1621 | 0 | 114.65s |
| ibm16 | 1.5727 | 0.0518 | 0.8680 | 2.1739 | 0 | 150.11s |
| ibm17 | 1.7506 | 0.0554 | 0.9505 | 2.4398 | 0 | 207.72s |
| ibm18 | 1.7928 | 0.0542 | 1.0435 | 2.4336 | 0 | 112.41s |

Leaderboard interpretation from the repository README:

- The score beats the SA baseline average by about 27.8%.
- It beats the RePlAce baseline on 3 of 17 benchmarks (`ibm02`, `ibm10`, `ibm12`), but is about 5.3% worse than the RePlAce average.
- It is about 5.4% worse than the current README rank-8 public score (`1.4568`) and about 6.6% worse than the rank-7 score (`1.4403`).
- Practical status: zero-overlap and stronger than SA, but not yet competitive with the current public leaderboard cutoff.

Next optimization targets:

- Improve final placement quality after legalization, because legality is now stable but legalization can slightly degrade proxy.
- Replace the simple L-routing congestion proxy with a closer approximation to the official 3-pin and multi-pin branch behavior.
- Add a local post-legalization refinement pass that preserves zero overlap while reducing HPWL and congestion.
- Tune SA move mix, temperature schedule, and candidate batch size per benchmark family.
