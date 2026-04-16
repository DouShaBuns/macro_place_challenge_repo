# DREAMPlace-Style Analytical Attempt

Challenge-native DREAMPlace-style implementation.

It is not a direct import of upstream DREAMPlace. The challenge API exposes a
`Benchmark` tensor object rather than Bookshelf/OpenDB inputs, so this directory
implements the relevant ideas directly in PyTorch:

- weighted-average wirelength over the extracted pin-level netlist
- bin density overflow
- hard-macro overlap and boundary penalties
- Nesterov/Adam continuous optimization
- legalization
- SA-style refinement from analytical warm starts
- official `compute_proxy_cost` reranking

Run:

```powershell
.\.venv\Scripts\Activate.ps1
$env:UV_CACHE_DIR='D:\workspace\partcl-macro-place-challenge\.uv-cache'
uv run evaluate team_trash_Workspace/dreamplace_gpu/placer.py -b ibm01
```

Useful knobs:

```powershell
$env:DP_DEVICE='cuda'
$env:DP_ANALYTICAL_ITERS='260'
$env:DP_REFINE_ITERS='80'
$env:DP_REFINE_CANDIDATE_BATCH='16'
$env:DP_SEEDS='42,43,44,45'
$env:DP_LOCAL_REFINE_TRIALS='220'
```

## Pure Analytical Full IBM Run 2026-04-16

This run disables the SA-style refinement stage and records the current
analytical-only baseline before optimization.

Environment:

```text
torch = 2.11.0+cu128
torch.version.cuda = 12.8
torch.cuda.is_available() = True
GPU = NVIDIA GeForce RTX 4060 Laptop GPU
```

Command:

```powershell
$env:DP_RUN_REFINE='0'
$env:DP_DEVICE='cuda:0'
uv run python team_trash_Workspace/dreamplace_gpu/parallel_runner.py --all --out team_trash_Workspace/dreamplace_gpu/results/analytical_only_full_20260416_142548.jsonl
```

Result:

```text
mode             = Pure analytical, no SA refine
result file      = team_trash_Workspace/dreamplace_gpu/results/analytical_only_full_20260416_142548.jsonl
average proxy    = 1.4930
valid benchmarks = 17 / 17
overlap count    = 0 on every benchmark
total runtime    = 1983.95 s
average runtime  = 116.70 s / benchmark
```

Per-benchmark results:

| Benchmark | Proxy | Wirelength | Density | Congestion | Overlaps | Runtime |
|-----------|------:|-----------:|--------:|-----------:|---------:|--------:|
| ibm01 | 1.1625 | 0.0723 | 0.9195 | 1.2610 | 0 | 25.11s |
| ibm02 | 1.6287 | 0.0768 | 0.7936 | 2.3103 | 0 | 32.92s |
| ibm03 | 1.4003 | 0.0804 | 0.8135 | 1.8262 | 0 | 25.15s |
| ibm04 | 1.3793 | 0.0717 | 0.8573 | 1.7580 | 0 | 29.97s |
| ibm06 | 1.7230 | 0.0637 | 0.8148 | 2.5037 | 0 | 31.89s |
| ibm07 | 1.4869 | 0.0656 | 0.8406 | 2.0021 | 0 | 47.03s |
| ibm08 | 1.5063 | 0.0691 | 0.8656 | 2.0088 | 0 | 59.16s |
| ibm09 | 1.1325 | 0.0578 | 0.8656 | 1.2837 | 0 | 41.96s |
| ibm10 | 1.4048 | 0.0689 | 0.7538 | 1.9181 | 0 | 189.90s |
| ibm11 | 1.2298 | 0.0544 | 0.8974 | 1.4535 | 0 | 62.35s |
| ibm12 | 1.6618 | 0.0603 | 0.8198 | 2.3832 | 0 | 185.10s |
| ibm13 | 1.4053 | 0.0540 | 0.9180 | 1.7845 | 0 | 75.29s |
| ibm14 | 1.5980 | 0.0523 | 0.9715 | 2.1200 | 0 | 215.17s |
| ibm15 | 1.6065 | 0.0586 | 0.9376 | 2.1582 | 0 | 132.28s |
| ibm16 | 1.5241 | 0.0497 | 0.8615 | 2.0873 | 0 | 252.81s |
| ibm17 | 1.7428 | 0.0540 | 0.9506 | 2.4270 | 0 | 433.41s |
| ibm18 | 1.7889 | 0.0533 | 1.0433 | 2.4279 | 0 | 144.45s |

## Optimized Pure Analytical Full IBM Run 2026-04-16

Conservative changes:

- Fixed candidate de-duplication to compare rounded per-macro coordinates instead of only `(sum(x), sum(y))`.
- Enabled post-legalization local refinement in pure analytical mode.
- Reranks local-refine candidates with the official `compute_proxy_cost`, so worse local moves cannot replace the best analytical candidate.
- Added `DP_LOCAL_REFINE_TRIALS` and used the default `220` trial cap for this run.

Commands:

```powershell
$env:DP_RUN_REFINE='0'
$env:DP_DEVICE='cuda:0'
uv run --extra dev pytest team_trash_Workspace/sa_gpu/tests
uv run evaluate team_trash_Workspace/dreamplace_gpu/placer.py -b ibm01
uv run evaluate team_trash_Workspace/dreamplace_gpu/placer.py -b ibm02
uv run evaluate team_trash_Workspace/dreamplace_gpu/placer.py -b ibm10
uv run python team_trash_Workspace/dreamplace_gpu/parallel_runner.py --all --out team_trash_Workspace/dreamplace_gpu/results/analytical_optimized_full_20260416_142548.jsonl
uv run python team_trash_Workspace/dreamplace_gpu/parallel_runner.py --benchmarks ibm15 ibm16 ibm17 ibm18 --out team_trash_Workspace/dreamplace_gpu/results/analytical_optimized_remaining_20260416_142548.jsonl
```

The first full command timed out after completing `ibm01` through `ibm14`; the remaining four benchmarks were resumed with the same code and environment. The combined record is:

```text
result file      = team_trash_Workspace/dreamplace_gpu/results/analytical_optimized_full_completed_20260416_142548.jsonl
mode             = Pure analytical, no SA refine, local refine enabled
average proxy    = 1.4929
valid benchmarks = 17 / 17
overlap count    = 0 on every benchmark
total runtime    = 10550.46 s
average runtime  = 620.62 s / benchmark
RePlAce average  = 1.4578
vs RePlAce avg   = 2.40% worse
RePlAce wins     = 3 / 17 benchmarks
```

Per-benchmark results:

| Benchmark | Proxy | Wirelength | Density | Congestion | Overlaps | Runtime |
|-----------|------:|-----------:|--------:|-----------:|---------:|--------:|
| ibm01 | 1.1622 | 0.0722 | 0.9193 | 1.2607 | 0 | 77.34s |
| ibm02 | 1.6275 | 0.0768 | 0.7914 | 2.3100 | 0 | 142.21s |
| ibm03 | 1.4003 | 0.0804 | 0.8135 | 1.8262 | 0 | 126.27s |
| ibm04 | 1.3793 | 0.0717 | 0.8573 | 1.7580 | 0 | 146.14s |
| ibm06 | 1.7230 | 0.0637 | 0.8148 | 2.5037 | 0 | 165.91s |
| ibm07 | 1.4868 | 0.0656 | 0.8405 | 2.0019 | 0 | 252.07s |
| ibm08 | 1.5063 | 0.0691 | 0.8656 | 2.0087 | 0 | 314.58s |
| ibm09 | 1.1325 | 0.0578 | 0.8656 | 1.2837 | 0 | 216.11s |
| ibm10 | 1.4044 | 0.0689 | 0.7531 | 1.9178 | 0 | 1023.90s |
| ibm11 | 1.2298 | 0.0544 | 0.8974 | 1.4535 | 0 | 374.54s |
| ibm12 | 1.6617 | 0.0603 | 0.8195 | 2.3833 | 0 | 1294.60s |
| ibm13 | 1.4053 | 0.0540 | 0.9180 | 1.7845 | 0 | 405.74s |
| ibm14 | 1.5980 | 0.0523 | 0.9715 | 2.1200 | 0 | 1456.02s |
| ibm15 | 1.6065 | 0.0586 | 0.9376 | 2.1582 | 0 | 857.25s |
| ibm16 | 1.5241 | 0.0497 | 0.8615 | 2.0873 | 0 | 1419.61s |
| ibm17 | 1.7428 | 0.0540 | 0.9506 | 2.4270 | 0 | 1795.62s |
| ibm18 | 1.7889 | 0.0533 | 1.0433 | 2.4278 | 0 | 482.55s |

Before/after proxy comparison:

| Benchmark | Before | Optimized | Delta |
|-----------|-------:|----------:|------:|
| ibm01 | 1.1625 | 1.1622 | -0.0003 |
| ibm02 | 1.6287 | 1.6275 | -0.0012 |
| ibm03 | 1.4003 | 1.4003 | -0.0000 |
| ibm04 | 1.3793 | 1.3793 | +0.0000 |
| ibm06 | 1.7230 | 1.7230 | +0.0000 |
| ibm07 | 1.4869 | 1.4868 | -0.0001 |
| ibm08 | 1.5063 | 1.5063 | -0.0000 |
| ibm09 | 1.1325 | 1.1325 | +0.0000 |
| ibm10 | 1.4048 | 1.4044 | -0.0005 |
| ibm11 | 1.2298 | 1.2298 | +0.0000 |
| ibm12 | 1.6618 | 1.6617 | -0.0001 |
| ibm13 | 1.4053 | 1.4053 | +0.0000 |
| ibm14 | 1.5980 | 1.5980 | +0.0000 |
| ibm15 | 1.6065 | 1.6065 | +0.0000 |
| ibm16 | 1.5241 | 1.5241 | +0.0000 |
| ibm17 | 1.7428 | 1.7428 | +0.0000 |
| ibm18 | 1.7889 | 1.7889 | -0.0000 |

Interpretation:

- The optimized run preserves legality: every benchmark is valid with zero hard-macro overlap.
- The proxy improvement is real but very small: average proxy improves from `1.4930` to `1.4929`.
- The local refinement default is expensive on large benchmarks; use `DP_LOCAL_REFINE_TRIALS=0` to reproduce the faster pre-optimization analytical behavior.
