# DREAMPlace-Style Analytical Attempt

Challenge-native DREAMPlace-style implementation.

It is not a direct import of upstream DREAMPlace. The challenge API exposes a
`Benchmark` tensor object rather than Bookshelf/OpenDB inputs, so this directory
implements the relevant ideas directly in PyTorch:

- weighted-average wirelength over the extracted pin-level netlist
- bin density overflow
- congestion-aware analytical losses
- hard-macro overlap and boundary penalties
- Nesterov/Adam continuous optimization
- legalization
- optional SA-style refinement from analytical warm starts
- official `compute_proxy_cost` reranking

The default mode is strict pure analytical: analytical candidates are generated,
legalized, reranked with the official proxy cost, and returned. SA refinement and
post-legalization local coordinate refinement are both disabled unless explicitly
enabled with environment variables.

## Current Status

Current default flow:

1. Generate analytical macro-placement candidates in PyTorch.
2. Repair only illegal candidates; legal candidates are kept unchanged.
3. Rerank candidates with the official `compute_proxy_cost`.
4. Run hard-macro-fixed soft macro relaxation with adaptive per-design policy.
5. Rerank soft-relax snapshots with the official proxy.
6. Run official local hard-macro refinement only when the adaptive policy allows
   it.

The adaptive policy is based on benchmark features rather than benchmark names:
`num_hard_macros`, `num_soft_macros`, total macro count, and official baseline
congestion. This is intended to make unknown designs choose a reasonable budget
without adding case-specific `ibmXX` rules.

Current performance-tuned defaults:

- PLC objects are cached in `BenchmarkContext` and reused inside a placement
  run.
- Official proxy calibration logging is disabled by default; enable it with
  `DP_LOG_PROXY_CALIBRATION=1`.
- Very large hard-macro cases skip official local refinement by default because
  the measured proxy gain was tiny compared with the runtime cost.
- Hard-macro overlap checking uses a hybrid implementation: scalar loop for
  smaller cases, vectorized NumPy for larger cases.

## Run Guide

The project uses `uv` for local Python execution. Do not call bare `python` or
`pip` for normal local runs.

Single benchmark:

```powershell
.\.venv\Scripts\Activate.ps1
$env:UV_CACHE_DIR='D:\workspace\partcl-macro-place-challenge\.uv-cache'
uv run evaluate team_trash_Workspace/dreamplace_gpu/placer.py -b ibm01
```

Runner with JSONL output:

```powershell
$env:DP_DEVICE='cuda:0'
uv run python team_trash_Workspace/dreamplace_gpu/parallel_runner.py --benchmarks ibm01 --out team_trash_Workspace/dreamplace_gpu/results/latest.jsonl
```

All ICCAD04 IBM cases:

```powershell
$env:DP_DEVICE='cuda:0'
uv run python team_trash_Workspace/dreamplace_gpu/parallel_runner.py --all --out team_trash_Workspace/dreamplace_gpu/results/full_current.jsonl
```

Focused verification:

```powershell
uv run --extra dev pytest team_trash_Workspace/dreamplace_gpu/tests
```

Use this when you intentionally want diagnostic official/torch calibration logs:

```powershell
$env:DP_LOG_PROXY_CALIBRATION='1'
```

Useful knobs:

```powershell
$env:DP_DEVICE='cuda'
$env:DP_ANALYTICAL_ITERS='260'
$env:DP_RECIPES='0.12,0.018,0.030,0.78;0.18,0.025,0.035,0.82;0.28,0.035,0.030,0.88'
$env:DP_CONGESTION_WEIGHT='0.05'
$env:DP_CONGESTION_TARGET='0.85'
$env:DP_CONGESTION_DENSITY_ALPHA='0.15'
$env:DP_CONGESTION_MAP_UPDATE_INTERVAL='20'
$env:DP_SOFT_ROUTE_CONGESTION_WEIGHT='0.02'
$env:DP_SOFT_ROUTE_TAU_SCALE='0.5'
$env:DP_SOFT_ROUTE_CHUNK_SIZE='512'
$env:DP_SEEDS='42,43,44,45'
$env:DP_TOP_K_CANDIDATES='8'
$env:DP_OFFICIAL_RERANK_LIMIT='0'
$env:DP_MAX_GPU_BATCH_CANDIDATES='0'
$env:DP_SOFT_RELAX_ITERS='200'
$env:DP_SOFT_RELAX_LR_SCALES='0.005,0.01'
$env:DP_OFFICIAL_REFINE_EVALS='24'
$env:DP_ADAPTIVE_LARGE_BUDGET='1'
```

Large-memory single-GPU starting point:

```powershell
$env:DP_SOFT_ROUTE_CHUNK_SIZE='1024'
$env:DP_OFFICIAL_REFINE_PREFILTER_CHUNK='256'
$env:DP_MAX_GPU_BATCH_CANDIDATES='256'
```

If GPU memory is still low and utilization is stable, try
`DP_SOFT_ROUTE_CHUNK_SIZE=2048` and
`DP_OFFICIAL_REFINE_PREFILTER_CHUNK=512`. `DP_OFFICIAL_RERANK_LIMIT=0`
means the official rerank prefilter keeps `2 * DP_TOP_K_CANDIDATES`
candidates when the candidate pool is larger than that; set an explicit larger
value if you want more official `compute_proxy_cost` calls for quality.

Optional hybrid/refinement knobs:

```powershell
$env:DP_RUN_REFINE='1'
$env:DP_REFINE_ITERS='80'
$env:DP_REFINE_CANDIDATE_BATCH='16'
$env:DP_LOCAL_REFINE_TRIALS='220'
```

## File Index

| Path | Purpose |
|------|---------|
| `team_trash_Workspace/dreamplace_gpu/placer.py` | Challenge placer entry point; parses `DP_*` environment variables into `DreamPlaceConfig`. |
| `team_trash_Workspace/dreamplace_gpu/optimizer.py` | Main analytical placement, soft macro relaxation, adaptive policy, official reranking, and local refinement logic. |
| `team_trash_Workspace/dreamplace_gpu/parallel_runner.py` | Convenience runner for IBM/NG45 batches with JSONL output. |
| `team_trash_Workspace/dreamplace_gpu/benchmark_context.py` | Builds GPU-ready netlist/routing tensors and caches the official PLC object. |
| `team_trash_Workspace/dreamplace_gpu/torch_objective.py` | Torch proxy evaluator used for fast candidate screening and differentiable routing/density terms. |
| `team_trash_Workspace/dreamplace_gpu/legalize.py` | Challenge-native legalization and canvas clamping helpers. |
| `team_trash_Workspace/dreamplace_gpu/trace_utils.py` | Optional placement trace GIF recorder. |
| `macro_place/objective.py` | Official `PlacementCost` wrapper and overlap metrics used for final scoring. |
| `macro_place/loader.py` | Benchmark loader; normalizes paths for the official parser on Windows. |
| `team_trash_Workspace/dreamplace_gpu/results/*.jsonl` | Recorded benchmark runs used for score comparisons. |

## Optimization Trace GIF

Tracing is disabled by default. Enable it with environment variables before
running the normal evaluator. For a pure analytical trace:

```powershell
$env:TRACE_PLACEMENT='1'
$env:TRACE_EVERY='10'
$env:TRACE_DIR='output/traces'
$env:TRACE_FPS='8'
uv run evaluate team_trash_Workspace/dreamplace_gpu/placer.py -b ibm01
```

This writes:

```text
output/traces/dreamplace_gpu/ibm01/dreamplace_gpu_ibm01.gif
```

The GIF records the initial legalized placement, periodic frames from each
analytical recipe, each recipe best placement, the best legalized candidate,
and the final placement. Set `TRACE_KEEP_FRAMES=1` to keep the intermediate PNG
frames.

## Pure Analytical Full IBM Run 2026-04-16

This historical run disabled the SA-style refinement stage and recorded the
analytical-only baseline before congestion-aware analytical losses were added.

Environment:

```text
torch = 2.11.0+cu128
torch.version.cuda = 12.8
torch.cuda.is_available() = True
GPU = NVIDIA GeForce RTX 4060 Laptop GPU
```

Command:

```powershell
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
uv run --extra dev pytest team_trash_Workspace/dreamplace_gpu/tests
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
- Local refinement is now optional and disabled by default; use `DP_LOCAL_REFINE_TRIALS=220` only when intentionally reproducing the older local-refine experiment.

## Soft Macro Relax + Adaptive Budget IBM Results 2026-04-18

This run series adds hard-macro-fixed soft macro relaxation, official proxy
reranking of soft-relax snapshots, adaptive per-benchmark runtime budgets, and
official local hard-macro refinement. The table below reports the best valid
under-1-hour result observed for each ICCAD04 IBM benchmark.

The results were collected in batches because a single full `--all` run takes
longer than the interactive command timeout. Each row is valid, has zero hard
macro overlap, and has runtime below 3600 seconds.

Representative result files:

```text
team_trash_Workspace/dreamplace_gpu/results/full_soft_relax_20260417_224532.jsonl
team_trash_Workspace/dreamplace_gpu/results/remaining_soft40_20260418_011149.jsonl
team_trash_Workspace/dreamplace_gpu/results/remaining_adaptive_20260418_033935.jsonl
team_trash_Workspace/dreamplace_gpu/results/verify_ibm01_adaptive_small.jsonl
team_trash_Workspace/dreamplace_gpu/results/verify_ibm14_adaptive_budget.jsonl
team_trash_Workspace/dreamplace_gpu/results/verify_ibm15_adaptive_budget.jsonl
team_trash_Workspace/dreamplace_gpu/results/verify_ibm17_adaptive_route.jsonl
```

Summary:

```text
mode             = Analytical placement + soft macro relax + official rerank
average proxy    = 1.300185
valid benchmarks = 17 / 17
overlap count    = 0 on every benchmark
RePlAce average  = 1.457841
vs RePlAce avg   = 10.81% better
RePlAce wins     = 0 / 17 benchmarks
```

RePlAce comparison:

| Benchmark | RePlAce Proxy | New Proxy | Delta | Delta % | Runtime |
|-----------|--------------:|----------:|------:|--------:|--------:|
| ibm01 | 0.997600 | 0.988129 | -0.009471 | -0.95% | 269.3s |
| ibm02 | 1.837000 | 1.410176 | -0.426824 | -23.23% | 452.2s |
| ibm03 | 1.322200 | 1.176604 | -0.145596 | -11.01% | 309.2s |
| ibm04 | 1.302400 | 1.160112 | -0.142288 | -10.93% | 424.6s |
| ibm06 | 1.618700 | 1.402912 | -0.215788 | -13.33% | 570.7s |
| ibm07 | 1.463300 | 1.273514 | -0.189786 | -12.97% | 805.6s |
| ibm08 | 1.428500 | 1.258862 | -0.169638 | -11.88% | 1044.3s |
| ibm09 | 1.119400 | 0.976735 | -0.142665 | -12.74% | 649.6s |
| ibm10 | 1.500900 | 1.303089 | -0.197811 | -13.18% | 1480.1s |
| ibm11 | 1.177400 | 1.089981 | -0.087419 | -7.42% | 812.0s |
| ibm12 | 1.726100 | 1.526685 | -0.199415 | -11.55% | 1297.4s |
| ibm13 | 1.335500 | 1.232522 | -0.102978 | -7.71% | 610.7s |
| ibm14 | 1.543600 | 1.448644 | -0.094956 | -6.15% | 2093.2s |
| ibm15 | 1.515900 | 1.403562 | -0.112338 | -7.41% | 1338.6s |
| ibm16 | 1.478000 | 1.346073 | -0.131927 | -8.93% | 2226.2s |
| ibm17 | 1.644600 | 1.536553 | -0.108047 | -6.57% | 3411.1s |
| ibm18 | 1.772200 | 1.568985 | -0.203215 | -11.47% | 2314.3s |

Interpretation:

- The new flow beats the RePlAce proxy on every listed IBM benchmark.
- The average proxy improves from `1.457841` to `1.300185`.
- The largest gains come from soft macro post-relaxation and official-reranked
  adaptive budget allocation on congestion-heavy cases.
- The closest remaining margins are `ibm01`, `ibm11`, `ibm14`, `ibm13`, and
  `ibm17`; these are the next targets for density and routing surrogate tuning.

## Performance-Tuned Current Defaults 2026-04-19

This is the current working version after runtime tuning. It trades a very small
amount of best-known proxy score for lower CPU-side stalls on large cases:

- disables diagnostic proxy calibration by default;
- reuses the parsed PLC from `BenchmarkContext`;
- uses hybrid overlap checking;
- skips official local refinement for `num_hard_macros >= 700` under adaptive
  mode.
- vectorizes the analytical congestion-map routing accumulation with chunked
  dense tensor masks instead of one Python loop per routing pair;
- batches Torch prefiltering before official rerank, with
  `DP_MAX_GPU_BATCH_CANDIDATES` as an optional memory cap;
- vectorizes SA-style candidate generation by move type when
  `DP_RUN_REFINE=1`.

Current expected full IBM average:

```text
mode             = Analytical placement + soft macro relax + adaptive runtime tuning
average proxy    = 1.300402
valid benchmarks = 17 / 17
overlap count    = 0 on every benchmark
RePlAce average  = 1.457841
vs RePlAce avg   = 10.80% better
```

Compared with the 2026-04-18 best-quality table, the expected average changes
from `1.300185` to `1.300402`. The difference is `+0.000217`, mainly because
large-case official local refinement is skipped by default.

Current expected per-benchmark proxy:

| Benchmark | Current Proxy | Notes |
|-----------|--------------:|-------|
| ibm01 | 0.988710 | Re-run after performance tuning, 243.04s. |
| ibm02 | 1.410176 | Same best valid result as previous table. |
| ibm03 | 1.176604 | Same best valid result as previous table. |
| ibm04 | 1.160112 | Same best valid result as previous table. |
| ibm06 | 1.402912 | Same best valid result as previous table. |
| ibm07 | 1.273514 | Same best valid result as previous table. |
| ibm08 | 1.258862 | Same best valid result as previous table. |
| ibm09 | 0.976735 | Same best valid result as previous table. |
| ibm10 | 1.305628 | Re-run after skipping very-large official local refine, 784.09s. |
| ibm11 | 1.089981 | Same best valid result as previous table. |
| ibm12 | 1.526685 | Same best valid result as previous table. |
| ibm13 | 1.232522 | Same best valid result as previous table. |
| ibm14 | 1.448644 | Same best valid result as previous table. |
| ibm15 | 1.403562 | Same best valid result as previous table. |
| ibm16 | 1.346073 | Same best valid result as previous table. |
| ibm17 | 1.537128 | Expected current behavior: soft-relax best, skipping official local refine. |
| ibm18 | 1.568985 | Same best valid result as previous table. |

Performance smoke checks:

| Case | Check | Before | Current | Result |
|------|-------|-------:|--------:|--------|
| ibm01 | Normal short recipe soft-relax run | 269.31s | 243.04s | About 9.8% faster; proxy remains below RePlAce. |
| ibm17 | One-iteration analytical smoke, soft relax disabled | 440.30s | 300.92s | About 31.7% faster after disabling calibration and large-case official refine. |

Use `DP_ADAPTIVE_LARGE_BUDGET=0` if you want to reproduce the slower
best-quality behavior with large-case official local refinement enabled.

## Latest Measured Full Run 2026-04-22

Latest downloaded full-run artifact:

```text
team_trash_Workspace/dreamplace_gpu/results/full_shm_preempt_20260422T221418Z.jsonl
```

Configuration:

```text
mode                  = throughput + official-final-only
scheduler             = dynamic GPU backfill with pause/resume
checkpoint backend    = shm
GPU jobs              = 1 primary + 1 opportunistic backfill
```

Measured summary:

```text
average proxy    = 1.273975
average runtime  = 34.48 s / benchmark
valid benchmarks = 17 / 17
overlap count    = 0 on every benchmark
```

Interpretation:

- This is the first full run with recoverable backfill pauses through shared
  memory checkpoints (`/dev/shm` on Linux).
- It is slightly worse than the earlier non-checkpoint throughput full run
  (`1.272149`) by `+0.001826`, but still materially better than the older
  runtime-tuned baseline (`1.300402`).
- The scheduler path is now much closer to preemptive behavior: backfill work
  is paused, checkpointed, and resumed instead of always being killed and
  restarted from scratch.

Measured per-benchmark proxy:

| Benchmark | Proxy | Runtime | Launch Kind |
|-----------|------:|--------:|-------------|
| ibm01 | 0.996172 | 0.78s | backfill |
| ibm02 | 1.406462 | 29.74s | backfill |
| ibm03 | 1.158355 | 0.98s | backfill |
| ibm04 | 1.155363 | 26.57s | backfill |
| ibm06 | 1.388359 | 1.10s | backfill |
| ibm07 | 1.278944 | 32.59s | backfill |
| ibm08 | 1.252629 | 36.49s | backfill |
| ibm09 | 0.987355 | 31.54s | backfill |
| ibm10 | 1.265398 | 71.97s | primary |
| ibm11 | 1.058859 | 23.80s | backfill |
| ibm12 | 1.459537 | 57.19s | backfill |
| ibm13 | 1.140109 | 25.74s | backfill |
| ibm14 | 1.404858 | 58.64s | backfill |
| ibm15 | 1.385061 | 50.69s | backfill |
| ibm16 | 1.300769 | 3.32s | backfill |
| ibm17 | 1.435642 | 82.01s | primary |
| ibm18 | 1.583700 | 53.04s | backfill |
