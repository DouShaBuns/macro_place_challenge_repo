from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_SA_GPU = _HERE.parent / "sa_gpu"
for path in (_HERE, _SA_GPU):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from optimizer import DreamPlaceConfig, DreamPlaceHybridOptimizer  # noqa: E402


class DreamPlaceGPUPlacer:
    def __init__(self):
        seed_text = os.getenv("DP_SEEDS", "42,43,44,45")
        seeds = tuple(int(x) for x in seed_text.split(",") if x.strip())
        self.config = DreamPlaceConfig(
            analytical_iters=int(os.getenv("DP_ANALYTICAL_ITERS", "80")),
            refine_iters=int(os.getenv("DP_REFINE_ITERS", "80")),
            refine_candidate_batch=int(os.getenv("DP_REFINE_CANDIDATE_BATCH", "16")),
            seeds=seeds,
            density_weight=float(os.getenv("DP_DENSITY_WEIGHT", "0.18")),
            overlap_weight=float(os.getenv("DP_OVERLAP_WEIGHT", "18.0")),
            boundary_weight=float(os.getenv("DP_BOUNDARY_WEIGHT", "25.0")),
            optimize_soft_macros=os.getenv("DP_OPTIMIZE_SOFT", "1") != "0",
            run_refine=os.getenv("DP_RUN_REFINE", "1") != "0",
            local_refine_trials=int(os.getenv("DP_LOCAL_REFINE_TRIALS", "220")),
        )
        self.device = os.getenv("DP_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

    def place(self, benchmark):
        optimizer = DreamPlaceHybridOptimizer(self.config, device=self.device)
        return optimizer.optimize(benchmark)
