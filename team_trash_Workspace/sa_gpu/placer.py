from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from sa_optimizer import SAConfig, SAOptimizer  # noqa: E402


class SAGPUPlacer:
    def __init__(
        self,
        seed: int | None = None,
        seeds: tuple[int, ...] | None = None,
        iters: int | None = None,
        candidate_batch: int | None = None,
        device: str | None = None,
    ):
        if seeds is None:
            if seed is None:
                seed_text = os.getenv("SA_GPU_SEEDS", "42,43,44,45")
                seeds = tuple(int(x) for x in seed_text.split(",") if x.strip())
            else:
                seeds = (int(seed),)
        self.config = SAConfig(
            seeds=seeds,
            iters=int(iters if iters is not None else os.getenv("SA_GPU_ITERS", "80")),
            candidate_batch=int(
                candidate_batch if candidate_batch is not None else os.getenv("SA_GPU_CANDIDATE_BATCH", "16")
            ),
            overlap_weight=float(os.getenv("SA_GPU_OVERLAP_WEIGHT", "1000.0")),
            boundary_weight=float(os.getenv("SA_GPU_BOUNDARY_WEIGHT", "1000.0")),
        )
        self.device = device or os.getenv("SA_GPU_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

    def place(self, benchmark) -> torch.Tensor:
        optimizer = SAOptimizer(self.config, device=self.device)
        return optimizer.optimize(benchmark)
