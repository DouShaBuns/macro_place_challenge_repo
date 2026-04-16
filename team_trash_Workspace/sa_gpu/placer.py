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
        top_k_final_candidates: int | None = None,
        local_refine_max_trials: int | None = None,
        search_congestion_mode: str | None = None,
        warm_start=None,
        warm_start_path: str | None = None,
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
            top_k_final_candidates=int(
                top_k_final_candidates
                if top_k_final_candidates is not None
                else os.getenv("SA_GPU_TOP_K_FINAL_CANDIDATES", "32")
            ),
            local_refine_max_trials=int(
                local_refine_max_trials
                if local_refine_max_trials is not None
                else os.getenv("SA_GPU_LOCAL_REFINE_MAX_TRIALS", "1000")
            ),
            search_congestion_mode=(
                search_congestion_mode
                if search_congestion_mode is not None
                else os.getenv("SA_GPU_SEARCH_CONGESTION_MODE", "fast")
            ),
        )
        self.warm_start = warm_start
        self.warm_start_path = warm_start_path or os.getenv("SA_GPU_WARM_START_PATH")
        self.device = device or os.getenv("SA_GPU_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

    def place(self, benchmark) -> torch.Tensor:
        optimizer = SAOptimizer(
            self.config,
            device=self.device,
            warm_start=self.warm_start,
            warm_start_path=self.warm_start_path,
        )
        return optimizer.optimize(benchmark)
