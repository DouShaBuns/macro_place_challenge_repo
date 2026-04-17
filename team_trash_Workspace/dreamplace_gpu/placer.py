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


def _parse_recipes(text: str | None) -> tuple[tuple[float, float, float, float], ...] | None:
    if text is None or not text.strip():
        return None
    recipes = []
    for chunk in text.split(";"):
        if not chunk.strip():
            continue
        values = [float(item.strip()) for item in chunk.split(",") if item.strip()]
        if len(values) != 4:
            raise ValueError(
                "DP_RECIPES entries must be density,gamma,lr,target; "
                f"got {chunk!r}"
            )
        recipes.append((values[0], values[1], values[2], values[3]))
    if not recipes:
        raise ValueError("DP_RECIPES did not contain any recipe entries")
    return tuple(recipes)


class DreamPlaceGPUPlacer:
    def __init__(self):
        seed_text = os.getenv("DP_SEEDS", "42,43,44,45")
        seeds = tuple(int(x) for x in seed_text.split(",") if x.strip())
        recipes = _parse_recipes(os.getenv("DP_RECIPES"))
        self.config = DreamPlaceConfig(
            analytical_iters=int(os.getenv("DP_ANALYTICAL_ITERS", "80")),
            refine_iters=int(os.getenv("DP_REFINE_ITERS", "80")),
            refine_candidate_batch=int(os.getenv("DP_REFINE_CANDIDATE_BATCH", "16")),
            seeds=seeds,
            density_weight=float(os.getenv("DP_DENSITY_WEIGHT", "0.18")),
            congestion_weight=float(os.getenv("DP_CONGESTION_WEIGHT", "0.05")),
            congestion_target=float(os.getenv("DP_CONGESTION_TARGET", "0.85")),
            congestion_density_alpha=float(os.getenv("DP_CONGESTION_DENSITY_ALPHA", "0.15")),
            congestion_map_update_interval=int(os.getenv("DP_CONGESTION_MAP_UPDATE_INTERVAL", "20")),
            soft_route_congestion_weight=float(os.getenv("DP_SOFT_ROUTE_CONGESTION_WEIGHT", "0.02")),
            soft_route_tau_scale=float(os.getenv("DP_SOFT_ROUTE_TAU_SCALE", "0.5")),
            soft_route_chunk_size=int(os.getenv("DP_SOFT_ROUTE_CHUNK_SIZE", "512")),
            overlap_weight=float(os.getenv("DP_OVERLAP_WEIGHT", "18.0")),
            boundary_weight=float(os.getenv("DP_BOUNDARY_WEIGHT", "25.0")),
            optimize_soft_macros=os.getenv("DP_OPTIMIZE_SOFT", "1") != "0",
            run_refine=os.getenv("DP_RUN_REFINE", "0") != "0",
            local_refine_trials=int(os.getenv("DP_LOCAL_REFINE_TRIALS", "0")),
            **({"recipes": recipes} if recipes is not None else {}),
        )
        self.device = os.getenv("DP_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

    def place(self, benchmark):
        optimizer = DreamPlaceHybridOptimizer(self.config, device=self.device)
        return optimizer.optimize(benchmark)
