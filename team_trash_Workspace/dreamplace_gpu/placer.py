from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

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


def _parse_float_tuple(text: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if text is None or not text.strip():
        return default
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise ValueError("Expected at least one float value")
    return values


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text).strip("_") or "benchmark"


def _save_final_placement(placement: torch.Tensor, benchmark) -> dict[str, str]:
    if not _env_bool("DP_SAVE_FINAL_PLACEMENT", True):
        return {}

    name = _safe_name(str(getattr(benchmark, "name", "benchmark")))
    out_dir = Path(os.getenv("DP_PLACEMENT_DIR", "output/dreamplace_gpu/placements"))
    out_dir.mkdir(parents=True, exist_ok=True)

    placement_cpu = placement.detach().cpu()
    tensor_path = out_dir / f"{name}.pt"
    csv_path = out_dir / f"{name}.csv"
    torch.save(placement_cpu, tensor_path)

    sizes = getattr(benchmark, "macro_sizes", None)
    fixed = getattr(benchmark, "macro_fixed", None)
    num_hard = int(getattr(benchmark, "num_hard_macros", placement_cpu.shape[0]))
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["macro_index", "x_center", "y_center", "width", "height", "fixed"])
        for idx in range(min(num_hard, placement_cpu.shape[0])):
            width = float(sizes[idx, 0].item()) if sizes is not None else ""
            height = float(sizes[idx, 1].item()) if sizes is not None else ""
            is_fixed = bool(fixed[idx].item()) if fixed is not None else ""
            writer.writerow(
                [
                    idx,
                    float(placement_cpu[idx, 0].item()),
                    float(placement_cpu[idx, 1].item()),
                    width,
                    height,
                    is_fixed,
                ]
            )

    return {"placement_pt": str(tensor_path), "placement_csv": str(csv_path)}


class DreamPlaceGPUPlacer:
    def __init__(self):
        seed_text = os.getenv("DP_SEEDS", "42,43,44,45")
        seeds = tuple(int(x) for x in seed_text.split(",") if x.strip())
        recipes = _parse_recipes(os.getenv("DP_RECIPES"))
        optimizer_name = os.getenv("DP_OPTIMIZER", "adam")
        self.config = DreamPlaceConfig(
            optimizer_name=optimizer_name,
            analytical_optimizer_name=os.getenv("DP_ANALYTICAL_OPTIMIZER", optimizer_name),
            soft_relax_optimizer_name=os.getenv("DP_SOFT_RELAX_OPTIMIZER", optimizer_name),
            analytical_iters=int(os.getenv("DP_ANALYTICAL_ITERS", "80")),
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
            optimize_soft_macros=_env_bool("DP_OPTIMIZE_SOFT", True),
            top_k_candidates=int(os.getenv("DP_TOP_K_CANDIDATES", "8")),
            official_rerank_limit=int(os.getenv("DP_OFFICIAL_RERANK_LIMIT", "0")),
            max_gpu_batch_candidates=int(os.getenv("DP_MAX_GPU_BATCH_CANDIDATES", "0")),
            local_refine_trials=int(os.getenv("DP_LOCAL_REFINE_TRIALS", "0")),
            official_refine_evals=int(os.getenv("DP_OFFICIAL_REFINE_EVALS", "24")),
            official_refine_macro_limit=int(os.getenv("DP_OFFICIAL_REFINE_MACRO_LIMIT", "1000")),
            official_refine_rounds=int(os.getenv("DP_OFFICIAL_REFINE_ROUNDS", "1")),
            official_refine_prefilter_chunk=int(os.getenv("DP_OFFICIAL_REFINE_PREFILTER_CHUNK", "64")),
            official_refine_full_prefilter_factor=int(os.getenv("DP_OFFICIAL_REFINE_FULL_PREFILTER_FACTOR", "0")),
            official_refine_verify_top_k=int(os.getenv("DP_OFFICIAL_REFINE_VERIFY_TOP_K", "8")),
            official_refine_step_scales=_parse_float_tuple(
                os.getenv("DP_OFFICIAL_REFINE_STEP_SCALES"),
                (0.25, 0.5, 1.0),
            ),
            analytical_snapshot_interval=int(os.getenv("DP_ANALYTICAL_SNAPSHOT_INTERVAL", "20")),
            analytical_snapshots_per_recipe=int(os.getenv("DP_ANALYTICAL_SNAPSHOTS_PER_RECIPE", "3")),
            soft_relax_iters=int(os.getenv("DP_SOFT_RELAX_ITERS", "200")),
            soft_relax_lr_scale=float(os.getenv("DP_SOFT_RELAX_LR_SCALE", "0.01")),
            soft_relax_lr_scales=_parse_float_tuple(
                os.getenv("DP_SOFT_RELAX_LR_SCALES"),
                (0.005, 0.01),
            ),
            soft_relax_start_k=int(os.getenv("DP_SOFT_RELAX_START_K", "1")),
            soft_relax_snapshot_interval=int(os.getenv("DP_SOFT_RELAX_SNAPSHOT_INTERVAL", "20")),
            soft_relax_snapshots=int(os.getenv("DP_SOFT_RELAX_SNAPSHOTS", "8")),
            soft_relax_official_eval_limit=int(os.getenv("DP_SOFT_RELAX_OFFICIAL_EVAL_LIMIT", "4")),
            batched_soft_relax=_env_bool("DP_BATCHED_SOFT_RELAX", True),
            adaptive_large_budget=_env_bool("DP_ADAPTIVE_LARGE_BUDGET", True),
            log_proxy_calibration=_env_bool("DP_LOG_PROXY_CALIBRATION", False),
            resource_mode=os.getenv("DP_RESOURCE_MODE", "balanced"),
            official_final_only=_env_bool("DP_OFFICIAL_FINAL_ONLY", False),
            **({"recipes": recipes} if recipes is not None else {}),
        )
        self.device = os.getenv("DP_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.last_profile: dict[str, float] = {}
        self.last_placement_paths: dict[str, str] = {}

    def place(self, benchmark):
        optimizer = DreamPlaceHybridOptimizer(self.config, device=self.device)
        placement = optimizer.optimize(benchmark)
        self.last_profile = dict(getattr(optimizer, "last_profile", {}))
        self.last_placement_paths = _save_final_placement(placement, benchmark)
        return placement
