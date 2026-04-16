from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from torch_replace import RePlAceConfig, TorchRePlAceOptimizer  # noqa: E402


class RePlAcePlacer:
    def __init__(self):
        self.config = RePlAceConfig(
            iterations=int(os.getenv("REPLACE_ITERS", "600")),
            learning_rate=float(os.getenv("REPLACE_LR", "0.08")),
            momentum=float(os.getenv("REPLACE_MOMENTUM", "0.92")),
            wirelength_weight=float(os.getenv("REPLACE_WL_WEIGHT", "1.0")),
            density_weight=float(os.getenv("REPLACE_DENSITY_WEIGHT", "0.20")),
            boundary_weight=float(os.getenv("REPLACE_BOUNDARY_WEIGHT", "10.0")),
            bin_grid_count=int(os.getenv("REPLACE_BIN_GRID_COUNT", "32")),
            smooth_gamma_scale=float(os.getenv("REPLACE_GAMMA_SCALE", "0.02")),
            target_density=float(os.getenv("REPLACE_TARGET_DENSITY", "0.70")),
            optimize_soft_macros=os.getenv("REPLACE_OPTIMIZE_SOFT", "1") != "0",
        )
        self.device = os.getenv("REPLACE_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

    def place(self, benchmark):
        optimizer = TorchRePlAceOptimizer(self.config, device=self.device)
        return optimizer.optimize(benchmark)
