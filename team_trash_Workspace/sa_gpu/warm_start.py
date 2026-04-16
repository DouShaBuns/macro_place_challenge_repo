from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch

from legalize import clamp_placement, legalize_initial, legalize_placement


@runtime_checkable
class WarmStartProvider(Protocol):
    def generate(self, benchmark) -> torch.Tensor:
        ...


class InitialPlacementWarmStart:
    def generate(self, benchmark) -> torch.Tensor:
        return legalize_initial(benchmark)


class TensorWarmStart:
    def __init__(self, placement: torch.Tensor):
        self.placement = placement

    def generate(self, benchmark) -> torch.Tensor:
        return self.placement


class PathWarmStart:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def generate(self, benchmark) -> torch.Tensor:
        placement = torch.load(self.path, map_location="cpu", weights_only=False)
        if isinstance(placement, dict):
            for key in ("placement", "macro_positions", "positions"):
                if key in placement:
                    placement = placement[key]
                    break
        if not isinstance(placement, torch.Tensor):
            placement = torch.as_tensor(placement, dtype=torch.float32)
        return placement


class MethodWarmStart:
    def __init__(self, method: Any):
        self.method = method

    def generate(self, benchmark) -> torch.Tensor:
        if callable(self.method) and not hasattr(self.method, "place") and not hasattr(self.method, "generate"):
            return self.method(benchmark)
        if hasattr(self.method, "generate"):
            return self.method.generate(benchmark)
        if hasattr(self.method, "place"):
            return self.method.place(benchmark)
        raise TypeError("Warm-start method must be callable or provide generate()/place().")


def build_warm_start_provider(source: Any = None, path: str | Path | None = None) -> WarmStartProvider:
    if path:
        return PathWarmStart(path)
    if source is None:
        return InitialPlacementWarmStart()
    if isinstance(source, (str, Path)):
        return PathWarmStart(source)
    if isinstance(source, torch.Tensor):
        return TensorWarmStart(source)
    if isinstance(source, WarmStartProvider):
        return source
    return MethodWarmStart(source)


def prepare_warm_start(benchmark, provider: WarmStartProvider) -> torch.Tensor:
    placement = provider.generate(benchmark)
    if not isinstance(placement, torch.Tensor):
        placement = torch.as_tensor(placement, dtype=torch.float32)
    placement = placement.detach().cpu().to(dtype=torch.float32)
    if placement.shape == (benchmark.num_hard_macros, 2):
        full = benchmark.macro_positions.clone().to(dtype=torch.float32)
        full[: benchmark.num_hard_macros] = placement
        placement = full
    if placement.shape != (benchmark.num_macros, 2):
        raise ValueError(
            "Warm-start placement must have shape "
            f"({benchmark.num_macros}, 2) or ({benchmark.num_hard_macros}, 2), got {tuple(placement.shape)}."
        )
    return clamp_placement(legalize_placement(placement, benchmark), benchmark)
