from __future__ import annotations

from dataclasses import dataclass

import torch

from macro_place.benchmark import Benchmark

from challenge_legalize import clamp_placement, legalize_placement


@dataclass
class RePlAceConfig:
    iterations: int = 600
    learning_rate: float = 0.08
    momentum: float = 0.92
    wirelength_weight: float = 1.0
    density_weight: float = 0.20
    boundary_weight: float = 10.0
    bin_grid_count: int = 32
    smooth_gamma_scale: float = 0.02
    target_density: float = 0.70
    optimize_soft_macros: bool = True


class TorchRePlAceOptimizer:
    """Small RePlAce-style analytical placer for the challenge tensor API.

    This is not the OpenROAD C++ implementation. It follows the same broad
    idea: smooth wirelength plus bin-density spreading optimized with a
    Nesterov-like continuous update, followed by hard-macro legalization.
    """

    def __init__(self, config: RePlAceConfig | None = None, device: str | torch.device | None = None):
        self.config = config or RePlAceConfig()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    def optimize(self, benchmark: Benchmark) -> torch.Tensor:
        placement = clamp_placement(benchmark.macro_positions, benchmark).to(self.device, dtype=torch.float32)
        original = benchmark.macro_positions.to(self.device, dtype=torch.float32)
        sizes = benchmark.macro_sizes.to(self.device, dtype=torch.float32)
        edge_i, edge_j, edge_w = self._make_edges(benchmark)
        fixed = benchmark.macro_fixed.to(self.device)
        movable = ~fixed
        if not self.config.optimize_soft_macros:
            movable = movable & benchmark.get_hard_macro_mask().to(self.device)

        x = placement.clone().detach()
        prev = x.clone()
        best = x.clone()
        best_loss = float("inf")

        base_lr = self.config.learning_rate * max(float(benchmark.canvas_width), float(benchmark.canvas_height))
        for step in range(max(int(self.config.iterations), 1)):
            y = x + self.config.momentum * (x - prev)
            y = self._project(y, benchmark, sizes, fixed, original)
            y = y.detach().requires_grad_(True)

            loss = self._objective(y, benchmark, sizes, edge_i, edge_j, edge_w)
            loss.backward()
            grad = torch.nan_to_num(y.grad, nan=0.0, posinf=0.0, neginf=0.0)
            grad = grad * movable.view(-1, 1).to(grad.dtype)

            grad_norm = grad.norm(dim=1).median().clamp_min(1.0e-6)
            lr = base_lr * (0.25 + 0.75 * (1.0 - step / max(float(self.config.iterations), 1.0)))
            update = lr * grad / grad_norm

            prev = x
            x = self._project(y.detach() - update, benchmark, sizes, fixed, original)

            loss_value = float(loss.detach().item())
            if loss_value < best_loss:
                best_loss = loss_value
                best = x.clone()

        final = legalize_placement(best.detach().cpu(), benchmark)
        return clamp_placement(final, benchmark).cpu()

    def _objective(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        sizes: torch.Tensor,
        edge_i: torch.Tensor,
        edge_j: torch.Tensor,
        edge_w: torch.Tensor,
    ) -> torch.Tensor:
        wl = self._edge_wirelength(placement, benchmark, edge_i, edge_j, edge_w)
        density = self._density_overflow(placement, benchmark, sizes)
        boundary = self._boundary_penalty(placement, benchmark, sizes)
        return (
            self.config.wirelength_weight * wl
            + self.config.density_weight * density
            + self.config.boundary_weight * boundary
        )

    def _make_edges(self, benchmark: Benchmark) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src: list[int] = []
        dst: list[int] = []
        weights: list[float] = []
        for net_id, nodes in enumerate(benchmark.net_nodes):
            if nodes.numel() <= 1:
                continue
            node_list = [int(x) for x in nodes.tolist()]
            valid = [idx for idx in node_list if 0 <= idx < benchmark.num_macros]
            if len(valid) <= 1:
                continue
            root = valid[0]
            weight = float(benchmark.net_weights[net_id].item()) / max(len(valid) - 1, 1)
            for idx in valid[1:]:
                src.append(root)
                dst.append(idx)
                weights.append(weight)
        if not src:
            empty_long = torch.zeros(0, dtype=torch.long, device=self.device)
            empty_float = torch.zeros(0, dtype=torch.float32, device=self.device)
            return empty_long, empty_long, empty_float
        return (
            torch.tensor(src, dtype=torch.long, device=self.device),
            torch.tensor(dst, dtype=torch.long, device=self.device),
            torch.tensor(weights, dtype=torch.float32, device=self.device),
        )

    def _edge_wirelength(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        edge_i: torch.Tensor,
        edge_j: torch.Tensor,
        edge_w: torch.Tensor,
    ) -> torch.Tensor:
        if edge_i.numel() == 0:
            return placement.new_tensor(0.0)
        gamma = self.config.smooth_gamma_scale * max(float(benchmark.canvas_width), float(benchmark.canvas_height))
        gamma = max(gamma, 1.0e-3)
        diff = placement[edge_i] - placement[edge_j]
        smooth_l1 = torch.sqrt(diff.square() + gamma * gamma).sum(dim=1)
        total = (edge_w.to(dtype=placement.dtype) * smooth_l1).sum()
        norm = max(
            (float(benchmark.canvas_width) + float(benchmark.canvas_height)) * max(float(edge_w.sum().item()), 1.0),
            1.0e-9,
        )
        return total / norm

    def _density_overflow(self, placement: torch.Tensor, benchmark: Benchmark, sizes: torch.Tensor) -> torch.Tensor:
        grid = min(max(int(self.config.bin_grid_count), 8), 64)
        grid = min(grid, max(int(benchmark.grid_rows), 1), max(int(benchmark.grid_cols), 1))
        if grid <= 1:
            return placement.new_tensor(0.0)

        xs = (torch.arange(grid, device=self.device, dtype=placement.dtype) + 0.5) * float(benchmark.canvas_width) / grid
        ys = (torch.arange(grid, device=self.device, dtype=placement.dtype) + 0.5) * float(benchmark.canvas_height) / grid
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        centers = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

        bin_w = float(benchmark.canvas_width) / grid
        bin_h = float(benchmark.canvas_height) / grid
        sigma_x = torch.clamp(sizes[:, 0] * 0.5 + bin_w, min=bin_w)
        sigma_y = torch.clamp(sizes[:, 1] * 0.5 + bin_h, min=bin_h)
        dx = (placement[:, 0:1] - centers[:, 0].view(1, -1)) / sigma_x.view(-1, 1)
        dy = (placement[:, 1:2] - centers[:, 1].view(1, -1)) / sigma_y.view(-1, 1)
        kernel = torch.exp(-0.5 * (dx.square() + dy.square()))
        area = (sizes[:, 0] * sizes[:, 1]).view(-1, 1)
        density = (kernel * area).sum(dim=0) / max(bin_w * bin_h, 1.0e-9)
        overflow = (density - float(self.config.target_density)).clamp_min(0)
        return overflow.square().mean()

    def _boundary_penalty(self, placement: torch.Tensor, benchmark: Benchmark, sizes: torch.Tensor) -> torch.Tensor:
        half = sizes / 2
        left = (half[:, 0] - placement[:, 0]).clamp_min(0)
        right = (placement[:, 0] + half[:, 0] - float(benchmark.canvas_width)).clamp_min(0)
        bottom = (half[:, 1] - placement[:, 1]).clamp_min(0)
        top = (placement[:, 1] + half[:, 1] - float(benchmark.canvas_height)).clamp_min(0)
        scale = max(float(benchmark.canvas_width) + float(benchmark.canvas_height), 1.0e-9)
        return (left + right + bottom + top).sum() / scale

    def _project(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        sizes: torch.Tensor,
        fixed: torch.Tensor,
        original: torch.Tensor,
    ) -> torch.Tensor:
        out = placement.clone()
        out[:, 0] = out[:, 0].clamp(sizes[:, 0] / 2, float(benchmark.canvas_width) - sizes[:, 0] / 2)
        out[:, 1] = out[:, 1].clamp(sizes[:, 1] / 2, float(benchmark.canvas_height) - sizes[:, 1] / 2)
        if bool(fixed.any()):
            out[fixed] = original[fixed]
        return out
