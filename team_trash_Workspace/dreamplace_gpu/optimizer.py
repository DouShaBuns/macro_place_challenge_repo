from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch

_HERE = Path(__file__).resolve().parent
_SA_GPU = _HERE.parent / "sa_gpu"
if str(_SA_GPU) not in sys.path:
    sys.path.insert(0, str(_SA_GPU))

from benchmark_context import build_benchmark_context, load_plc_for_benchmark  # noqa: E402
from legalize import clamp_placement, legalize_initial, legalize_placement  # noqa: E402
from torch_objective import TorchProxyCostEvaluator  # noqa: E402

from macro_place.objective import compute_proxy_cost  # noqa: E402


@dataclass
class DreamPlaceConfig:
    analytical_iters: int = 260
    refine_iters: int = 80
    refine_candidate_batch: int = 16
    seeds: tuple[int, ...] = (42, 43, 44, 45)
    density_weight: float = 0.18
    overlap_weight: float = 18.0
    boundary_weight: float = 25.0
    target_density: float = 0.82
    gamma_scale: float = 0.025
    learning_rate: float = 0.035
    bin_grid_cap: int = 48
    optimize_soft_macros: bool = True
    run_refine: bool = True
    top_k_candidates: int = 8
    local_refine_trials: int = 220


class DreamPlaceHybridOptimizer:
    def __init__(self, config: DreamPlaceConfig | None = None, device: str | torch.device | None = None):
        self.config = config or DreamPlaceConfig()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    def optimize(self, benchmark) -> torch.Tensor:
        ctx = build_benchmark_context(benchmark, self.device)
        analytical = DreamPlaceAnalyticalOptimizer(self.config, self.device, ctx)
        candidates = analytical.generate_candidates(benchmark)

        legalized = [clamp_placement(legalize_placement(pos.cpu(), benchmark, gap=0.01), benchmark) for pos in candidates]
        legalized.append(legalize_placement(benchmark.macro_positions, benchmark, gap=0.01))
        legalized = self._unique_candidates(legalized)

        ranked = self._rank_official(legalized, benchmark)
        starts = [pos for _, pos in ranked[: max(1, min(self.config.top_k_candidates, len(ranked)))]]
        local = self._local_refine_best(starts, benchmark, ctx)
        if local:
            reranked = self._rank_official(starts + local, benchmark)
            starts = [pos for _, pos in reranked[: max(1, min(self.config.top_k_candidates, len(reranked)))]]
        if not self.config.run_refine or self.config.refine_iters <= 0:
            return starts[0].cpu()

        refiner = AnalyticalSARefiner(self.config, self.device, ctx)
        refined = refiner.refine(benchmark, starts)
        all_final = starts + refined
        ranked_final = self._rank_official(all_final, benchmark)
        return ranked_final[0][1].cpu()

    def _local_refine_best(self, candidates: list[torch.Tensor], benchmark, ctx) -> list[torch.Tensor]:
        ranked = self._rank_official(candidates, benchmark)
        if not ranked:
            return []
        evaluator = TorchProxyCostEvaluator(ctx, overlap_weight=1000.0, boundary_weight=1000.0)
        current = ranked[0][1].to(self.device, dtype=torch.float32)
        costs = evaluator.evaluate_batch(current)
        if not bool(costs.is_legal.item()):
            return []
        best = current.clone()
        best_score = costs.official_proxy.view(()).clone()
        movable = (benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()).to(self.device)
        movable_idx = torch.where(movable)[0].tolist()
        if not movable_idx:
            return [best.detach().cpu()]
        step_base = max(
            float(benchmark.canvas_width) / max(int(benchmark.grid_cols), 1),
            float(benchmark.canvas_height) / max(int(benchmark.grid_rows), 1),
        )
        steps = [0.75 * step_base, 0.35 * step_base, 0.15 * step_base]
        directions = ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))
        max_trials = max(int(self.config.local_refine_trials), 0)
        trials = 0
        for step in steps:
            improved = True
            while improved and trials < max_trials:
                improved = False
                for macro_idx in movable_idx:
                    if trials >= max_trials:
                        break
                    for dx, dy in directions:
                        trials += 1
                        cand = best.clone()
                        cand[macro_idx, 0] += float(dx) * step
                        cand[macro_idx, 1] += float(dy) * step
                        sizes = benchmark.macro_sizes.to(self.device, dtype=torch.float32)
                        cand[:, 0] = cand[:, 0].clamp(sizes[:, 0] / 2, float(benchmark.canvas_width) - sizes[:, 0] / 2)
                        cand[:, 1] = cand[:, 1].clamp(sizes[:, 1] / 2, float(benchmark.canvas_height) - sizes[:, 1] / 2)
                        cand_cost = evaluator.evaluate_batch(cand)
                        if not bool(cand_cost.is_legal.item()):
                            continue
                        cand_score = cand_cost.official_proxy.view(())
                        if bool(cand_score < best_score):
                            best = cand
                            best_score = cand_score.clone()
                            improved = True
                            break
                    if improved:
                        break
        return [best.detach().cpu()]

    def _rank_official(self, candidates: list[torch.Tensor], benchmark) -> list[tuple[float, torch.Tensor]]:
        plc = load_plc_for_benchmark(benchmark.name)
        if plc is None:
            evaluator = TorchProxyCostEvaluator(build_benchmark_context(benchmark, self.device))
            stacked = torch.stack([c.to(self.device, dtype=torch.float32) for c in candidates])
            costs = evaluator.evaluate_batch(stacked)
            rows = [
                (float(costs.official_proxy[i].item()) if bool(costs.is_legal[i].item()) else float("inf"), candidates[i])
                for i in range(len(candidates))
            ]
            return sorted(rows, key=lambda item: item[0])

        rows: list[tuple[float, torch.Tensor]] = []
        for pos in candidates:
            costs = compute_proxy_cost(pos, benchmark, plc)
            score = float(costs["proxy_cost"]) if int(costs["overlap_count"]) == 0 else float("inf")
            rows.append((score, pos))
        return sorted(rows, key=lambda item: item[0])

    def _unique_candidates(self, candidates: list[torch.Tensor]) -> list[torch.Tensor]:
        out: list[torch.Tensor] = []
        seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
        for pos in candidates:
            rounded = torch.round(pos.detach().cpu() * 1000).to(torch.int64)
            key = (tuple(int(x) for x in rounded[:, 0].tolist()), tuple(int(y) for y in rounded[:, 1].tolist()))
            if key in seen:
                continue
            seen.add(key)
            out.append(pos)
        return out


class DreamPlaceAnalyticalOptimizer:
    def __init__(self, config: DreamPlaceConfig, device: torch.device, ctx):
        self.config = config
        self.device = device
        self.ctx = ctx

    def generate_candidates(self, benchmark) -> list[torch.Tensor]:
        base = clamp_placement(legalize_initial(benchmark), benchmark).to(self.device, dtype=torch.float32)
        candidates = [base.detach().cpu()]

        recipes = (
            (0.12, 0.018, 0.030, 0.78),
            (0.18, 0.025, 0.035, 0.82),
            (0.28, 0.035, 0.030, 0.88),
        )
        for density_weight, gamma_scale, lr_scale, target_density in recipes:
            pos = self._run_one(
                benchmark,
                base,
                density_weight=density_weight,
                gamma_scale=gamma_scale,
                lr_scale=lr_scale,
                target_density=target_density,
            )
            candidates.append(pos.detach().cpu())
        return candidates

    def _run_one(
        self,
        benchmark,
        start: torch.Tensor,
        density_weight: float,
        gamma_scale: float,
        lr_scale: float,
        target_density: float,
    ) -> torch.Tensor:
        sizes = benchmark.macro_sizes.to(self.device, dtype=torch.float32)
        original = benchmark.macro_positions.to(self.device, dtype=torch.float32)
        fixed = benchmark.macro_fixed.to(self.device)
        movable = ~fixed
        if not self.config.optimize_soft_macros:
            movable = movable & benchmark.get_hard_macro_mask().to(self.device)

        x = start.clone().detach().to(self.device)
        x.requires_grad_(True)
        opt = torch.optim.Adam([x], lr=lr_scale * max(float(benchmark.canvas_width), float(benchmark.canvas_height)))
        best = x.detach().clone()
        best_loss = float("inf")

        for step in range(max(int(self.config.analytical_iters), 1)):
            opt.zero_grad(set_to_none=True)
            frac = step / max(float(self.config.analytical_iters - 1), 1.0)
            gamma = gamma_scale * (1.0 - 0.65 * frac) * max(float(benchmark.canvas_width), float(benchmark.canvas_height))
            gamma = max(gamma, 1.0e-3)
            wl = self._weighted_average_wirelength(x, benchmark, gamma)
            density = self._density_loss(x, benchmark, sizes, target_density)
            overlap = self._overlap_loss(x, benchmark, sizes)
            boundary = self._boundary_loss(x, benchmark, sizes)
            loss = wl + density_weight * density + self.config.overlap_weight * overlap + self.config.boundary_weight * boundary
            loss.backward()
            with torch.no_grad():
                if x.grad is not None:
                    x.grad *= movable.view(-1, 1).to(x.grad.dtype)
                    precond = (sizes[:, 0] * sizes[:, 1]).sqrt().clamp_min(1.0).view(-1, 1)
                    x.grad /= precond
            opt.step()
            with torch.no_grad():
                self._project_(x, benchmark, sizes, fixed, original)
                loss_value = float(loss.detach().item())
                if loss_value < best_loss:
                    best_loss = loss_value
                    best = x.detach().clone()
        return best

    def _pin_positions(self, placement: torch.Tensor):
        ctx = self.ctx
        if ctx.net_pin_parent.numel() == 0:
            return torch.zeros((0, 2), dtype=torch.float32, device=self.device)
        gathered = placement[ctx.net_pin_parent.clamp_min(0)] + ctx.net_pin_offset
        return torch.where(ctx.net_pin_is_port.view(-1, 1), ctx.net_pin_port_pos, gathered)

    def _weighted_average_wirelength(self, placement: torch.Tensor, benchmark, gamma: float) -> torch.Tensor:
        ctx = self.ctx
        if ctx.net_pin_parent.numel() == 0 or ctx.num_nets == 0:
            return placement.new_tensor(0.0)
        pin_pos = self._pin_positions(placement)
        net_ids = ctx.net_pin_net_id
        x = pin_pos[:, 0]
        y = pin_pos[:, 1]
        wx = self._wa_axis(x, net_ids, ctx.num_nets, gamma)
        wy = self._wa_axis(y, net_ids, ctx.num_nets, gamma)
        hpwl = wx + wy
        denom = (float(benchmark.canvas_width) + float(benchmark.canvas_height)) * max(float(ctx.wirelength_norm_net_count), 1.0)
        return (hpwl * ctx.net_weights).sum() / max(denom, 1.0e-9)

    def _wa_axis(self, coord: torch.Tensor, net_ids: torch.Tensor, num_nets: int, gamma: float) -> torch.Tensor:
        scaled = coord / gamma
        max_pos = torch.full((num_nets,), -torch.inf, dtype=coord.dtype, device=self.device)
        max_neg = torch.full((num_nets,), -torch.inf, dtype=coord.dtype, device=self.device)
        max_pos.scatter_reduce_(0, net_ids, scaled, reduce="amax", include_self=True)
        max_neg.scatter_reduce_(0, net_ids, -scaled, reduce="amax", include_self=True)
        ep = torch.exp((scaled - max_pos[net_ids]).clamp(min=-60.0, max=60.0))
        en = torch.exp((-scaled - max_neg[net_ids]).clamp(min=-60.0, max=60.0))
        sump = torch.zeros((num_nets,), dtype=coord.dtype, device=self.device).scatter_add_(0, net_ids, ep)
        sumn = torch.zeros((num_nets,), dtype=coord.dtype, device=self.device).scatter_add_(0, net_ids, en)
        xep = torch.zeros((num_nets,), dtype=coord.dtype, device=self.device).scatter_add_(0, net_ids, coord * ep)
        xen = torch.zeros((num_nets,), dtype=coord.dtype, device=self.device).scatter_add_(0, net_ids, coord * en)
        return xep / sump.clamp_min(1.0e-12) - xen / sumn.clamp_min(1.0e-12)

    def _density_loss(self, placement: torch.Tensor, benchmark, sizes: torch.Tensor, target_density: float) -> torch.Tensor:
        rows = max(1, min(int(benchmark.grid_rows), int(self.config.bin_grid_cap)))
        cols = max(1, min(int(benchmark.grid_cols), int(self.config.bin_grid_cap)))
        if rows * cols <= 1:
            return placement.new_tensor(0.0)
        grid = self._grid_boxes(benchmark, rows, cols, placement.dtype)
        boxes = self._macro_boxes(placement, sizes)
        x_overlap = (
            torch.minimum(boxes[:, 2].unsqueeze(1), grid[:, 2].unsqueeze(0))
            - torch.maximum(boxes[:, 0].unsqueeze(1), grid[:, 0].unsqueeze(0))
        ).clamp_min(0)
        y_overlap = (
            torch.minimum(boxes[:, 3].unsqueeze(1), grid[:, 3].unsqueeze(0))
            - torch.maximum(boxes[:, 1].unsqueeze(1), grid[:, 1].unsqueeze(0))
        ).clamp_min(0)
        bin_area = (float(benchmark.canvas_width) / cols) * (float(benchmark.canvas_height) / rows)
        density = (x_overlap * y_overlap).sum(dim=0) / max(bin_area, 1.0e-9)
        overflow = (density - target_density).clamp_min(0)
        k = max(int(overflow.numel() * 0.10), 1)
        return torch.topk(overflow.square(), k=k).values.mean()

    def _overlap_loss(self, placement: torch.Tensor, benchmark, sizes: torch.Tensor) -> torch.Tensor:
        n = int(benchmark.num_hard_macros)
        if n <= 1:
            return placement.new_tensor(0.0)
        pos = placement[:n]
        s = sizes[:n]
        dx = ((s[:, 0].unsqueeze(1) + s[:, 0].unsqueeze(0)) / 2 - (pos[:, 0].unsqueeze(1) - pos[:, 0].unsqueeze(0)).abs()).clamp_min(0)
        dy = ((s[:, 1].unsqueeze(1) + s[:, 1].unsqueeze(0)) / 2 - (pos[:, 1].unsqueeze(1) - pos[:, 1].unsqueeze(0)).abs()).clamp_min(0)
        area = dx * dy
        tri = torch.triu(torch.ones((n, n), dtype=torch.bool, device=self.device), diagonal=1)
        norm = max(float(benchmark.canvas_width) * float(benchmark.canvas_height), 1.0e-9)
        return area[tri].sum() / norm

    def _boundary_loss(self, placement: torch.Tensor, benchmark, sizes: torch.Tensor) -> torch.Tensor:
        half = sizes / 2
        left = (half[:, 0] - placement[:, 0]).clamp_min(0)
        right = (placement[:, 0] + half[:, 0] - float(benchmark.canvas_width)).clamp_min(0)
        bottom = (half[:, 1] - placement[:, 1]).clamp_min(0)
        top = (placement[:, 1] + half[:, 1] - float(benchmark.canvas_height)).clamp_min(0)
        return (left + right + bottom + top).sum() / max(float(benchmark.canvas_width) + float(benchmark.canvas_height), 1.0e-9)

    def _macro_boxes(self, placement: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
        half = sizes / 2
        return torch.stack(
            [placement[:, 0] - half[:, 0], placement[:, 1] - half[:, 1], placement[:, 0] + half[:, 0], placement[:, 1] + half[:, 1]],
            dim=1,
        )

    def _grid_boxes(self, benchmark, rows: int, cols: int, dtype: torch.dtype) -> torch.Tensor:
        xs = torch.linspace(0, float(benchmark.canvas_width), cols + 1, device=self.device, dtype=dtype)
        ys = torch.linspace(0, float(benchmark.canvas_height), rows + 1, device=self.device, dtype=dtype)
        boxes = []
        for r in range(rows):
            for c in range(cols):
                boxes.append(torch.stack([xs[c], ys[r], xs[c + 1], ys[r + 1]]))
        return torch.stack(boxes, dim=0)

    def _project_(self, placement: torch.Tensor, benchmark, sizes: torch.Tensor, fixed: torch.Tensor, original: torch.Tensor) -> None:
        placement[:, 0].clamp_(sizes[:, 0] / 2, float(benchmark.canvas_width) - sizes[:, 0] / 2)
        placement[:, 1].clamp_(sizes[:, 1] / 2, float(benchmark.canvas_height) - sizes[:, 1] / 2)
        if bool(fixed.any()):
            placement[fixed] = original[fixed]


class AnalyticalSARefiner:
    def __init__(self, config: DreamPlaceConfig, device: torch.device, ctx):
        self.config = config
        self.device = device
        self.ctx = ctx

    def refine(self, benchmark, starts: list[torch.Tensor]) -> list[torch.Tensor]:
        evaluator = TorchProxyCostEvaluator(self.ctx, overlap_weight=1000.0, boundary_weight=1000.0)
        seeds = tuple(int(s) for s in self.config.seeds)
        state = []
        for i, seed in enumerate(seeds):
            base = starts[i % len(starts)].to(self.device, dtype=torch.float32)
            state.append(base)
        state_t = torch.stack(state, dim=0)
        self._restore_fixed(state_t, benchmark)
        current = evaluator.evaluate_batch(state_t)
        best_pos = state_t.clone()
        best_cost = torch.where(current.is_legal, current.official_proxy, torch.full_like(current.official_proxy, torch.inf))
        top_pos, top_score = self._add_top(None, None, state_t, current.search_score)

        for step in range(max(int(self.config.refine_iters), 0)):
            temp = self._temperature(step)
            candidates = self._generate_candidates(state_t, benchmark, seeds, step, temp)
            flat = candidates.reshape(-1, benchmark.num_macros, 2)
            cand_cost = evaluator.evaluate_batch(flat)
            top_pos, top_score = self._add_top(top_pos, top_score, flat, cand_cost.search_score)
            c = int(self.config.refine_candidate_batch)
            score = cand_cost.search_score.reshape(len(seeds), c)
            official = cand_cost.official_proxy.reshape(len(seeds), c)
            legal = cand_cost.is_legal.reshape(len(seeds), c)
            delta = score - current.search_score.view(-1, 1)
            accept = (delta <= 0) | (torch.rand_like(delta) < torch.exp((-delta / max(temp, 1.0e-9)).clamp(max=60.0)))
            masked_score = torch.where(accept, score, torch.full_like(score, torch.inf))
            chosen_score, chosen = masked_score.min(dim=1)
            has_accept = torch.isfinite(chosen_score)
            if bool(has_accept.any()):
                row = torch.arange(len(seeds), device=self.device)
                chosen_flat = row * c + chosen
                new_state = flat[chosen_flat]
                state_t = torch.where(has_accept.view(-1, 1, 1), new_state, state_t)
                current = evaluator.evaluate_batch(state_t)
            legal_better = legal & (official < best_cost.view(-1, 1))
            if bool(legal_better.any()):
                masked = torch.where(legal_better, official, torch.full_like(official, torch.inf))
                _, best_idx = masked.min(dim=1)
                row = torch.arange(len(seeds), device=self.device)
                any_better = legal_better.any(dim=1)
                best_flat = row * c + best_idx
                best_pos = torch.where(any_better.view(-1, 1, 1), flat[best_flat], best_pos)
                best_cost = torch.where(any_better, official[row, best_idx], best_cost)

        finals = []
        for pos in best_pos.detach().cpu():
            finals.append(clamp_placement(legalize_placement(pos, benchmark, gap=0.01), benchmark))
        if top_pos is not None:
            keep = min(int(top_pos.shape[0]), self.config.top_k_candidates)
            for pos in top_pos[:keep].detach().cpu():
                finals.append(clamp_placement(legalize_placement(pos, benchmark, gap=0.01), benchmark))
        return finals

    def _temperature(self, step: int) -> float:
        start = 0.045
        end = 0.0008
        if self.config.refine_iters <= 1:
            return end
        frac = step / float(self.config.refine_iters - 1)
        return start * ((end / start) ** frac)

    def _generate_candidates(self, state: torch.Tensor, benchmark, seeds, step: int, temp: float) -> torch.Tensor:
        s, n, _ = state.shape
        c = int(self.config.refine_candidate_batch)
        candidates = state.unsqueeze(1).repeat(1, c, 1, 1).reshape(s * c, n, 2)
        sizes = benchmark.macro_sizes.to(self.device, dtype=torch.float32)
        movable = (benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()).to(self.device)
        movable_idx = torch.where(movable)[0]
        if movable_idx.numel() == 0:
            return candidates.reshape(s, c, n, 2)
        total = s * c
        generator = torch.Generator(device=self.device)
        generator.manual_seed((sum(int(x) for x in seeds) + 1000003 * (step + 1)) % (2**63 - 1))
        move_types = torch.randint(0, 5, (total,), device=self.device, generator=generator)
        chosen_i = movable_idx[torch.randint(0, movable_idx.numel(), (total,), device=self.device, generator=generator)]
        chosen_j = movable_idx[torch.randint(0, movable_idx.numel(), (total,), device=self.device, generator=generator)]
        scale = max(float(benchmark.canvas_width), float(benchmark.canvas_height)) * max(temp, 1.0e-4)
        noise = torch.randn((total, 2), device=self.device, generator=generator) * scale
        for row in range(total):
            i = int(chosen_i[row].item())
            j = int(chosen_j[row].item())
            move = int(move_types[row].item())
            if move == 0:
                candidates[row, i] += noise[row]
            elif move == 1 and i != j:
                old_i = candidates[row, i].clone()
                candidates[row, i] = candidates[row, j]
                candidates[row, j] = old_i
            elif move == 2 and i != j:
                alpha = 0.08 + 0.30 * torch.rand((), device=self.device, generator=generator)
                candidates[row, i] = candidates[row, i] + alpha * (candidates[row, j] - candidates[row, i])
            elif move == 3:
                candidates[row, i, 0] = torch.rand((), device=self.device, generator=generator) * float(benchmark.canvas_width)
                candidates[row, i, 1] = torch.rand((), device=self.device, generator=generator) * float(benchmark.canvas_height)
            else:
                k = min(5, int(movable_idx.numel()))
                perm_src = movable_idx[torch.randperm(movable_idx.numel(), device=self.device, generator=generator)[:k]]
                perm_dst = perm_src[torch.randperm(k, device=self.device, generator=generator)]
                candidates[row, perm_src] = candidates[row, perm_dst].clone()
        half = sizes / 2
        candidates[:, :, 0] = candidates[:, :, 0].clamp(half[:, 0], float(benchmark.canvas_width) - half[:, 0])
        candidates[:, :, 1] = candidates[:, :, 1].clamp(half[:, 1], float(benchmark.canvas_height) - half[:, 1])
        self._restore_fixed(candidates, benchmark)
        return candidates.reshape(s, c, n, 2)

    def _restore_fixed(self, placements: torch.Tensor, benchmark) -> None:
        fixed = benchmark.macro_fixed.to(self.device)
        if bool(fixed.any()):
            orig = benchmark.macro_positions.to(self.device, dtype=placements.dtype)
            placements[:, fixed, :] = orig[fixed]

    def _add_top(self, top_pos, top_score, candidates: torch.Tensor, scores: torch.Tensor):
        k = max(int(self.config.top_k_candidates), 0)
        if k == 0:
            return top_pos, top_score
        pos = candidates.reshape(-1, candidates.shape[-2], candidates.shape[-1]).detach()
        score = scores.reshape(-1).detach()
        finite = torch.isfinite(score)
        if not bool(finite.any()):
            return top_pos, top_score
        pos = pos[finite]
        score = score[finite]
        if top_pos is not None and top_score is not None:
            pos = torch.cat([top_pos, pos], dim=0)
            score = torch.cat([top_score, score], dim=0)
        keep = min(k, int(score.numel()))
        _, idx = torch.topk(score, k=keep, largest=False)
        return pos[idx].clone(), score[idx].clone()
