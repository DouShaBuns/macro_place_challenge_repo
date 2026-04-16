from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from benchmark_context import build_benchmark_context
from legalize import clamp_placement, legalize_placement
from torch_objective import TorchProxyCostEvaluator
from warm_start import build_warm_start_provider, prepare_warm_start


@dataclass
class SAConfig:
    seeds: tuple[int, ...] = (42, 43, 44, 45)
    iters: int = 80
    candidate_batch: int = 16
    t_start_scale: float = 0.05
    t_end_scale: float = 0.001
    overlap_weight: float = 1000.0
    boundary_weight: float = 1000.0
    top_k_final_candidates: int = 32
    local_refine_max_trials: int = 1000
    search_congestion_mode: str = "fast"


class SAOptimizer:
    def __init__(
        self,
        config: SAConfig | None = None,
        device: str | torch.device | None = None,
        warm_start=None,
        warm_start_path: str | None = None,
    ):
        self.config = config or SAConfig()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.warm_start_provider = build_warm_start_provider(warm_start, warm_start_path)

    def optimize(self, benchmark) -> torch.Tensor:
        ctx = build_benchmark_context(benchmark, self.device)
        evaluator = TorchProxyCostEvaluator(
            ctx,
            overlap_weight=self.config.overlap_weight,
            boundary_weight=self.config.boundary_weight,
            congestion_mode=self.config.search_congestion_mode,
        )
        exact_evaluator = TorchProxyCostEvaluator(
            ctx,
            overlap_weight=self.config.overlap_weight,
            boundary_weight=self.config.boundary_weight,
            congestion_mode="exact",
        )
        init = prepare_warm_start(benchmark, self.warm_start_provider).to(self.device, dtype=torch.float32)
        init = clamp_placement(init, benchmark).to(self.device)
        seeds = tuple(int(s) for s in self.config.seeds)
        state = init.unsqueeze(0).repeat(len(seeds), 1, 1)
        self._restore_fixed(state, benchmark)
        current = evaluator.evaluate_batch(state)
        best_pos = state.clone()
        best_cost = torch.where(
            current.is_legal,
            current.official_proxy,
            torch.full_like(current.official_proxy, torch.inf),
        )
        top_pos, top_score = self._add_top_candidates(None, None, state, current.search_score)

        for step in range(max(int(self.config.iters), 0)):
            temp = self._temperature(step)
            candidates = self._generate_candidates(state, benchmark, seeds, step, temp)
            flat = candidates.reshape(-1, benchmark.num_macros, 2)
            cand_cost = evaluator.evaluate_batch(flat)
            top_pos, top_score = self._add_top_candidates(top_pos, top_score, flat, cand_cost.search_score)
            c = int(self.config.candidate_batch)
            score = cand_cost.search_score.reshape(len(seeds), c)
            official = cand_cost.official_proxy.reshape(len(seeds), c)
            legal = cand_cost.is_legal.reshape(len(seeds), c)
            delta = score - current.search_score.view(-1, 1)
            rand = torch.rand_like(delta)
            accept = (delta <= 0) | (rand < torch.exp((-delta / max(temp, 1.0e-9)).clamp(max=60.0)))
            masked_score = torch.where(accept, score, torch.full_like(score, torch.inf))
            chosen_score, chosen = masked_score.min(dim=1)
            has_accept = torch.isfinite(chosen_score)
            if bool(has_accept.any()):
                row = torch.arange(len(seeds), device=self.device)
                chosen_flat = row * c + chosen
                new_state = flat[chosen_flat]
                state = torch.where(has_accept.view(-1, 1, 1), new_state, state)
                current = evaluator.evaluate_batch(state)
                top_pos, top_score = self._add_top_candidates(top_pos, top_score, state, current.search_score)

            legal_better = legal & (official < best_cost.view(-1, 1))
            any_better = legal_better.any(dim=1)
            if bool(any_better.any()):
                masked_official = torch.where(legal_better, official, torch.full_like(official, torch.inf))
                _, best_idx = masked_official.min(dim=1)
                row = torch.arange(len(seeds), device=self.device)
                best_flat = row * c + best_idx
                best_pos = torch.where(any_better.view(-1, 1, 1), flat[best_flat], best_pos)
                best_cost = torch.where(any_better, official[row, best_idx], best_cost)

            state_legal_better = current.is_legal & (current.official_proxy < best_cost)
            if bool(state_legal_better.any()):
                best_pos = torch.where(state_legal_better.view(-1, 1, 1), state, best_pos)
                best_cost = torch.where(state_legal_better, current.official_proxy, best_cost)

        if torch.isfinite(best_cost).any():
            best_seed = int(torch.argmin(best_cost).item())
            result = best_pos[best_seed].detach().cpu()
        else:
            result = init.detach().cpu()
        final = self._select_legalized_final(benchmark, exact_evaluator, top_pos, best_pos, result)
        refined = self._local_refine(final, benchmark, evaluator)
        final = self._select_legalized_final(
            benchmark,
            exact_evaluator,
            None,
            torch.stack([final, refined], dim=0),
            final,
        )
        return clamp_placement(final, benchmark).cpu()

    def _temperature(self, step: int) -> float:
        t_start = self.config.t_start_scale
        t_end = self.config.t_end_scale
        if self.config.iters <= 1:
            return t_end
        frac = step / float(self.config.iters - 1)
        return t_start * ((t_end / t_start) ** frac)

    def _generate_candidates(
        self, state: torch.Tensor, benchmark, seeds: Sequence[int], step: int, temp: float
    ) -> torch.Tensor:
        s, n, _ = state.shape
        c = int(self.config.candidate_batch)
        candidates = state.unsqueeze(1).repeat(1, c, 1, 1).reshape(s * c, n, 2)
        sizes = benchmark.macro_sizes.to(self.device, dtype=torch.float32)
        movable = (benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()).to(self.device)
        movable_idx = torch.where(movable)[0]
        if movable_idx.numel() == 0:
            return candidates.reshape(s, c, n, 2)
        total = s * c
        generator = torch.Generator(device=self.device)
        generator.manual_seed((sum(int(s) for s in seeds) + 1000003 * (step + 1)) % (2**63 - 1))
        move_types = torch.randint(0, 4, (total,), device=self.device, generator=generator)
        chosen_i = movable_idx[
            torch.randint(0, movable_idx.numel(), (total,), device=self.device, generator=generator)
        ]
        chosen_j = movable_idx[
            torch.randint(0, movable_idx.numel(), (total,), device=self.device, generator=generator)
        ]
        shift_scale = max(float(benchmark.canvas_width), float(benchmark.canvas_height)) * max(temp, 1.0e-4)
        noise = torch.randn((total, 2), device=self.device, generator=generator) * shift_scale
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
                alpha = 0.10 + 0.25 * torch.rand((), device=self.device, generator=generator)
                candidates[row, i] = candidates[row, i] + alpha * (candidates[row, j] - candidates[row, i])
            else:
                k = min(4, int(movable_idx.numel()))
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

    def _add_top_candidates(
        self,
        top_pos: torch.Tensor | None,
        top_score: torch.Tensor | None,
        candidates: torch.Tensor,
        scores: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        k = max(int(self.config.top_k_final_candidates), 0)
        if k == 0 or candidates.numel() == 0:
            return top_pos, top_score
        flat_pos = candidates.reshape(-1, candidates.shape[-2], candidates.shape[-1]).detach()
        flat_score = scores.reshape(-1).detach()
        finite = torch.isfinite(flat_score)
        if not bool(finite.any()):
            return top_pos, top_score
        flat_pos = flat_pos[finite]
        flat_score = flat_score[finite]
        if top_pos is not None and top_score is not None:
            flat_pos = torch.cat([top_pos, flat_pos], dim=0)
            flat_score = torch.cat([top_score, flat_score], dim=0)
        keep = min(k, int(flat_score.numel()))
        _, idx = torch.topk(flat_score, k=keep, largest=False)
        return flat_pos[idx].clone(), flat_score[idx].clone()

    def _select_legalized_final(
        self,
        benchmark,
        evaluator: TorchProxyCostEvaluator,
        top_pos: torch.Tensor | None,
        best_pos: torch.Tensor,
        fallback: torch.Tensor,
    ) -> torch.Tensor:
        candidates = []
        if top_pos is not None:
            candidates.extend(pos.detach().cpu() for pos in top_pos)
        candidates.extend(pos.detach().cpu() for pos in best_pos)
        candidates.append(fallback.detach().cpu())

        legalized = [clamp_placement(legalize_placement(pos, benchmark), benchmark) for pos in candidates]
        stacked = torch.stack(legalized, dim=0).to(self.device, dtype=torch.float32)
        costs = evaluator.evaluate_batch(stacked)
        legal_cost = torch.where(
            costs.is_legal,
            costs.official_proxy,
            torch.full_like(costs.official_proxy, torch.inf),
        )
        if torch.isfinite(legal_cost).any():
            best_idx = int(torch.argmin(legal_cost).item())
            return stacked[best_idx].detach().cpu()
        return clamp_placement(legalize_placement(fallback.detach().cpu(), benchmark), benchmark).cpu()

    def _local_refine(
        self,
        placement: torch.Tensor,
        benchmark,
        evaluator: TorchProxyCostEvaluator,
    ) -> torch.Tensor:
        max_trials = max(int(self.config.local_refine_max_trials), 0)
        if max_trials == 0:
            return placement.cpu()

        current = clamp_placement(placement, benchmark).to(self.device, dtype=torch.float32)
        current_cost = evaluator.evaluate_batch(current)
        if not bool(current_cost.is_legal.item()):
            return current.detach().cpu()
        best_cost = current_cost.official_proxy.view(()).clone()
        movable = (benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()).to(self.device)
        movable_idx = torch.where(movable)[0].tolist()
        if not movable_idx:
            return current.detach().cpu()

        grid_step = max(min(evaluator.grid_w, evaluator.grid_h), 1.0e-6)
        steps = [2.0 * grid_step, 1.0 * grid_step, 0.5 * grid_step, 0.25 * grid_step]
        directions = (
            (1.0, 0.0),
            (-1.0, 0.0),
            (0.0, 1.0),
            (0.0, -1.0),
        )
        trials = 0
        for step in steps:
            for macro_idx in movable_idx:
                for dx, dy in directions:
                    if trials >= max_trials:
                        return current.detach().cpu()
                    trials += 1
                    candidate = current.clone()
                    candidate[macro_idx, 0] += float(dx) * step
                    candidate[macro_idx, 1] += float(dy) * step
                    candidate = clamp_placement(candidate, benchmark).to(self.device, dtype=torch.float32)
                    cand_cost = evaluator.evaluate_batch(candidate)
                    if not bool(cand_cost.is_legal.item()):
                        continue
                    cand_proxy = cand_cost.official_proxy.view(())
                    if bool(cand_proxy < best_cost):
                        current = candidate
                        best_cost = cand_proxy.clone()
        return current.detach().cpu()
