from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_SA_GPU = _HERE.parent / "sa_gpu"
if str(_SA_GPU) not in sys.path:
    sys.path.insert(0, str(_SA_GPU))

from benchmark_context import build_benchmark_context, load_plc_for_benchmark  # noqa: E402
from legalize import clamp_placement, legalize_initial, legalize_placement  # noqa: E402
from trace_utils import PlacementTraceRecorder  # noqa: E402
from torch_objective import TorchProxyCostEvaluator  # noqa: E402

from macro_place.objective import compute_overlap_metrics, compute_proxy_cost  # noqa: E402


@dataclass
class DreamPlaceConfig:
    analytical_iters: int = 260
    refine_iters: int = 80
    refine_candidate_batch: int = 16
    seeds: tuple[int, ...] = (42, 43, 44, 45)
    density_weight: float = 0.18
    congestion_weight: float = 0.05
    congestion_target: float = 0.85
    congestion_density_alpha: float = 0.15
    congestion_map_update_interval: int = 20
    soft_route_congestion_weight: float = 0.02
    soft_route_tau_scale: float = 0.5
    soft_route_chunk_size: int = 512
    overlap_weight: float = 18.0
    boundary_weight: float = 25.0
    target_density: float = 0.82
    gamma_scale: float = 0.025
    learning_rate: float = 0.035
    bin_grid_cap: int = 48
    optimize_soft_macros: bool = True
    run_refine: bool = False
    top_k_candidates: int = 8
    official_rerank_limit: int = 0
    max_gpu_batch_candidates: int = 0
    local_refine_trials: int = 0
    official_refine_evals: int = 24
    official_refine_macro_limit: int = 1000
    official_refine_rounds: int = 1
    official_refine_prefilter_chunk: int = 64
    official_refine_step_scales: tuple[float, ...] = (0.25, 0.5, 1.0)
    analytical_snapshot_interval: int = 20
    analytical_snapshots_per_recipe: int = 3
    soft_relax_iters: int = 200
    soft_relax_lr_scale: float = 0.01
    soft_relax_lr_scales: tuple[float, ...] = (0.005, 0.01)
    soft_relax_start_k: int = 1
    soft_relax_snapshot_interval: int = 20
    soft_relax_snapshots: int = 8
    adaptive_large_budget: bool = True
    log_proxy_calibration: bool = False
    recipes: tuple[tuple[float, float, float, float], ...] = (
        (0.12, 0.018, 0.030, 0.78),
        (0.18, 0.025, 0.035, 0.82),
        (0.28, 0.035, 0.030, 0.88),
    )


class DreamPlaceHybridOptimizer:
    def __init__(self, config: DreamPlaceConfig | None = None, device: str | torch.device | None = None):
        self.config = config or DreamPlaceConfig()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    def optimize(self, benchmark) -> torch.Tensor:
        trace = PlacementTraceRecorder.from_env("dreamplace_gpu", benchmark)
        ctx = build_benchmark_context(benchmark, self.device)
        analytical = DreamPlaceAnalyticalOptimizer(self.config, self.device, ctx)
        candidates = analytical.generate_candidates(benchmark, trace=trace)

        legalized = [self._repair_candidate(pos.cpu(), benchmark) for pos in candidates]
        legalized.append(self._repair_candidate(benchmark.macro_positions, benchmark))
        legalized = self._unique_candidates(legalized)

        ranked = self._rank_official(legalized, benchmark, ctx)
        starts = [pos for _, pos in ranked[: max(1, min(self.config.top_k_candidates, len(ranked)))]]
        if trace is not None and starts:
            trace.record(starts[0], "legalized_best")
        if starts and self.config.log_proxy_calibration:
            self._log_proxy_calibration(starts[0], benchmark, ctx)
        local = self._local_refine_best(starts, benchmark, ctx) if self.config.local_refine_trials > 0 else []
        if local:
            reranked = self._rank_official(starts + local, benchmark, ctx)
            starts = [pos for _, pos in reranked[: max(1, min(self.config.top_k_candidates, len(reranked)))]]
        soft_relaxed = []
        if starts and self.config.soft_relax_iters > 0:
            for start_pos in starts[: max(1, min(int(self.config.soft_relax_start_k), len(starts)))]:
                soft_relaxed.extend(self._soft_relax_candidates(start_pos, benchmark, ctx))
        if soft_relaxed:
            reranked = self._rank_official(starts + soft_relaxed, benchmark, ctx)
            starts = [pos for _, pos in reranked[: max(1, min(self.config.top_k_candidates, len(reranked)))]]
        official_local = self._official_refine_best(starts[0], benchmark, ctx) if starts and self.config.official_refine_evals > 0 else None
        if official_local is not None:
            reranked = self._rank_official(starts + [official_local], benchmark, ctx)
            starts = [pos for _, pos in reranked[: max(1, min(self.config.top_k_candidates, len(reranked)))]]
        if not self.config.run_refine or self.config.refine_iters <= 0:
            result = starts[0].cpu()
            if trace is not None:
                trace.record(result, "final")
                trace.close()
            return result

        refiner = AnalyticalSARefiner(self.config, self.device, ctx)
        refined = refiner.refine(benchmark, starts)
        all_final = starts + refined
        ranked_final = self._rank_official(all_final, benchmark, ctx)
        result = ranked_final[0][1].cpu()
        if trace is not None:
            trace.record(result, "final")
            trace.close()
        return result

    def _local_refine_best(self, candidates: list[torch.Tensor], benchmark, ctx) -> list[torch.Tensor]:
        ranked = self._rank_official(candidates, benchmark, ctx)
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
        directions = (
            (1.0, 0.0),
            (-1.0, 0.0),
            (0.0, 1.0),
            (0.0, -1.0),
            (1.0, 1.0),
            (1.0, -1.0),
            (-1.0, 1.0),
            (-1.0, -1.0),
        )
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

    def _repair_candidate(self, placement: torch.Tensor, benchmark) -> torch.Tensor:
        clamped = clamp_placement(placement, benchmark)
        if compute_overlap_metrics(clamped, benchmark)["overlap_count"] == 0:
            return clamped
        return clamp_placement(legalize_placement(clamped, benchmark, gap=0.001), benchmark)

    def _soft_relax_candidates(self, start: torch.Tensor, benchmark, ctx) -> list[torch.Tensor]:
        if int(benchmark.num_soft_macros) <= 0:
            return []
        plc = self._plc_for_benchmark(benchmark, ctx)
        if plc is None:
            return []
        baseline = compute_proxy_cost(start, benchmark, plc)
        if int(baseline["overlap_count"]) != 0:
            return []
        best_score = float(baseline["proxy_cost"])
        out: list[torch.Tensor] = []
        route_weight = self._soft_relax_route_weight(benchmark, baseline)
        scales = self._soft_relax_lr_scales(benchmark, baseline)
        seen: set[float] = set()
        for lr_scale in scales:
            key = round(float(lr_scale), 12)
            if key in seen:
                continue
            seen.add(key)
            out.extend(self._soft_relax_one(start, benchmark, ctx, plc, best_score, float(lr_scale), route_weight))
        return out

    def _soft_relax_one(
        self,
        start: torch.Tensor,
        benchmark,
        ctx,
        plc,
        best_score: float,
        lr_scale: float,
        route_weight: float,
    ) -> list[torch.Tensor]:

        x = start.detach().to(self.device, dtype=torch.float32).clone()
        original_hard = x[: int(benchmark.num_hard_macros)].detach().clone()
        sizes = benchmark.macro_sizes.to(self.device, dtype=torch.float32)
        soft_mask = torch.zeros(int(benchmark.num_macros), dtype=torch.bool, device=self.device)
        soft_mask[int(benchmark.num_hard_macros) :] = True
        fixed = benchmark.macro_fixed.to(self.device)
        movable = soft_mask & ~fixed
        if not bool(movable.any()):
            return []

        x.requires_grad_(True)
        lr = float(lr_scale) * max(float(benchmark.canvas_width), float(benchmark.canvas_height))
        opt = torch.optim.Adam([x], lr=lr)
        analytical = DreamPlaceAnalyticalOptimizer(self.config, self.device, ctx)
        best_loss = float("inf")
        snapshots: list[tuple[float, torch.Tensor]] = []
        rows, cols = analytical._bin_grid_shape(benchmark)

        soft_iters = self._large_case_soft_iters(benchmark)
        for step in range(max(int(soft_iters), 0)):
            opt.zero_grad(set_to_none=True)
            frac = step / max(float(soft_iters - 1), 1.0)
            gamma = max(float(self.config.gamma_scale) * (1.0 - 0.5 * frac), 1.0e-3) * max(
                float(benchmark.canvas_width), float(benchmark.canvas_height)
            )
            wl = analytical._weighted_average_wirelength(x, benchmark, gamma)
            density = analytical._density_loss(x, benchmark, sizes, self.config.target_density)
            soft_route_cong = analytical._soft_route_congestion_loss(x, benchmark, rows, cols)
            boundary = analytical._boundary_loss(x, benchmark, sizes)
            loss = wl + self.config.density_weight * density + route_weight * soft_route_cong + self.config.boundary_weight * boundary
            loss.backward()
            with torch.no_grad():
                if x.grad is not None:
                    x.grad *= movable.view(-1, 1).to(x.grad.dtype)
            opt.step()
            with torch.no_grad():
                x[: int(benchmark.num_hard_macros)] = original_hard
                x[:, 0].clamp_(sizes[:, 0] / 2, float(benchmark.canvas_width) - sizes[:, 0] / 2)
                x[:, 1].clamp_(sizes[:, 1] / 2, float(benchmark.canvas_height) - sizes[:, 1] / 2)
                loss_value = float(loss.detach().item())
                if loss_value < best_loss:
                    best_loss = loss_value
                    snapshots.append((loss_value, x.detach().cpu().clone()))
                interval = max(int(self.config.soft_relax_snapshot_interval), 0)
                if interval > 0 and (step % interval == 0 or step == soft_iters - 1):
                    snapshots.append((loss_value, x.detach().cpu().clone()))

        snapshots.sort(key=lambda item: item[0])
        out: list[torch.Tensor] = []
        seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
        snapshot_limit = self._large_case_soft_snapshot_limit(benchmark)
        for _, cand in snapshots[: max(1, int(snapshot_limit))]:
            rounded = torch.round(cand * 1000).to(torch.int64)
            key = (tuple(int(v) for v in rounded[:, 0].tolist()), tuple(int(v) for v in rounded[:, 1].tolist()))
            if key in seen:
                continue
            seen.add(key)
            official = compute_proxy_cost(cand, benchmark, plc)
            if int(official["overlap_count"]) == 0 and float(official["proxy_cost"]) < best_score:
                print(f"[dreamplace_gpu] soft_relax lr={lr_scale:.6g} proxy={float(official['proxy_cost']):.6f}")
                out.append(cand)
        return out

    def _soft_relax_lr_scales(self, benchmark, baseline: dict) -> tuple[float, ...]:
        scales = self.config.soft_relax_lr_scales or (float(self.config.soft_relax_lr_scale),)
        if not self.config.adaptive_large_budget:
            return scales
        n = int(benchmark.num_hard_macros)
        total = int(benchmark.num_macros)
        if n <= 260 and total <= 1200:
            return (0.003, min(float(scale) for scale in scales))
        congestion = float(baseline.get("congestion_cost", 0.0))
        if n >= 700 and congestion >= 2.3:
            return (min(float(scale) for scale in scales),)
        return scales

    def _soft_relax_route_weight(self, benchmark, baseline: dict) -> float:
        weight = float(self.config.soft_route_congestion_weight)
        if not self.config.adaptive_large_budget:
            return weight
        congestion = float(baseline.get("congestion_cost", 0.0))
        if int(benchmark.num_hard_macros) >= 700 and congestion >= 2.3:
            return max(weight, 0.04)
        return weight

    def _large_case_soft_iters(self, benchmark) -> int:
        iters = max(int(self.config.soft_relax_iters), 0)
        if not self.config.adaptive_large_budget:
            return iters
        n = int(benchmark.num_hard_macros)
        total = int(benchmark.num_macros)
        if n <= 260 and total <= 1200:
            return max(iters, 300)
        if n >= 700:
            return min(iters, 30)
        if n >= 580:
            return min(iters, 60)
        if n >= 380:
            if total <= 1600:
                return min(iters, 100)
            return min(iters, 40)
        return iters

    def _large_case_soft_snapshot_limit(self, benchmark) -> int:
        limit = max(int(self.config.soft_relax_snapshots), 1)
        if not self.config.adaptive_large_budget:
            return limit
        n = int(benchmark.num_hard_macros)
        total = int(benchmark.num_macros)
        if n >= 700:
            return min(limit, 1)
        if n >= 580:
            return min(limit, 2)
        if n >= 380:
            if total <= 1600:
                return min(limit, 4)
            return min(limit, 2)
        return limit

    def _large_case_official_refine_budget(self, benchmark) -> int:
        budget = max(int(self.config.official_refine_evals), 0)
        if not self.config.adaptive_large_budget:
            return budget
        n = int(benchmark.num_hard_macros)
        total = int(benchmark.num_macros)
        if n >= 700:
            return 0
        if n >= 580:
            return min(budget, 8)
        if n >= 380:
            if total <= 1600:
                return min(budget, 8)
            return min(budget, 8)
        return budget

    def _official_refine_macro_limit(self, benchmark) -> int:
        limit = max(1, int(self.config.official_refine_macro_limit))
        if not self.config.adaptive_large_budget:
            return limit
        n = int(benchmark.num_hard_macros)
        if n >= 700:
            return min(limit, 192)
        if n >= 580:
            return min(limit, 384)
        return limit

    def _official_refine_best(self, start: torch.Tensor, benchmark, ctx) -> torch.Tensor | None:
        plc = self._plc_for_benchmark(benchmark, ctx)
        if plc is None:
            return None

        best = start.detach().cpu()
        best_costs = compute_proxy_cost(best, benchmark, plc)
        if int(best_costs["overlap_count"]) != 0:
            return None
        best_score = float(best_costs["proxy_cost"])

        evaluator = TorchProxyCostEvaluator(ctx, overlap_weight=1000.0, boundary_weight=1000.0)
        sizes = benchmark.macro_sizes
        movable = torch.where(benchmark.get_movable_mask() & benchmark.get_hard_macro_mask())[0].tolist()
        if not movable:
            return None
        movable.sort(key=lambda i: -float(sizes[i, 0] * sizes[i, 1]))
        macro_limit = self._official_refine_macro_limit(benchmark)
        movable = movable[: min(macro_limit, len(movable))]

        step_base = max(
            float(benchmark.canvas_width) / max(int(benchmark.grid_cols), 1),
            float(benchmark.canvas_height) / max(int(benchmark.grid_rows), 1),
        )
        directions = (
            (1.0, 0.0),
            (-1.0, 0.0),
            (0.0, 1.0),
            (0.0, -1.0),
            (1.0, 1.0),
            (1.0, -1.0),
            (-1.0, 1.0),
            (-1.0, -1.0),
        )
        improved = False
        rounds = max(1, int(self.config.official_refine_rounds))
        official_budget = self._large_case_official_refine_budget(benchmark)

        best_device = best.to(self.device, dtype=torch.float32)

        for round_idx in range(rounds):
            if official_budget <= 0:
                break
            keep = official_budget
            scored: list[tuple[float, tuple[int, float, float, float]]] = []
            chunk_size = max(1, int(self.config.official_refine_prefilter_chunk))
            specs: list[tuple[int, float, float, float]] = []
            with torch.no_grad():
                for scale in self.config.official_refine_step_scales:
                    step = step_base * float(scale) * (0.5 ** round_idx)
                    for macro_idx in movable:
                        for direction_idx, (dx, dy) in enumerate(directions):
                            specs.append((macro_idx, step, float(dx), float(dy)))
                            if len(specs) < chunk_size:
                                continue
                            stacked = self._build_official_refine_chunk(best_device, specs, sizes, benchmark)
                            costs = evaluator.evaluate_batch(stacked)
                            search_score = torch.where(
                                costs.is_legal,
                                costs.search_score,
                                torch.full_like(costs.search_score, torch.inf),
                            )
                            for offset, score in enumerate(search_score.detach().cpu().tolist()):
                                if score != float("inf"):
                                    scored.append((float(score), specs[offset]))
                            specs = []
                if specs:
                    stacked = self._build_official_refine_chunk(best_device, specs, sizes, benchmark)
                    costs = evaluator.evaluate_batch(stacked)
                    search_score = torch.where(
                        costs.is_legal,
                        costs.search_score,
                        torch.full_like(costs.search_score, torch.inf),
                    )
                    for offset, score in enumerate(search_score.detach().cpu().tolist()):
                        if score != float("inf"):
                            scored.append((float(score), specs[offset]))
            if not scored:
                break
            scored.sort(key=lambda item: item[0])
            order = [spec for _, spec in scored[:keep]]

            round_best = best
            round_best_score = best_score
            for spec in order:
                cand = self._materialize_official_refine_candidate(best, spec, benchmark)
                official = compute_proxy_cost(cand, benchmark, plc)
                official_budget -= 1
                if int(official["overlap_count"]) != 0:
                    continue
                score = float(official["proxy_cost"])
                if score < round_best_score:
                    round_best = cand
                    round_best_score = score
                if official_budget <= 0:
                    break
            if round_best_score < best_score:
                best = round_best
                best_device = best.to(self.device, dtype=torch.float32)
                best_score = round_best_score
                improved = True
            else:
                break

        if improved:
            print(f"[dreamplace_gpu] official_refine proxy={best_score:.6f}")
            return best
        return None

    def _build_official_refine_chunk(
        self,
        base: torch.Tensor,
        specs: list[tuple[int, float, float, float]],
        sizes: torch.Tensor,
        benchmark,
    ) -> torch.Tensor:
        stacked = base.unsqueeze(0).repeat(len(specs), 1, 1)
        macro_idx = torch.tensor([item[0] for item in specs], dtype=torch.long, device=self.device)
        delta = torch.tensor(
            [[item[1] * item[2], item[1] * item[3]] for item in specs],
            dtype=torch.float32,
            device=self.device,
        )
        rows = torch.arange(len(specs), dtype=torch.long, device=self.device)
        stacked[rows, macro_idx] += delta
        sizes = sizes.to(self.device, dtype=torch.float32)
        stacked[:, :, 0].clamp_(sizes[:, 0] / 2, float(benchmark.canvas_width) - sizes[:, 0] / 2)
        stacked[:, :, 1].clamp_(sizes[:, 1] / 2, float(benchmark.canvas_height) - sizes[:, 1] / 2)
        return stacked

    def _materialize_official_refine_candidate(
        self,
        base: torch.Tensor,
        spec: tuple[int, float, float, float],
        benchmark,
    ) -> torch.Tensor:
        macro_idx, step, dx, dy = spec
        cand = base.clone()
        cand[macro_idx, 0] += dx * step
        cand[macro_idx, 1] += dy * step
        return clamp_placement(cand, benchmark)

    def _rank_official(self, candidates: list[torch.Tensor], benchmark, ctx=None) -> list[tuple[float, torch.Tensor]]:
        plc = self._plc_for_benchmark(benchmark, ctx) if ctx is not None else load_plc_for_benchmark(benchmark.name)
        evaluator_ctx = ctx if ctx is not None else build_benchmark_context(benchmark, self.device)
        candidate_order = list(range(len(candidates)))
        limit = int(self.config.official_rerank_limit)
        if limit <= 0:
            limit = max(1, int(self.config.top_k_candidates) * 2)
        if len(candidates) > limit:
            evaluator = TorchProxyCostEvaluator(evaluator_ctx)
            torch_scores = self._score_torch_candidates(candidates, evaluator)
            candidate_order = [
                int(idx)
                for _, idx in sorted(
                    (score, idx) for idx, score in enumerate(torch_scores) if score != float("inf")
                )[:limit]
            ]
            if not candidate_order:
                candidate_order = list(range(min(limit, len(candidates))))
        if plc is None:
            evaluator = TorchProxyCostEvaluator(evaluator_ctx)
            torch_scores = self._score_torch_candidates(candidates, evaluator)
            rows = [(torch_scores[i], candidates[i]) for i in range(len(candidates))]
            return sorted(rows, key=lambda item: item[0])

        rows: list[tuple[float, torch.Tensor]] = [(float("inf"), pos) for pos in candidates]
        for idx in candidate_order:
            pos = candidates[idx]
            costs = compute_proxy_cost(pos, benchmark, plc)
            score = float(costs["proxy_cost"]) if int(costs["overlap_count"]) == 0 else float("inf")
            rows[idx] = (score, pos)
        return sorted(rows, key=lambda item: item[0])

    def _score_torch_candidates(
        self,
        candidates: list[torch.Tensor],
        evaluator: TorchProxyCostEvaluator,
    ) -> list[float]:
        if not candidates:
            return []
        batch_limit = max(int(self.config.max_gpu_batch_candidates), 0)
        if batch_limit <= 0:
            batch_limit = len(candidates)
        scores: list[float] = []
        with torch.no_grad():
            for start in range(0, len(candidates), batch_limit):
                chunk = candidates[start : start + batch_limit]
                stacked = torch.stack([c.to(self.device, dtype=torch.float32) for c in chunk])
                costs = evaluator.evaluate_batch(stacked)
                chunk_scores = torch.where(
                    costs.is_legal,
                    costs.official_proxy,
                    torch.full_like(costs.official_proxy, torch.inf),
                )
                scores.extend(float(x) for x in chunk_scores.detach().cpu().tolist())
        return scores

    def _log_proxy_calibration(self, placement: torch.Tensor, benchmark, ctx) -> None:
        plc = self._plc_for_benchmark(benchmark, ctx)
        if plc is None:
            return
        official = compute_proxy_cost(placement, benchmark, plc)
        evaluator = TorchProxyCostEvaluator(ctx)
        with torch.no_grad():
            proxy = evaluator.evaluate_batch(placement.to(self.device, dtype=torch.float32))
        print(
            f"[dreamplace_gpu] calibration "
            f"official_cong={float(official['congestion_cost']):.6f} "
            f"torch_cong={float(proxy.congestion_cost.view(-1)[0].item()):.6f} "
            f"official_proxy={float(official['proxy_cost']):.6f} "
            f"torch_proxy={float(proxy.official_proxy.view(-1)[0].item()):.6f}"
        )

    def _plc_for_benchmark(self, benchmark, ctx):
        plc = getattr(ctx, "plc", None) if ctx is not None else None
        if plc is None:
            plc = load_plc_for_benchmark(benchmark.name)
        return plc

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

    def generate_candidates(self, benchmark, trace: PlacementTraceRecorder | None = None) -> list[torch.Tensor]:
        base = clamp_placement(legalize_initial(benchmark), benchmark).to(self.device, dtype=torch.float32)
        candidates = [base.detach().cpu()]
        if trace is not None:
            trace.record(base, "initial")

        for recipe_idx, (density_weight, gamma_scale, lr_scale, target_density) in enumerate(self.config.recipes):
            recipe_candidates = self._run_one(
                benchmark,
                base,
                recipe_idx=recipe_idx,
                density_weight=density_weight,
                gamma_scale=gamma_scale,
                lr_scale=lr_scale,
                target_density=target_density,
                trace=trace,
            )
            candidates.extend(pos.detach().cpu() for pos in recipe_candidates)
        return candidates

    def _run_one(
        self,
        benchmark,
        start: torch.Tensor,
        recipe_idx: int,
        density_weight: float,
        gamma_scale: float,
        lr_scale: float,
        target_density: float,
        trace: PlacementTraceRecorder | None = None,
    ) -> list[torch.Tensor]:
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
        snapshots: list[tuple[float, torch.Tensor]] = []
        detached_congestion_map = None
        loss_breakdown = None

        for step in range(max(int(self.config.analytical_iters), 1)):
            opt.zero_grad(set_to_none=True)
            frac = step / max(float(self.config.analytical_iters - 1), 1.0)
            gamma = gamma_scale * (1.0 - 0.65 * frac) * max(float(benchmark.canvas_width), float(benchmark.canvas_height))
            gamma = max(gamma, 1.0e-3)
            rows, cols = self._bin_grid_shape(benchmark)
            if self.config.congestion_density_alpha > 0 and self.config.congestion_map_update_interval > 0:
                if step % int(self.config.congestion_map_update_interval) == 0 or detached_congestion_map is None:
                    with torch.no_grad():
                        congestion_map = self._discrete_congestion_map(x.detach(), benchmark, sizes, rows, cols)
                        detached_congestion_map = self._density_target_from_congestion(congestion_map, target_density)
            wl = self._weighted_average_wirelength(x, benchmark, gamma)
            density = self._density_loss(x, benchmark, sizes, target_density, detached_congestion_map)
            macro_congestion = self._macro_blockage_congestion_loss(x, benchmark, sizes, rows, cols)
            soft_route_congestion = self._soft_route_congestion_loss(x, benchmark, rows, cols)
            overlap = self._overlap_loss(x, benchmark, sizes)
            boundary = self._boundary_loss(x, benchmark, sizes)
            loss = (
                wl
                + density_weight * density
                + self.config.congestion_weight * macro_congestion
                + self.config.soft_route_congestion_weight * soft_route_congestion
                + self.config.overlap_weight * overlap
                + self.config.boundary_weight * boundary
            )
            loss_breakdown = (wl, density, macro_congestion, soft_route_congestion, overlap, boundary)
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
                interval = max(int(self.config.analytical_snapshot_interval), 0)
                if interval > 0 and (step % interval == 0 or step == self.config.analytical_iters - 1):
                    snapshots.append((loss_value, x.detach().clone()))
                if trace is not None and trace.should_record_step(step):
                    trace.record(x.detach(), f"recipe_{recipe_idx}_step_{step:04d}")
        if trace is not None:
            trace.record(best.detach(), f"recipe_{recipe_idx}_best")
        if loss_breakdown is not None:
            wl, density, macro_congestion, soft_route_congestion, overlap, boundary = loss_breakdown
            print(
                f"[dreamplace_gpu] recipe={recipe_idx} "
                f"wl={float(wl.detach().item()):.6f} "
                f"density={float(density.detach().item()):.6f} "
                f"macro_cong={float(macro_congestion.detach().item()):.6f} "
                f"soft_route_cong={float(soft_route_congestion.detach().item()):.6f} "
                f"overlap={float(overlap.detach().item()):.6f} "
                f"boundary={float(boundary.detach().item()):.6f}"
            )
        out = [best]
        keep = max(int(self.config.analytical_snapshots_per_recipe), 0)
        if keep > 0 and snapshots:
            snapshots.sort(key=lambda item: item[0])
            for _, snapshot in snapshots[:keep]:
                out.append(snapshot)
        return out

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

    def _density_loss(
        self,
        placement: torch.Tensor,
        benchmark,
        sizes: torch.Tensor,
        target_density: float,
        target_density_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rows, cols = self._bin_grid_shape(benchmark)
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
        if target_density_map is None:
            target = placement.new_full((rows * cols,), float(target_density))
        else:
            target = target_density_map.to(device=self.device, dtype=placement.dtype)
        overflow = (density - target).clamp_min(0)
        k = max(int(overflow.numel() * 0.10), 1)
        return torch.topk(overflow.square(), k=k).values.mean()

    def _macro_blockage_congestion_loss(
        self,
        placement: torch.Tensor,
        benchmark,
        sizes: torch.Tensor,
        rows: int,
        cols: int,
    ) -> torch.Tensor:
        n = int(benchmark.num_hard_macros)
        if n <= 0 or rows * cols <= 1:
            return placement.new_tensor(0.0)
        grid_w = float(benchmark.canvas_width) / max(cols, 1)
        grid_h = float(benchmark.canvas_height) / max(rows, 1)
        grid_v_routes = max(grid_w * float(benchmark.vroutes_per_micron), 1.0e-9)
        grid_h_routes = max(grid_h * float(benchmark.hroutes_per_micron), 1.0e-9)
        v_macro, h_macro = self._macro_blockage_maps(
            placement, benchmark, sizes, rows, cols, grid_v_routes, grid_h_routes
        )
        pressure = torch.cat([v_macro.reshape(-1), h_macro.reshape(-1)])
        overflow = (pressure - float(self.config.congestion_target)).clamp_min(0).square()
        k = max(int(overflow.numel() * 0.10), 1)
        return torch.topk(overflow, k=k).values.mean()

    def _soft_route_congestion_loss(self, placement: torch.Tensor, benchmark, rows: int, cols: int) -> torch.Tensor:
        ctx = self.ctx
        pair_count = int(ctx.routing_weights.numel())
        if pair_count == 0 or rows * cols <= 1 or self.config.soft_route_congestion_weight <= 0:
            return placement.new_tensor(0.0)
        src = self._routing_pin_positions(
            placement,
            ctx.routing_src_parent,
            ctx.routing_src_offset,
            ctx.routing_src_port_pos,
            ctx.routing_src_is_port,
        )
        dst = self._routing_pin_positions(
            placement,
            ctx.routing_dst_parent,
            ctx.routing_dst_offset,
            ctx.routing_dst_port_pos,
            ctx.routing_dst_is_port,
        )
        grid_w = float(benchmark.canvas_width) / max(cols, 1)
        grid_h = float(benchmark.canvas_height) / max(rows, 1)
        tau_x = max(grid_w * float(self.config.soft_route_tau_scale), 1.0e-3)
        tau_y = max(grid_h * float(self.config.soft_route_tau_scale), 1.0e-3)
        col_centers = torch.linspace(
            grid_w * 0.5,
            float(benchmark.canvas_width) - grid_w * 0.5,
            cols,
            device=self.device,
            dtype=placement.dtype,
        )
        row_centers = torch.linspace(
            grid_h * 0.5,
            float(benchmark.canvas_height) - grid_h * 0.5,
            rows,
            device=self.device,
            dtype=placement.dtype,
        )
        h = placement.new_zeros((rows, cols))
        v = placement.new_zeros((rows, cols))
        chunk_size = max(int(self.config.soft_route_chunk_size), 1)
        for start in range(0, pair_count, chunk_size):
            end = min(start + chunk_size, pair_count)
            src_c = src[start:end]
            dst_c = dst[start:end]
            weights = ctx.routing_weights[start:end].to(dtype=placement.dtype).view(-1, 1, 1)
            x_min = torch.minimum(src_c[:, 0], dst_c[:, 0]).view(-1, 1)
            x_max = torch.maximum(src_c[:, 0], dst_c[:, 0]).view(-1, 1)
            y_min = torch.minimum(src_c[:, 1], dst_c[:, 1]).view(-1, 1)
            y_max = torch.maximum(src_c[:, 1], dst_c[:, 1]).view(-1, 1)
            h_window = torch.sigmoid((col_centers.view(1, -1) - x_min) / tau_x) * torch.sigmoid(
                (x_max - col_centers.view(1, -1)) / tau_x
            )
            v_window = torch.sigmoid((row_centers.view(1, -1) - y_min) / tau_y) * torch.sigmoid(
                (y_max - row_centers.view(1, -1)) / tau_y
            )
            h_row = torch.softmax(-0.5 * ((row_centers.view(1, -1) - src_c[:, 1].view(-1, 1)) / tau_y).square(), dim=1)
            v_col = torch.softmax(-0.5 * ((col_centers.view(1, -1) - dst_c[:, 0].view(-1, 1)) / tau_x).square(), dim=1)
            h = h + (weights * h_row.unsqueeze(2) * h_window.unsqueeze(1)).sum(dim=0)
            v = v + (weights * v_window.unsqueeze(2) * v_col.unsqueeze(1)).sum(dim=0)
        grid_v_routes = max(grid_w * float(benchmark.vroutes_per_micron), 1.0e-9)
        grid_h_routes = max(grid_h * float(benchmark.hroutes_per_micron), 1.0e-9)
        h = self._smooth_h_matrix(h / grid_h_routes)
        v = self._smooth_v_matrix(v / grid_v_routes)
        pressure = torch.cat([v.reshape(-1), h.reshape(-1)])
        overflow = (pressure - float(self.config.congestion_target)).clamp_min(0).square()
        k = max(int(overflow.numel() * 0.10), 1)
        return torch.topk(overflow, k=k).values.mean()

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

    def _discrete_congestion_map(
        self,
        placement: torch.Tensor,
        benchmark,
        sizes: torch.Tensor,
        rows: int,
        cols: int,
    ) -> torch.Tensor:
        ctx = self.ctx
        if rows * cols == 0:
            return placement.new_zeros((0,))
        h = placement.new_zeros((rows, cols))
        v = placement.new_zeros((rows, cols))
        pair_count = int(ctx.routing_weights.numel())
        grid_w = float(benchmark.canvas_width) / max(cols, 1)
        grid_h = float(benchmark.canvas_height) / max(rows, 1)
        if pair_count > 0:
            src = self._routing_pin_positions(
                placement,
                ctx.routing_src_parent,
                ctx.routing_src_offset,
                ctx.routing_src_port_pos,
                ctx.routing_src_is_port,
            )
            dst = self._routing_pin_positions(
                placement,
                ctx.routing_dst_parent,
                ctx.routing_dst_offset,
                ctx.routing_dst_port_pos,
                ctx.routing_dst_is_port,
            )
            src_col = torch.floor(src[:, 0] / max(grid_w, 1.0e-9)).to(torch.long).clamp(0, cols - 1)
            dst_col = torch.floor(dst[:, 0] / max(grid_w, 1.0e-9)).to(torch.long).clamp(0, cols - 1)
            src_row = torch.floor(src[:, 1] / max(grid_h, 1.0e-9)).to(torch.long).clamp(0, rows - 1)
            dst_row = torch.floor(dst[:, 1] / max(grid_h, 1.0e-9)).to(torch.long).clamp(0, rows - 1)
            weights = ctx.routing_weights.to(dtype=placement.dtype)
            col_ids = torch.arange(cols, device=self.device).view(1, cols)
            row_ids = torch.arange(rows, device=self.device).view(1, rows)
            chunk_size = max(int(self.config.soft_route_chunk_size), 1)
            for start in range(0, pair_count, chunk_size):
                end = min(start + chunk_size, pair_count)
                c0 = torch.minimum(src_col[start:end], dst_col[start:end]).view(-1, 1)
                c1 = torch.maximum(src_col[start:end], dst_col[start:end]).view(-1, 1)
                r0 = torch.minimum(src_row[start:end], dst_row[start:end]).view(-1, 1)
                r1 = torch.maximum(src_row[start:end], dst_row[start:end]).view(-1, 1)
                weight = weights[start:end].view(-1, 1, 1)
                h_cols = (col_ids >= c0) & (col_ids < c1)
                v_rows = (row_ids >= r0) & (row_ids < r1)
                h_row_onehot = F.one_hot(src_row[start:end], num_classes=rows).to(dtype=placement.dtype)
                v_col_onehot = F.one_hot(dst_col[start:end], num_classes=cols).to(dtype=placement.dtype)
                h = h + (h_row_onehot.unsqueeze(2) * h_cols.unsqueeze(1).to(dtype=placement.dtype) * weight).sum(dim=0)
                v = v + (v_rows.unsqueeze(2).to(dtype=placement.dtype) * v_col_onehot.unsqueeze(1) * weight).sum(dim=0)
        grid_v_routes = max(grid_w * float(benchmark.vroutes_per_micron), 1.0e-9)
        grid_h_routes = max(grid_h * float(benchmark.hroutes_per_micron), 1.0e-9)
        h = self._smooth_h_matrix(h / grid_h_routes)
        v = self._smooth_v_matrix(v / grid_v_routes)
        v_macro, h_macro = self._macro_blockage_maps(placement, benchmark, sizes, rows, cols, grid_v_routes, grid_h_routes)
        return torch.maximum(v + v_macro, h + h_macro).reshape(-1).detach()

    def _density_target_from_congestion(self, congestion_map: torch.Tensor, target_density: float) -> torch.Tensor:
        if congestion_map.numel() == 0:
            return congestion_map
        target = float(target_density) / (1.0 + float(self.config.congestion_density_alpha) * congestion_map.clamp_min(0))
        return target.clamp(min=float(target_density) * 0.35, max=float(target_density)).detach()

    def _macro_blockage_maps(
        self,
        placement: torch.Tensor,
        benchmark,
        sizes: torch.Tensor,
        rows: int,
        cols: int,
        grid_v_routes: float,
        grid_h_routes: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = int(benchmark.num_hard_macros)
        if n <= 0:
            z = placement.new_zeros((rows, cols))
            return z, z
        grid = self._grid_boxes(benchmark, rows, cols, placement.dtype)
        boxes = self._macro_boxes(placement[:n], sizes[:n])
        x_overlap = (
            torch.minimum(boxes[:, 2].unsqueeze(1), grid[:, 2].unsqueeze(0))
            - torch.maximum(boxes[:, 0].unsqueeze(1), grid[:, 0].unsqueeze(0))
        ).clamp_min(0)
        y_overlap = (
            torch.minimum(boxes[:, 3].unsqueeze(1), grid[:, 3].unsqueeze(0))
            - torch.maximum(boxes[:, 1].unsqueeze(1), grid[:, 1].unsqueeze(0))
        ).clamp_min(0)
        intersects = (x_overlap > 0) & (y_overlap > 0)
        row_ids = torch.arange(rows, device=self.device).repeat_interleave(cols)
        col_ids = torch.arange(cols, device=self.device).repeat(rows)
        left = boxes[:, 0]
        right = boxes[:, 2]
        bottom = boxes[:, 1]
        top = boxes[:, 3]
        grid_w = float(benchmark.canvas_width) / max(cols, 1)
        grid_h = float(benchmark.canvas_height) / max(rows, 1)
        bl_col = torch.floor(left / max(grid_w, 1.0e-9)).to(torch.long).clamp(0, cols - 1)
        ur_col = torch.floor(right / max(grid_w, 1.0e-9)).to(torch.long).clamp(0, cols - 1)
        bl_row = torch.floor(bottom / max(grid_h, 1.0e-9)).to(torch.long).clamp(0, rows - 1)
        ur_row = torch.floor(top / max(grid_h, 1.0e-9)).to(torch.long).clamp(0, rows - 1)
        bottom_partial = y_overlap.gather(1, (bl_row * cols + bl_col).view(-1, 1)).squeeze(1)
        top_partial = y_overlap.gather(1, (ur_row * cols + bl_col).view(-1, 1)).squeeze(1)
        left_partial = x_overlap.gather(1, (bl_row * cols + bl_col).view(-1, 1)).squeeze(1)
        right_partial = x_overlap.gather(1, (bl_row * cols + ur_col).view(-1, 1)).squeeze(1)
        partial_v = (ur_row != bl_row) & (
            ((bottom_partial - grid_h).abs() > 1.0e-5)
            | ((top_partial - grid_h).abs() > 1.0e-5)
        )
        partial_h = (ur_col != bl_col) & (
            ((left_partial - grid_w).abs() > 1.0e-5)
            | ((right_partial - grid_w).abs() > 1.0e-5)
        )
        v_keep = intersects & ~(partial_v.unsqueeze(1) & (row_ids.view(1, -1) == ur_row.unsqueeze(1)))
        h_keep = intersects & ~(partial_h.unsqueeze(1) & (col_ids.view(1, -1) == ur_col.unsqueeze(1)))
        v = (
            (x_overlap * v_keep.to(x_overlap.dtype) * float(self.ctx.vrouting_alloc))
            .sum(dim=0)
            .reshape(rows, cols)
            / grid_v_routes
        )
        h = (
            (y_overlap * h_keep.to(y_overlap.dtype) * float(self.ctx.hrouting_alloc))
            .sum(dim=0)
            .reshape(rows, cols)
            / grid_h_routes
        )
        return v, h

    def _routing_pin_positions(
        self,
        placement: torch.Tensor,
        parent: torch.Tensor,
        offset: torch.Tensor,
        port_pos: torch.Tensor,
        is_port: torch.Tensor,
    ) -> torch.Tensor:
        if parent.numel() == 0:
            return torch.zeros((0, 2), dtype=placement.dtype, device=self.device)
        gathered = placement[parent.clamp_min(0)] + offset.to(dtype=placement.dtype)
        return torch.where(is_port.view(-1, 1), port_pos.to(dtype=placement.dtype), gathered)

    def _smooth_v_matrix(self, v: torch.Tensor) -> torch.Tensor:
        r = max(int(self.ctx.smooth_range), 0)
        if r == 0:
            return v
        _, cols = v.shape
        out = torch.zeros_like(v)
        for col in range(cols):
            lp = max(0, col - r)
            rp = min(cols - 1, col + r)
            out[:, lp : rp + 1] += v[:, col : col + 1] / float(rp - lp + 1)
        return out

    def _smooth_h_matrix(self, h: torch.Tensor) -> torch.Tensor:
        r = max(int(self.ctx.smooth_range), 0)
        if r == 0:
            return h
        rows, _ = h.shape
        out = torch.zeros_like(h)
        for row in range(rows):
            lp = max(0, row - r)
            rp = min(rows - 1, row + r)
            out[lp : rp + 1, :] += h[row : row + 1, :] / float(rp - lp + 1)
        return out

    def _grid_boxes(self, benchmark, rows: int, cols: int, dtype: torch.dtype) -> torch.Tensor:
        xs = torch.linspace(0, float(benchmark.canvas_width), cols + 1, device=self.device, dtype=dtype)
        ys = torch.linspace(0, float(benchmark.canvas_height), rows + 1, device=self.device, dtype=dtype)
        boxes = []
        for r in range(rows):
            for c in range(cols):
                boxes.append(torch.stack([xs[c], ys[r], xs[c + 1], ys[r + 1]]))
        return torch.stack(boxes, dim=0)

    def _bin_grid_shape(self, benchmark) -> tuple[int, int]:
        rows = max(1, min(int(benchmark.grid_rows), int(self.config.bin_grid_cap)))
        cols = max(1, min(int(benchmark.grid_cols), int(self.config.bin_grid_cap)))
        return rows, cols

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
        rows = torch.arange(total, device=self.device)

        move_mask = move_types == 0
        if bool(move_mask.any()):
            candidates[rows[move_mask], chosen_i[move_mask]] += noise[move_mask]

        swap_mask = (move_types == 1) & (chosen_i != chosen_j)
        if bool(swap_mask.any()):
            swap_rows = rows[swap_mask]
            i = chosen_i[swap_mask]
            j = chosen_j[swap_mask]
            old_i = candidates[swap_rows, i].clone()
            candidates[swap_rows, i] = candidates[swap_rows, j]
            candidates[swap_rows, j] = old_i

        attract_mask = (move_types == 2) & (chosen_i != chosen_j)
        if bool(attract_mask.any()):
            attract_rows = rows[attract_mask]
            i = chosen_i[attract_mask]
            j = chosen_j[attract_mask]
            alpha = 0.08 + 0.30 * torch.rand((int(attract_rows.numel()), 1), device=self.device, generator=generator)
            candidates[attract_rows, i] = candidates[attract_rows, i] + alpha * (
                candidates[attract_rows, j] - candidates[attract_rows, i]
            )

        random_mask = move_types == 3
        if bool(random_mask.any()):
            random_rows = rows[random_mask]
            i = chosen_i[random_mask]
            random_xy = torch.rand((int(random_rows.numel()), 2), device=self.device, generator=generator)
            random_xy[:, 0] *= float(benchmark.canvas_width)
            random_xy[:, 1] *= float(benchmark.canvas_height)
            candidates[random_rows, i] = random_xy

        perm_mask = move_types == 4
        if bool(perm_mask.any()):
            k = min(5, int(movable_idx.numel()))
            perm_rows = rows[perm_mask]
            if k > 1:
                rand = torch.rand((int(perm_rows.numel()), int(movable_idx.numel())), device=self.device, generator=generator)
                perm_src_pos = rand.topk(k=k, dim=1, largest=False).indices
                perm_src = movable_idx[perm_src_pos]
                dst_order = torch.rand((int(perm_rows.numel()), k), device=self.device, generator=generator).argsort(dim=1)
                perm_dst = perm_src.gather(1, dst_order)
                candidates[perm_rows.unsqueeze(1), perm_src] = candidates[perm_rows.unsqueeze(1), perm_dst].clone()
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
