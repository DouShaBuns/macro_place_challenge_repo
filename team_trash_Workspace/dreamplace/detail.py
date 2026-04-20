"""Detail placement：格点 hill-climb。

对每个可动 hard macro，在 4 步长 × 4 方向上试探。只接受 official proxy 严格下降
且仍合法（overlap=0）的步。用官方 compute_proxy_cost 做裁判，拿到的就是最终分。
"""

from __future__ import annotations

import time

import torch

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost

from legalize import clamp_inside_canvas


def hill_climb(
    placement: torch.Tensor,
    benchmark: Benchmark,
    plc,
    max_trials: int = 1000,
    grid_w: float | None = None,
    time_budget: float = 120.0,
) -> tuple[torch.Tensor, dict]:
    """局部格点 hill-climb。外循环 step（粗到细），内循环 macro，最内 4 方向。

    返回 (new_placement, diag) — diag 含 trials 数、accepts 数、proxy 变化等。
    """
    if grid_w is None:
        grid_w = min(benchmark.canvas_width, benchmark.canvas_height) / 256.0

    step_sizes = [2.0 * grid_w, 1.0 * grid_w, 0.5 * grid_w, 0.25 * grid_w]
    directions = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]

    current = clamp_inside_canvas(placement.clone(), benchmark)
    base = compute_proxy_cost(current, benchmark, plc)
    base_proxy = float(base["proxy_cost"])
    if int(base["overlap_count"]) != 0:
        return current, {
            "trials": 0, "accepts": 0,
            "proxy_start": base_proxy, "proxy_end": base_proxy,
            "abort": "overlap_nonzero",
        }

    n_hard = int(benchmark.num_hard_macros)
    fixed = benchmark.macro_fixed[:n_hard].detach().cpu().numpy()
    movable = [i for i in range(n_hard) if not fixed[i]]

    trials = 0
    accepts = 0
    t0 = time.time()
    proxy_start = base_proxy

    for step in step_sizes:
        improved_in_pass = False
        for idx in movable:
            if trials >= max_trials:
                break
            if time.time() - t0 > time_budget:
                break
            for dx, dy in directions:
                trials += 1
                cand = current.clone()
                cand[idx, 0] = cand[idx, 0] + dx * step
                cand[idx, 1] = cand[idx, 1] + dy * step
                cand = clamp_inside_canvas(cand, benchmark)
                m = compute_proxy_cost(cand, benchmark, plc)
                if int(m["overlap_count"]) != 0:
                    continue
                if float(m["proxy_cost"]) < base_proxy - 1.0e-6:
                    current = cand
                    base_proxy = float(m["proxy_cost"])
                    accepts += 1
                    improved_in_pass = True
                    break  # 贪心：本 macro 有改进就进下一个
        if trials >= max_trials or time.time() - t0 > time_budget:
            break
        if not improved_in_pass:
            # 当前步长无改进，换更细步长
            continue

    elapsed = time.time() - t0
    return current, {
        "trials": trials,
        "accepts": accepts,
        "proxy_start": proxy_start,
        "proxy_end": base_proxy,
        "proxy_delta": base_proxy - proxy_start,
        "elapsed": elapsed,
    }
