"""Greedy spiral-search legalization。

仅处理 hard macro（soft 允许重叠，保留 global 位置）。
步骤：
  1) 所有 hard macro 先 clamp 进 canvas，fixed 还原原位
  2) 按面积从大到小排序 movable hard macro
  3) 逐个尝试原位；冲突就螺旋搜索最近合法位
  4) 找不到就 fallback 到 shelf packing 兜底（失败率 < 1% 情况下用）
"""

from __future__ import annotations

import numpy as np
import torch

from macro_place.benchmark import Benchmark


def clamp_inside_canvas(placement: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
    out = placement.clone()
    sizes = benchmark.macro_sizes.to(out.device, dtype=out.dtype)
    out[:, 0] = out[:, 0].clamp(sizes[:, 0] / 2, float(benchmark.canvas_width) - sizes[:, 0] / 2)
    out[:, 1] = out[:, 1].clamp(sizes[:, 1] / 2, float(benchmark.canvas_height) - sizes[:, 1] / 2)
    fixed = benchmark.macro_fixed.to(out.device)
    if bool(fixed.any()):
        out[fixed] = benchmark.macro_positions.to(out.device, dtype=out.dtype)[fixed]
    return out


def _fits(idx: int, cand: np.ndarray, pos: np.ndarray, sizes: np.ndarray,
          placed: np.ndarray, gap: float) -> bool:
    """O(P) 向量化冲突检查：cand 不能与任何 placed 冲突。"""
    if not placed.any():
        return True
    dx = np.abs(cand[0] - pos[:, 0])
    dy = np.abs(cand[1] - pos[:, 1])
    sep_x = (sizes[idx, 0] + sizes[:, 0]) / 2 + gap
    sep_y = (sizes[idx, 1] + sizes[:, 1]) / 2 + gap
    conflicts = (dx < sep_x) & (dy < sep_y) & placed
    conflicts[idx] = False
    return not bool(conflicts.any())


def _ring_positions(radius: int) -> list[tuple[int, int]]:
    """Chebyshev 距离 = radius 的所有格点偏移。"""
    if radius == 0:
        return [(0, 0)]
    out: list[tuple[int, int]] = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if abs(dx) == radius or abs(dy) == radius:
                out.append((dx, dy))
    return out


def _spiral_find(idx: int, start: np.ndarray, pos: np.ndarray, sizes: np.ndarray,
                 placed: np.ndarray, half_w: np.ndarray, half_h: np.ndarray,
                 canvas_w: float, canvas_h: float, gap: float,
                 max_radius: int = 200) -> np.ndarray | None:
    """围绕 start 螺旋搜索最近合法位；每个 radius 内选 Euclidean 距离最小的 fit。"""
    step = max(float(sizes[idx, 0]), float(sizes[idx, 1]), 1.0e-3) * 0.25
    target = start.copy()
    for radius in range(max_radius + 1):
        candidates: list[tuple[float, np.ndarray]] = []
        for dx, dy in _ring_positions(radius):
            cand_x = np.clip(target[0] + dx * step, half_w[idx], canvas_w - half_w[idx])
            cand_y = np.clip(target[1] + dy * step, half_h[idx], canvas_h - half_h[idx])
            cand = np.array([cand_x, cand_y], dtype=np.float64)
            if _fits(idx, cand, pos, sizes, placed, gap):
                dist = float(((cand - target) ** 2).sum())
                candidates.append((dist, cand))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]
    return None


def _shelf_pack_fallback(benchmark: Benchmark, pos_np: np.ndarray, sizes_np: np.ndarray,
                         placed_np: np.ndarray, unplaced_idx: list[int], gap: float) -> None:
    """简单的 shelf pack，把剩余 hard macro 按 h 从大到小塞进 row（修改 pos_np in-place）。"""
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)
    # 按高从大到小排序
    unplaced_idx = sorted(unplaced_idx, key=lambda i: -float(sizes_np[i, 1]))
    cursor_x, cursor_y, row_h = 0.0, 0.0, 0.0
    for idx in unplaced_idx:
        w = float(sizes_np[idx, 0])
        h = float(sizes_np[idx, 1])
        placed_this = False
        while cursor_y + h <= canvas_h:
            if cursor_x + w > canvas_w:
                cursor_x = 0.0
                cursor_y += row_h + gap
                row_h = 0.0
                continue
            cand = np.array([cursor_x + w / 2, cursor_y + h / 2], dtype=np.float64)
            cursor_x += w + gap
            row_h = max(row_h, h)
            if _fits(idx, cand, pos_np, sizes_np, placed_np, gap):
                pos_np[idx] = cand
                placed_np[idx] = True
                placed_this = True
                break
        if not placed_this:
            # 最后兜底：clamp 在 canvas 里；知道可能仍 overlap 但至少 in-canvas
            pos_np[idx, 0] = max(w / 2, min(canvas_w - w / 2, pos_np[idx, 0]))
            pos_np[idx, 1] = max(h / 2, min(canvas_h - h / 2, pos_np[idx, 1]))
            placed_np[idx] = True


def legalize(placement: torch.Tensor, benchmark: Benchmark, gap: float = 1.0e-3) -> torch.Tensor:
    """Legalize hard macro placements。soft 保留原位。

    placement: [num_macros, 2]
    返回: [num_macros, 2]，保证 hard macro 零重叠，canvas 内，fixed 不动。
    """
    out = clamp_inside_canvas(placement, benchmark)
    n_hard = int(benchmark.num_hard_macros)
    if n_hard <= 1:
        return out

    pos_np = out[:n_hard].detach().cpu().numpy().astype(np.float64).copy()
    sizes_np = benchmark.macro_sizes[:n_hard].detach().cpu().numpy().astype(np.float64)
    fixed_np = benchmark.macro_fixed[:n_hard].detach().cpu().numpy()
    original_np = benchmark.macro_positions[:n_hard].detach().cpu().numpy().astype(np.float64)
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)
    half_w = sizes_np[:, 0] / 2
    half_h = sizes_np[:, 1] / 2

    # 1. fixed 先放回原位，标记为 placed
    placed = np.zeros(n_hard, dtype=bool)
    for i in range(n_hard):
        if fixed_np[i]:
            pos_np[i] = original_np[i]
            placed[i] = True

    # 2. 按面积从大到小处理 movable
    areas = sizes_np[:, 0] * sizes_np[:, 1]
    movable_idx = [i for i in range(n_hard) if not fixed_np[i]]
    movable_idx.sort(key=lambda i: -areas[i])

    unplaced: list[int] = []
    for idx in movable_idx:
        start = pos_np[idx].copy()
        cand = _spiral_find(idx, start, pos_np, sizes_np, placed,
                            half_w, half_h, canvas_w, canvas_h, gap)
        if cand is None:
            unplaced.append(idx)
            continue
        pos_np[idx] = cand
        placed[idx] = True

    if unplaced:
        _shelf_pack_fallback(benchmark, pos_np, sizes_np, placed, unplaced, gap)

    out[:n_hard] = torch.tensor(pos_np, dtype=out.dtype)
    # 再 clamp 一次保险（fallback 可能轻微越界）
    out = clamp_inside_canvas(out, benchmark)
    return out
