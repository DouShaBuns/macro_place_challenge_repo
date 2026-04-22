from __future__ import annotations

import numpy as np
import torch

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_overlap_metrics


def clamp_placement(placement: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
    out = placement.clone()
    sizes = benchmark.macro_sizes.to(out.device, dtype=out.dtype)
    out[:, 0] = out[:, 0].clamp(sizes[:, 0] / 2, float(benchmark.canvas_width) - sizes[:, 0] / 2)
    out[:, 1] = out[:, 1].clamp(sizes[:, 1] / 2, float(benchmark.canvas_height) - sizes[:, 1] / 2)
    fixed = benchmark.macro_fixed.to(out.device)
    if bool(fixed.any()):
        out[fixed] = benchmark.macro_positions.to(out.device, dtype=out.dtype)[fixed]
    return out


def legalize_initial(benchmark: Benchmark, gap: float = 0.001) -> torch.Tensor:
    return legalize_placement(benchmark.macro_positions, benchmark, gap=gap)


def legalize_placement(source: torch.Tensor, benchmark: Benchmark, gap: float = 0.001) -> torch.Tensor:
    n_hard = int(benchmark.num_hard_macros)
    placement = source.clone()
    if n_hard <= 1:
        return clamp_placement(placement, benchmark)

    pos = placement[:n_hard].cpu().numpy().astype(np.float64).copy()
    original = benchmark.macro_positions[:n_hard].cpu().numpy().astype(np.float64)
    sizes = benchmark.macro_sizes[:n_hard].cpu().numpy().astype(np.float64)
    movable = benchmark.get_movable_mask()[:n_hard].cpu().numpy()
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2 + gap
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2 + gap

    legal = pos.copy()
    legal[:, 0] = np.clip(legal[:, 0], half_w, cw - half_w)
    legal[:, 1] = np.clip(legal[:, 1], half_h, ch - half_h)
    legal[~movable] = original[~movable]
    order = sorted((i for i in range(n_hard) if movable[i]), key=lambda i: -sizes[i, 0] * sizes[i, 1])
    placed = ~movable

    for idx in order:
        if _fits(idx, legal[idx], legal, sep_x, sep_y, placed):
            placed[idx] = True
            continue
        step = max(float(sizes[idx, 0]), float(sizes[idx, 1]), 1.0e-3) * 0.25
        best = legal[idx].copy()
        best_dist = float("inf")
        for radius in range(1, 180):
            found = False
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) != radius and abs(dy) != radius:
                        continue
                    cand = np.array(
                        [
                            np.clip(pos[idx, 0] + dx * step, half_w[idx], cw - half_w[idx]),
                            np.clip(pos[idx, 1] + dy * step, half_h[idx], ch - half_h[idx]),
                        ]
                    )
                    if not _fits(idx, cand, legal, sep_x, sep_y, placed):
                        continue
                    dist = float(((cand - pos[idx]) ** 2).sum())
                    if dist < best_dist:
                        best = cand
                        best_dist = dist
                        found = True
            if found:
                break
        legal[idx] = best
        placed[idx] = True

    placement[:n_hard] = torch.tensor(legal, dtype=placement.dtype)
    placement = clamp_placement(placement, benchmark)
    if compute_overlap_metrics(placement, benchmark)["overlap_count"] > 0:
        placement = shelf_pack(benchmark, gap=gap)
    return placement


def shelf_pack(benchmark: Benchmark, gap: float = 0.001) -> torch.Tensor:
    placement = benchmark.macro_positions.clone()
    movable = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
    hard = benchmark.get_hard_macro_mask()
    indices = torch.where(movable)[0].tolist()
    sizes = benchmark.macro_sizes
    indices.sort(key=lambda i: -float(sizes[i, 1]))
    cursor_x = 0.0
    cursor_y = 0.0
    row_height = 0.0
    pos = placement[: benchmark.num_hard_macros].cpu().numpy().astype(np.float64).copy()
    sizes_np = sizes[: benchmark.num_hard_macros].cpu().numpy().astype(np.float64)
    hard_np = hard[: benchmark.num_hard_macros].cpu().numpy()
    placed = hard_np & ~movable[: benchmark.num_hard_macros].cpu().numpy()
    sep_x = (sizes_np[:, 0:1] + sizes_np[:, 0:1].T) / 2 + gap
    sep_y = (sizes_np[:, 1:2] + sizes_np[:, 1:2].T) / 2 + gap
    for idx in indices:
        w = float(sizes[idx, 0])
        h = float(sizes[idx, 1])
        found = False
        while cursor_y + h <= float(benchmark.canvas_height):
            if cursor_x + w > float(benchmark.canvas_width):
                cursor_x = 0.0
                cursor_y += row_height + gap
                row_height = 0.0
                continue
            cand = np.array([cursor_x + w / 2, cursor_y + h / 2], dtype=np.float64)
            cursor_x += w + gap
            row_height = max(row_height, h)
            if _fits(int(idx), cand, pos, sep_x, sep_y, placed):
                pos[int(idx)] = cand
                placement[idx] = torch.tensor(cand, dtype=placement.dtype)
                placed[int(idx)] = True
                found = True
                break
        if not found:
            placement[idx, 0] = min(max(w / 2, placement[idx, 0].item()), float(benchmark.canvas_width) - w / 2)
            placement[idx, 1] = min(max(h / 2, placement[idx, 1].item()), float(benchmark.canvas_height) - h / 2)
    return clamp_placement(placement, benchmark)


def _fits(idx: int, cand: np.ndarray, legal: np.ndarray, sep_x: np.ndarray, sep_y: np.ndarray, placed: np.ndarray) -> bool:
    if not placed.any():
        return True
    dx = np.abs(cand[0] - legal[:, 0])
    dy = np.abs(cand[1] - legal[:, 1])
    conflicts = (dx < sep_x[idx]) & (dy < sep_y[idx]) & placed
    conflicts[idx] = False
    return not bool(conflicts.any())
