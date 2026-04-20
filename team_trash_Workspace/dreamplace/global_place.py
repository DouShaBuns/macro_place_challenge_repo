"""Global placement 主训练循环。

变量：所有 macro（hard movable + hard fixed + soft）的中心坐标
固定 macro 通过梯度 mask 不更新；canvas 边界用 clamp 强制（不进 loss）
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

from benchmark_context import WLContext, build_wl_context
from torch_density import DensityContext, adaptive_grid, build_density_context, density_energy
from torch_loss import wa_wirelength


@dataclass
class GlobalConfig:
    iters: int = 1500
    grid_base: int = 256
    lr_factor: float = 0.05         # lr = lr_factor * grid_w
    lambda_init_ratio: float = 1.0  # λ_init = ratio * (WL_0 / D_0)，让 λ·D 起始 ≈ ratio·WL
    lambda_mult: float = 1.20
    lambda_step: int = 50           # 每多少步乘一次
    gamma_start_factor: float = 4.0   # γ_start = factor * grid_w
    gamma_end_factor: float = 1.0
    log_every: int = 50
    time_budget: float = 540.0
    log_path: Optional[str] = None
    diagnostic_history: bool = False  # True 时把每步指标存内存


@dataclass
class GlobalResult:
    placement: torch.Tensor
    history: list[dict] = field(default_factory=list)
    final_step: int = 0
    elapsed: float = 0.0
    stop_reason: str = "max_iters"


def run_global_place(
    benchmark,
    config: GlobalConfig,
    device: torch.device | str = "cpu",
) -> GlobalResult:
    device = torch.device(device)

    # 1. 构造 contexts
    wl_ctx = build_wl_context(benchmark, device=device)
    rows, cols = adaptive_grid(benchmark.canvas_width, benchmark.canvas_height,
                               sizes=benchmark.macro_sizes, base=config.grid_base)
    d_ctx = build_density_context(benchmark.canvas_width, benchmark.canvas_height,
                                  rows, cols, device=device)

    # 2. 变量 + Adam
    sizes = benchmark.macro_sizes.to(device=device, dtype=torch.float32)
    fixed_mask = benchmark.macro_fixed.to(device=device)
    original_pos = benchmark.macro_positions.to(device=device, dtype=torch.float32)
    pos = original_pos.clone().detach().requires_grad_(True)

    grid_w = d_ctx.grid_w
    lr = config.lr_factor * grid_w
    optimizer = torch.optim.Adam([pos], lr=lr)

    # 3. 初始 λ：让 λ·D ≈ ratio · WL
    with torch.no_grad():
        gamma_init = config.gamma_start_factor * grid_w
        wl0, _ = wa_wirelength(pos.detach(), wl_ctx, gamma=gamma_init, normalize=True)
        d0, _ = density_energy(pos.detach(), sizes, d_ctx, normalize=True)
        d0_safe = max(float(d0.item()), 1.0e-9)
        wl0_val = float(wl0.item())
        lam = config.lambda_init_ratio * wl0_val / d0_safe
    print(f"[init] WL={wl0_val:.6f} D={float(d0.item()):.6f} lr={lr:.4f} "
          f"grid={rows}x{cols} grid_w={grid_w:.4f} λ_init={lam:.4f}")

    # 4. 主循环
    log_lines: list[str] = []
    history: list[dict] = []
    t0 = time.time()
    stop_reason = "max_iters"
    step = 0
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    canvas_w = benchmark.canvas_width
    canvas_h = benchmark.canvas_height

    for step in range(config.iters):
        elapsed = time.time() - t0
        if elapsed > config.time_budget:
            stop_reason = "time_budget"
            break

        # γ 几何衰减
        if config.iters > 1:
            frac = step / (config.iters - 1)
        else:
            frac = 1.0
        gamma = (config.gamma_start_factor *
                 (config.gamma_end_factor / config.gamma_start_factor) ** frac) * grid_w

        # λ 几何增长
        if step > 0 and step % config.lambda_step == 0:
            lam *= config.lambda_mult

        optimizer.zero_grad()
        wl, _ = wa_wirelength(pos, wl_ctx, gamma=gamma, normalize=True)
        d, diag = density_energy(pos, sizes, d_ctx, normalize=True)
        loss = wl + lam * d
        loss.backward()

        # 固定 macro 梯度归零
        if bool(fixed_mask.any()):
            pos.grad[fixed_mask] = 0.0

        optimizer.step()

        # canvas 边界 clamp + 强制还原 fixed macro 位置（防 Adam momentum 漂移）
        with torch.no_grad():
            pos.data[:, 0].clamp_(half_w, canvas_w - half_w)
            pos.data[:, 1].clamp_(half_h, canvas_h - half_h)
            if bool(fixed_mask.any()):
                pos.data[fixed_mask] = original_pos[fixed_mask]

        if (step % config.log_every == 0) or (step == config.iters - 1):
            rec = {
                "step": step,
                "wl": float(wl.detach().item()),
                "density": float(d.detach().item()),
                "loss": float(loss.detach().item()),
                "lambda": float(lam),
                "gamma": float(gamma),
                "lr": float(lr),
                "overflow": float(diag["overflow"]),
                "rho_max": float(diag["rho_max"]),
                "elapsed": float(elapsed),
            }
            if config.log_path:
                log_lines.append(json.dumps(rec))
            if config.diagnostic_history:
                history.append(rec)

    elapsed = time.time() - t0

    if config.log_path:
        Path(config.log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config.log_path, "a", encoding="utf-8") as f:
            for line in log_lines:
                f.write(line + "\n")

    return GlobalResult(
        placement=pos.detach().cpu(),
        history=history,
        final_step=step,
        elapsed=elapsed,
        stop_reason=stop_reason,
    )
