"""可微 density 项（eDensity / 静电势能）。

物理类比：把 macro 当电荷云，求 -∇²ψ = ρ - ρ̄ 的电势 ψ；
电势能 D = ½∫ρψ 越高表示越拥挤；梯度 ∇D 是排斥力。

v1 实现选型：
- Painting 用硬相交矩形（对中心几乎处处可导）
- Poisson 用 2D FFT + 周期边界（不严格 Neumann，但 macro 不接触边界时近似良好）
- Grid 自适应：每维 clamp(ceil(canvas/min_macro)*2, 64, 512)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class DensityContext:
    """eDensity 计算所需的网格几何参数。"""

    rows: int
    cols: int
    canvas_w: float
    canvas_h: float
    grid_w: float
    grid_h: float
    grid_area: float
    grid_x_edges: torch.Tensor  # [cols+1]
    grid_y_edges: torch.Tensor  # [rows+1]
    inv_k2: torch.Tensor        # [rows, cols]，1/(k_u^2 + k_v^2)，DC 已置 0


def adaptive_grid(canvas_w: float, canvas_h: float, sizes: torch.Tensor | None = None,
                  base: int = 256, lo: int = 64, hi: int = 512) -> tuple[int, int]:
    """按 canvas 长宽比分配 base 大小的网格。

    `sizes` 暂保留参数兼容性（v1 不使用，避免被极小的 soft macro 拉爆 rows/cols）。
    长边等于 base，短边按 canvas aspect 缩放，再 clamp 到 [lo, hi]。
    """
    if canvas_w >= canvas_h:
        cols = base
        rows = max(lo, min(hi, int(round(base * canvas_h / canvas_w))))
    else:
        rows = base
        cols = max(lo, min(hi, int(round(base * canvas_w / canvas_h))))
    return rows, cols


def build_density_context(canvas_w: float, canvas_h: float, rows: int, cols: int,
                          device: torch.device | str = "cpu") -> DensityContext:
    device = torch.device(device)
    grid_w = canvas_w / cols
    grid_h = canvas_h / rows
    grid_x_edges = torch.linspace(0.0, canvas_w, cols + 1, dtype=torch.float32, device=device)
    grid_y_edges = torch.linspace(0.0, canvas_h, rows + 1, dtype=torch.float32, device=device)

    # FFT 频率 (周期 BC)：2π * fftfreq / spacing
    ku = torch.fft.fftfreq(rows, d=grid_h).to(device) * (2.0 * math.pi)  # [rows]
    kv = torch.fft.fftfreq(cols, d=grid_w).to(device) * (2.0 * math.pi)  # [cols]
    k2 = ku.view(-1, 1) ** 2 + kv.view(1, -1) ** 2                       # [rows, cols]
    inv_k2 = torch.where(k2 > 0, 1.0 / k2, torch.zeros_like(k2))         # DC = 0

    return DensityContext(
        rows=rows, cols=cols,
        canvas_w=canvas_w, canvas_h=canvas_h,
        grid_w=grid_w, grid_h=grid_h,
        grid_area=grid_w * grid_h,
        grid_x_edges=grid_x_edges,
        grid_y_edges=grid_y_edges,
        inv_k2=inv_k2,
    )


def paint_density(macro_pos: torch.Tensor, sizes: torch.Tensor,
                  ctx: DensityContext) -> torch.Tensor:
    """把 macro 涂到 [rows, cols] 密度场。

    macro_pos: [N, 2] 中心坐标，可 requires_grad
    sizes: [N, 2] (w, h)
    返回: rho [rows, cols]，单位 = 占用面积比例（无量纲）
    """
    half = sizes / 2
    left = (macro_pos[:, 0] - half[:, 0]).unsqueeze(1)   # [N, 1]
    right = (macro_pos[:, 0] + half[:, 0]).unsqueeze(1)
    bot = (macro_pos[:, 1] - half[:, 1]).unsqueeze(1)
    top = (macro_pos[:, 1] + half[:, 1]).unsqueeze(1)

    gx_l = ctx.grid_x_edges[:-1].view(1, -1)             # [1, cols]
    gx_r = ctx.grid_x_edges[1:].view(1, -1)
    gy_b = ctx.grid_y_edges[:-1].view(1, -1)             # [1, rows]
    gy_t = ctx.grid_y_edges[1:].view(1, -1)

    ox = (torch.minimum(right, gx_r) - torch.maximum(left, gx_l)).clamp_min(0.0)   # [N, cols]
    oy = (torch.minimum(top, gy_t) - torch.maximum(bot, gy_b)).clamp_min(0.0)      # [N, rows]
    # 外积聚合：rho[r, c] = Σ_i oy[i, r] * ox[i, c]
    rho = torch.einsum("ir,ic->rc", oy, ox) / ctx.grid_area
    return rho


def solve_poisson_fft(rho: torch.Tensor, ctx: DensityContext) -> torch.Tensor:
    """周期边界 2D Poisson 解算：-∇²ψ = ρ - ρ̄"""
    rho_zm = rho - rho.mean()
    rho_hat = torch.fft.fft2(rho_zm)
    psi_hat = rho_hat * ctx.inv_k2
    psi = torch.fft.ifft2(psi_hat).real
    return psi


def density_energy(macro_pos: torch.Tensor, sizes: torch.Tensor,
                   ctx: DensityContext, normalize: bool = True) -> tuple[torch.Tensor, dict]:
    """eDensity 标量能量。

    返回 (energy, diag)；diag 含 overflow（重叠率诊断指标）。
    overflow 定义：sum(max(rho - target, 0)) / sum(rho)，target=1 表示满载。
    """
    rho = paint_density(macro_pos, sizes, ctx)
    psi = solve_poisson_fft(rho, ctx)
    energy = 0.5 * (rho * psi).sum() * ctx.grid_area
    if normalize:
        # 归一化让 energy 接近 O(1)：除以 (canvas_w * canvas_h)
        energy = energy / (ctx.canvas_w * ctx.canvas_h)
    target = 1.0  # 满载阈值
    rho_detached = rho.detach()
    overflow = (rho_detached - target).clamp_min(0).sum() / rho_detached.sum().clamp_min(1.0e-9)
    return energy, {
        "rho_max": float(rho_detached.max().item()),
        "rho_mean": float(rho_detached.mean().item()),
        "overflow": float(overflow.item()),
    }
