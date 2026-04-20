"""可微 wirelength loss（Weighted Average / RePlAce 公式）。

WA(x; gamma) ≈ max(x) 当 gamma → 0；γ 越大越平滑（梯度噪声越小）。
数值稳定：先减 per-net max/min，再做 exp，避免溢出。
"""

from __future__ import annotations

import torch

from benchmark_context import WLContext


def wa_max(x: torch.Tensor, net_id: torch.Tensor, num_nets: int, gamma: float) -> torch.Tensor:
    """逐 net 的 weighted average max。

    x: [P] pin 坐标
    net_id: [P] 每个 pin 属于哪个 net
    返回: [num_nets] 每个 net 的 WA-max
    """
    # 1. 每 net 的真 max（用于减偏移，防止 exp 溢出）
    max_per_net = torch.full(
        (num_nets,), float("-inf"), dtype=x.dtype, device=x.device
    ).scatter_reduce(0, net_id, x.detach(), reduce="amax", include_self=True)
    # 单 pin net 不可能出现（context 已过滤），但保险：将 -inf 替换 0
    max_per_net = torch.where(torch.isfinite(max_per_net), max_per_net, torch.zeros_like(max_per_net))
    max_per_pin = max_per_net[net_id]
    z = torch.exp((x - max_per_pin) / gamma)
    sum_z = torch.zeros(num_nets, dtype=x.dtype, device=x.device).scatter_add_(0, net_id, z)
    sum_xz = torch.zeros(num_nets, dtype=x.dtype, device=x.device).scatter_add_(0, net_id, x * z)
    return sum_xz / sum_z.clamp_min(1.0e-30)


def wa_min(x: torch.Tensor, net_id: torch.Tensor, num_nets: int, gamma: float) -> torch.Tensor:
    """逐 net 的 weighted average min（用 -wa_max(-x) 复用同一稳定化路径）。"""
    return -wa_max(-x, net_id, num_nets, gamma)


def wa_wirelength(
    macro_pos: torch.Tensor,
    ctx: WLContext,
    gamma: float,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """完整 WA wirelength。

    macro_pos: [num_macros, 2] requires_grad=True
    ctx: WLContext
    gamma: 平滑度（典型 = grid_w * 几）
    normalize: 是否除以 (canvas_w + canvas_h) * num_nets，使数值与官方 proxy 同量纲
    返回 (loss_scalar, per_net_hpwl_unnormalized) 后者用于诊断
    """
    if ctx.num_ports > 0:
        all_pos = torch.cat([macro_pos, ctx.port_positions.detach()], dim=0)
    else:
        all_pos = macro_pos
    pin_x = all_pos[ctx.pin_node_id, 0]
    pin_y = all_pos[ctx.pin_node_id, 1]
    n = ctx.num_kept_nets
    wa_x_max = wa_max(pin_x, ctx.pin_net_id, n, gamma)
    wa_x_min = wa_min(pin_x, ctx.pin_net_id, n, gamma)
    wa_y_max = wa_max(pin_y, ctx.pin_net_id, n, gamma)
    wa_y_min = wa_min(pin_y, ctx.pin_net_id, n, gamma)
    per_net_wl = (wa_x_max - wa_x_min) + (wa_y_max - wa_y_min)
    weighted = (per_net_wl * ctx.net_weights).sum()
    if normalize:
        denom = (ctx.canvas_w + ctx.canvas_h) * max(n, 1)
        return weighted / denom, per_net_wl
    return weighted, per_net_wl
