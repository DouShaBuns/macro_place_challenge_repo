"""把 Benchmark 摊平成 GPU 友好的张量上下文。

设计：
- pin = macro 中心（v1 不抽 pin offset）
- node 索引 < num_macros 是 movable/fixed macro，>= num_macros 是 boundary port
- 把 macro_positions 和 port_positions 拼起来，统一通过 node 索引查 pin 坐标
- 跳过单 pin net（无 WL 贡献）
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from macro_place.benchmark import Benchmark


@dataclass
class WLContext:
    """Wirelength 计算所需的扁平化数据。"""

    pin_node_id: torch.Tensor  # [P] long, node 索引；可能指向 macro 或 port
    pin_net_id: torch.Tensor   # [P] long
    net_weights: torch.Tensor  # [num_nets_kept] float32
    port_positions: torch.Tensor  # [num_ports, 2] float32（不参与梯度）
    num_macros: int
    num_ports: int
    num_kept_nets: int
    canvas_w: float
    canvas_h: float
    device: torch.device


def build_wl_context(benchmark: Benchmark, device: torch.device | str = "cpu") -> WLContext:
    device = torch.device(device)
    pin_node_id: list[int] = []
    pin_net_id: list[int] = []
    kept_weights: list[float] = []
    kept_id = 0
    for orig_net_id, nodes in enumerate(benchmark.net_nodes):
        if nodes.numel() < 2:
            continue
        # 去重保留顺序：同一 net 里同一节点重复出现没有 WL 贡献
        seen: set[int] = set()
        unique: list[int] = []
        for n in nodes.tolist():
            n = int(n)
            if n not in seen:
                seen.add(n)
                unique.append(n)
        if len(unique) < 2:
            continue
        pin_node_id.extend(unique)
        pin_net_id.extend([kept_id] * len(unique))
        kept_weights.append(float(benchmark.net_weights[orig_net_id].item()))
        kept_id += 1

    return WLContext(
        pin_node_id=torch.tensor(pin_node_id, dtype=torch.long, device=device),
        pin_net_id=torch.tensor(pin_net_id, dtype=torch.long, device=device),
        net_weights=torch.tensor(kept_weights, dtype=torch.float32, device=device),
        port_positions=benchmark.port_positions.to(device=device, dtype=torch.float32),
        num_macros=int(benchmark.num_macros),
        num_ports=int(benchmark.port_positions.shape[0]),
        num_kept_nets=kept_id,
        canvas_w=float(benchmark.canvas_width),
        canvas_h=float(benchmark.canvas_height),
        device=device,
    )
