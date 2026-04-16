from __future__ import annotations

from dataclasses import dataclass

import torch

from benchmark_context import BenchmarkContext


@dataclass
class CostBreakdown:
    search_score: torch.Tensor
    official_proxy: torch.Tensor
    wirelength_cost: torch.Tensor
    density_cost: torch.Tensor
    congestion_cost: torch.Tensor
    overlap_count: torch.Tensor
    total_overlap_area: torch.Tensor
    max_overlap_area: torch.Tensor
    boundary_violation: torch.Tensor
    legality_penalty: torch.Tensor
    is_legal: torch.Tensor


class TorchProxyCostEvaluator:
    def __init__(
        self,
        context: BenchmarkContext,
        overlap_weight: float = 1000.0,
        boundary_weight: float = 1000.0,
        gap: float = 1.0e-4,
        congestion_mode: str = "exact",
    ):
        self.ctx = context
        self.benchmark = context.benchmark
        self.device = context.device
        self.overlap_weight = float(overlap_weight)
        self.boundary_weight = float(boundary_weight)
        self.gap = float(gap)
        self.congestion_mode = congestion_mode.lower()
        if self.congestion_mode not in {"exact", "fast", "none"}:
            raise ValueError(f"Unsupported congestion_mode: {congestion_mode}")
        self.sizes = self.benchmark.macro_sizes.to(self.device, dtype=torch.float32)
        self.fixed_mask = self.benchmark.macro_fixed.to(self.device)
        self.original_positions = self.benchmark.macro_positions.to(self.device, dtype=torch.float32)
        self.num_hard = int(self.benchmark.num_hard_macros)
        self.canvas_w = float(self.benchmark.canvas_width)
        self.canvas_h = float(self.benchmark.canvas_height)
        self.grid_rows = int(self.benchmark.grid_rows)
        self.grid_cols = int(self.benchmark.grid_cols)
        self.grid_w = self.canvas_w / max(self.grid_cols, 1)
        self.grid_h = self.canvas_h / max(self.grid_rows, 1)
        self.grid_area = self.grid_w * self.grid_h
        self.grid_boxes = self._make_grid_boxes()
        self.net_pin_ranges = self._make_net_pin_ranges()

    def evaluate_batch(self, placements: torch.Tensor) -> CostBreakdown:
        if placements.dim() == 2:
            placements = placements.unsqueeze(0)
        placements = placements.to(self.device, dtype=torch.float32)
        wirelength = self._wirelength(placements)
        density = self._density(placements)
        congestion = self._congestion(placements)
        overlap_count, total_overlap, max_overlap = self._overlap(placements)
        boundary = self._boundary_violation(placements)
        fixed_violation = self._fixed_violation(placements)
        legality_penalty = (
            self.overlap_weight * total_overlap
            + self.boundary_weight * boundary
            + self.boundary_weight * fixed_violation
        )
        official_proxy = wirelength + 0.5 * density + 0.5 * congestion
        search_score = official_proxy + legality_penalty
        is_legal = (overlap_count == 0) & (boundary <= self.gap) & (fixed_violation <= self.gap)
        return CostBreakdown(
            search_score=search_score,
            official_proxy=official_proxy,
            wirelength_cost=wirelength,
            density_cost=density,
            congestion_cost=congestion,
            overlap_count=overlap_count,
            total_overlap_area=total_overlap,
            max_overlap_area=max_overlap,
            boundary_violation=boundary + fixed_violation,
            legality_penalty=legality_penalty,
            is_legal=is_legal,
        )

    def _pin_positions(self, placements, parent, offset, port_pos, is_port):
        batch = placements.shape[0]
        if parent.numel() == 0:
            return torch.zeros((batch, 0, 2), dtype=torch.float32, device=self.device)
        gathered = placements[:, parent.clamp_min(0), :] + offset.unsqueeze(0)
        ports = port_pos.unsqueeze(0).expand(batch, -1, -1)
        return torch.where(is_port.view(1, -1, 1), ports, gathered)

    def _wirelength(self, placements):
        ctx = self.ctx
        batch = placements.shape[0]
        if ctx.net_pin_parent.numel() == 0 or ctx.num_nets == 0:
            return torch.zeros(batch, dtype=torch.float32, device=self.device)
        pin_pos = self._pin_positions(
            placements,
            ctx.net_pin_parent,
            ctx.net_pin_offset,
            ctx.net_pin_port_pos,
            ctx.net_pin_is_port,
        )
        net_ids = ctx.net_pin_net_id.unsqueeze(0).expand(batch, -1)
        x = pin_pos[:, :, 0]
        y = pin_pos[:, :, 1]
        x_min = torch.full((batch, ctx.num_nets), torch.inf, dtype=torch.float32, device=self.device)
        x_max = torch.full((batch, ctx.num_nets), -torch.inf, dtype=torch.float32, device=self.device)
        y_min = torch.full((batch, ctx.num_nets), torch.inf, dtype=torch.float32, device=self.device)
        y_max = torch.full((batch, ctx.num_nets), -torch.inf, dtype=torch.float32, device=self.device)
        x_min.scatter_reduce_(1, net_ids, x, reduce="amin", include_self=True)
        x_max.scatter_reduce_(1, net_ids, x, reduce="amax", include_self=True)
        y_min.scatter_reduce_(1, net_ids, y, reduce="amin", include_self=True)
        y_max.scatter_reduce_(1, net_ids, y, reduce="amax", include_self=True)
        valid = torch.isfinite(x_min) & torch.isfinite(x_max)
        hpwl = torch.where(valid, (x_max - x_min).abs() + (y_max - y_min).abs(), 0.0)
        denom = (self.canvas_w + self.canvas_h) * max(float(ctx.wirelength_norm_net_count), 1.0)
        return (hpwl * ctx.net_weights.unsqueeze(0)).sum(dim=1) / max(denom, 1.0e-9)

    def _overlap(self, placements):
        batch = placements.shape[0]
        n = self.num_hard
        if n <= 1:
            z = torch.zeros(batch, dtype=torch.float32, device=self.device)
            return z.to(torch.long), z, z
        pos = placements[:, :n, :]
        sizes = self.sizes[:n]
        left = pos[:, :, 0] - sizes[:, 0].view(1, n) / 2
        right = pos[:, :, 0] + sizes[:, 0].view(1, n) / 2
        bottom = pos[:, :, 1] - sizes[:, 1].view(1, n) / 2
        top = pos[:, :, 1] + sizes[:, 1].view(1, n) / 2
        ox = (torch.minimum(right.unsqueeze(2), right.unsqueeze(1)) - torch.maximum(left.unsqueeze(2), left.unsqueeze(1))).clamp_min(0)
        oy = (torch.minimum(top.unsqueeze(2), top.unsqueeze(1)) - torch.maximum(bottom.unsqueeze(2), bottom.unsqueeze(1))).clamp_min(0)
        area = ox * oy
        tri = torch.triu(torch.ones((n, n), dtype=torch.bool, device=self.device), diagonal=1)
        pair_overlap = (ox[:, tri] > 0) & (oy[:, tri] > 0)
        pair_area = area[:, tri]
        overlap_area = torch.where(pair_overlap, pair_area, torch.zeros_like(pair_area))
        count = pair_overlap.sum(dim=1)
        total = overlap_area.sum(dim=1)
        max_area = overlap_area.max(dim=1).values if overlap_area.shape[1] else torch.zeros(batch, device=self.device)
        return count, total, max_area

    def _boundary_violation(self, placements):
        half = self.sizes / 2
        left = (half[:, 0].unsqueeze(0) - placements[:, :, 0]).clamp_min(0)
        right = (placements[:, :, 0] + half[:, 0].unsqueeze(0) - self.canvas_w).clamp_min(0)
        bottom = (half[:, 1].unsqueeze(0) - placements[:, :, 1]).clamp_min(0)
        top = (placements[:, :, 1] + half[:, 1].unsqueeze(0) - self.canvas_h).clamp_min(0)
        return (left + right + bottom + top).sum(dim=1)

    def _fixed_violation(self, placements):
        if not bool(self.fixed_mask.any()):
            return torch.zeros(placements.shape[0], dtype=torch.float32, device=self.device)
        diff = (placements[:, self.fixed_mask, :] - self.original_positions[self.fixed_mask].unsqueeze(0)).abs()
        return diff.sum(dim=(1, 2))

    def _density(self, placements):
        batch = placements.shape[0]
        grid_count = self.grid_rows * self.grid_cols
        if grid_count == 0:
            return torch.zeros(batch, dtype=torch.float32, device=self.device)
        macro_boxes = self._macro_boxes(placements)
        grid = self.grid_boxes
        x_overlap = (
            torch.minimum(macro_boxes[:, :, 2].unsqueeze(2), grid[:, 2].view(1, 1, -1))
            - torch.maximum(macro_boxes[:, :, 0].unsqueeze(2), grid[:, 0].view(1, 1, -1))
        ).clamp_min(0)
        y_overlap = (
            torch.minimum(macro_boxes[:, :, 3].unsqueeze(2), grid[:, 3].view(1, 1, -1))
            - torch.maximum(macro_boxes[:, :, 1].unsqueeze(2), grid[:, 1].view(1, 1, -1))
        ).clamp_min(0)
        density = (x_overlap * y_overlap).sum(dim=1) / max(self.grid_area, 1.0e-9)
        if grid_count < 10:
            occupied = density > 0
            return 0.5 * (density.sum(dim=1) / occupied.sum(dim=1).clamp_min(1))
        k = max(int(grid_count * 0.1), 1)
        return 0.5 * torch.topk(density, k=k, dim=1).values.mean(dim=1)

    def _congestion(self, placements):
        if self.congestion_mode == "fast":
            return self._congestion_fast(placements)
        if self.congestion_mode == "none":
            return torch.zeros(placements.shape[0], dtype=torch.float32, device=self.device)

        ctx = self.ctx
        batch = placements.shape[0]
        grid_count = self.grid_rows * self.grid_cols
        if grid_count == 0 or not self.net_pin_ranges:
            return torch.zeros(batch, dtype=torch.float32, device=self.device)
        h = torch.zeros((batch, self.grid_rows, self.grid_cols), dtype=torch.float32, device=self.device)
        v = torch.zeros_like(h)
        pin_pos = self._pin_positions(
            placements,
            ctx.net_pin_parent,
            ctx.net_pin_offset,
            ctx.net_pin_port_pos,
            ctx.net_pin_is_port,
        )
        pin_col, pin_row = self._grid_location(pin_pos)

        for start, end, weight in self.net_pin_ranges:
            source = torch.stack((pin_row[:, start], pin_col[:, start]), dim=1)
            net_nodes = torch.stack((pin_row[:, start:end], pin_col[:, start:end]), dim=2)
            self._route_official_net(h, v, source, net_nodes, weight)

        grid_v_routes = max(self.grid_w * float(self.benchmark.vroutes_per_micron), 1.0e-9)
        grid_h_routes = max(self.grid_h * float(self.benchmark.hroutes_per_micron), 1.0e-9)
        h = self._smooth_h(h / grid_h_routes)
        v = self._smooth_v(v / grid_v_routes)
        v_macro, h_macro = self._macro_blockage(placements, grid_v_routes, grid_h_routes)
        combined = torch.cat([(v + v_macro).reshape(batch, -1), (h + h_macro).reshape(batch, -1)], dim=1)
        k = int(combined.shape[1] * 0.05)
        if k <= 0:
            return combined.max(dim=1).values
        return torch.topk(combined, k=k, dim=1).values.mean(dim=1)

    def _congestion_fast(self, placements):
        ctx = self.ctx
        batch = placements.shape[0]
        grid_count = self.grid_rows * self.grid_cols
        pair_count = ctx.routing_weights.numel()
        if grid_count == 0 or pair_count == 0:
            return torch.zeros(batch, dtype=torch.float32, device=self.device)

        src = self._pin_positions(
            placements,
            ctx.routing_src_parent,
            ctx.routing_src_offset,
            ctx.routing_src_port_pos,
            ctx.routing_src_is_port,
        )
        dst = self._pin_positions(
            placements,
            ctx.routing_dst_parent,
            ctx.routing_dst_offset,
            ctx.routing_dst_port_pos,
            ctx.routing_dst_is_port,
        )
        src_col, src_row = self._grid_location(src)
        dst_col, dst_row = self._grid_location(dst)

        h_diff = torch.zeros(
            (batch, self.grid_rows, self.grid_cols + 1),
            dtype=torch.float32,
            device=self.device,
        )
        v_diff = torch.zeros(
            (batch, self.grid_rows + 1, self.grid_cols),
            dtype=torch.float32,
            device=self.device,
        )
        batch_ids = torch.arange(batch, device=self.device).view(-1, 1).expand(batch, pair_count)
        weights = ctx.routing_weights.view(1, -1).expand(batch, pair_count)

        col_min = torch.minimum(src_col, dst_col)
        col_max = torch.maximum(src_col, dst_col)
        h_mask = col_max > col_min
        if bool(h_mask.any()):
            h_base = batch_ids * (self.grid_rows * (self.grid_cols + 1)) + src_row * (self.grid_cols + 1)
            h_flat = h_diff.reshape(-1)
            h_flat.scatter_add_(0, (h_base + col_min)[h_mask], weights[h_mask])
            h_flat.scatter_add_(0, (h_base + col_max)[h_mask], -weights[h_mask])

        row_min = torch.minimum(src_row, dst_row)
        row_max = torch.maximum(src_row, dst_row)
        v_mask = row_max > row_min
        if bool(v_mask.any()):
            v_base = batch_ids * ((self.grid_rows + 1) * self.grid_cols) + dst_col
            v_flat = v_diff.reshape(-1)
            v_flat.scatter_add_(0, (v_base + row_min * self.grid_cols)[v_mask], weights[v_mask])
            v_flat.scatter_add_(0, (v_base + row_max * self.grid_cols)[v_mask], -weights[v_mask])

        h = h_diff.cumsum(dim=2)[:, :, : self.grid_cols]
        v = v_diff.cumsum(dim=1)[:, : self.grid_rows, :]

        grid_v_routes = max(self.grid_w * float(self.benchmark.vroutes_per_micron), 1.0e-9)
        grid_h_routes = max(self.grid_h * float(self.benchmark.hroutes_per_micron), 1.0e-9)
        h = self._smooth_h(h / grid_h_routes)
        v = self._smooth_v(v / grid_v_routes)
        v_macro, h_macro = self._macro_blockage(placements, grid_v_routes, grid_h_routes)
        combined = torch.cat([(v + v_macro).reshape(batch, -1), (h + h_macro).reshape(batch, -1)], dim=1)
        k = int(combined.shape[1] * 0.05)
        if k <= 0:
            return combined.max(dim=1).values
        return torch.topk(combined, k=k, dim=1).values.mean(dim=1)

    def _route_official_net(self, h, v, source, net_nodes, weight: float) -> None:
        for batch_idx in range(net_nodes.shape[0]):
            source_gcell = (int(source[batch_idx, 0].item()), int(source[batch_idx, 1].item()))
            node_gcells = {
                (int(row.item()), int(col.item()))
                for row, col in net_nodes[batch_idx]
            }
            node_gcells.add(source_gcell)
            if len(node_gcells) == 2:
                self._route_two_pin(h, v, batch_idx, source_gcell, node_gcells, weight)
            elif len(node_gcells) == 3:
                self._route_three_pin(h, v, batch_idx, node_gcells, weight)
            elif len(node_gcells) > 3:
                for sink_gcell in node_gcells:
                    if sink_gcell != source_gcell:
                        self._route_two_pin(h, v, batch_idx, source_gcell, {source_gcell, sink_gcell}, weight)

    def _route_two_pin(self, h, v, batch_idx: int, source_gcell, node_gcells, weight: float) -> None:
        nodes = list(node_gcells)
        sink_gcell = nodes[1] if nodes[0] == source_gcell else nodes[0]
        row_min = min(sink_gcell[0], source_gcell[0])
        row_max = max(sink_gcell[0], source_gcell[0])
        col_min = min(sink_gcell[1], source_gcell[1])
        col_max = max(sink_gcell[1], source_gcell[1])
        if col_max > col_min:
            h[batch_idx, source_gcell[0], col_min:col_max] += weight
        if row_max > row_min:
            v[batch_idx, row_min:row_max, sink_gcell[1]] += weight

    def _route_three_pin(self, h, v, batch_idx: int, node_gcells, weight: float) -> None:
        nodes = sorted(node_gcells, key=lambda x: (x[1], x[0]))
        y1, x1 = nodes[0]
        y2, x2 = nodes[1]
        y3, x3 = nodes[2]
        if x1 < x2 and x2 < x3 and min(y1, y3) < y2 and max(y1, y3) > y2:
            self._route_l(h, v, batch_idx, nodes, weight)
        elif x2 == x3 and x1 < x2 and y1 < min(y2, y3):
            if x2 > x1:
                h[batch_idx, y1, x1:x2] += weight
            if max(y2, y3) > y1:
                v[batch_idx, y1:max(y2, y3), x2] += weight
        elif y2 == y3:
            if x2 > x1:
                h[batch_idx, y1, x1:x2] += weight
            if x3 > x2:
                h[batch_idx, y2, x2:x3] += weight
            if max(y2, y1) > min(y2, y1):
                v[batch_idx, min(y2, y1):max(y2, y1), x2] += weight
        else:
            self._route_t(h, v, batch_idx, node_gcells, weight)

    def _route_l(self, h, v, batch_idx: int, nodes, weight: float) -> None:
        nodes = sorted(nodes, key=lambda x: (x[1], x[0]))
        y1, x1 = nodes[0]
        y2, x2 = nodes[1]
        y3, x3 = nodes[2]
        if x2 > x1:
            h[batch_idx, y1, x1:x2] += weight
        if x3 > x2:
            h[batch_idx, y2, x2:x3] += weight
        if max(y1, y2) > min(y1, y2):
            v[batch_idx, min(y1, y2):max(y1, y2), x2] += weight
        if max(y2, y3) > min(y2, y3):
            v[batch_idx, min(y2, y3):max(y2, y3), x3] += weight

    def _route_t(self, h, v, batch_idx: int, node_gcells, weight: float) -> None:
        nodes = sorted(node_gcells)
        y1, x1 = nodes[0]
        y2, x2 = nodes[1]
        y3, x3 = nodes[2]
        xmin = min(x1, x2, x3)
        xmax = max(x1, x2, x3)
        if xmax > xmin:
            h[batch_idx, y2, xmin:xmax] += weight
        if max(y1, y2) > min(y1, y2):
            v[batch_idx, min(y1, y2):max(y1, y2), x1] += weight
        if max(y2, y3) > min(y2, y3):
            v[batch_idx, min(y2, y3):max(y2, y3), x3] += weight

    def _macro_blockage(self, placements, grid_v_routes: float, grid_h_routes: float):
        batch = placements.shape[0]
        if self.num_hard == 0:
            z = torch.zeros((batch, self.grid_rows, self.grid_cols), dtype=torch.float32, device=self.device)
            return z, z
        macro_boxes = self._macro_boxes(placements[:, : self.num_hard, :], sizes=self.sizes[: self.num_hard])
        grid = self.grid_boxes
        x_overlap = (
            torch.minimum(macro_boxes[:, :, 2].unsqueeze(2), grid[:, 2].view(1, 1, -1))
            - torch.maximum(macro_boxes[:, :, 0].unsqueeze(2), grid[:, 0].view(1, 1, -1))
        ).clamp_min(0)
        y_overlap = (
            torch.minimum(macro_boxes[:, :, 3].unsqueeze(2), grid[:, 3].view(1, 1, -1))
            - torch.maximum(macro_boxes[:, :, 1].unsqueeze(2), grid[:, 1].view(1, 1, -1))
        ).clamp_min(0)
        overlap_mask = (x_overlap > 0) & (y_overlap > 0)
        x_overlap = torch.where(overlap_mask, x_overlap, torch.zeros_like(x_overlap))
        y_overlap = torch.where(overlap_mask, y_overlap, torch.zeros_like(y_overlap))
        v_contrib = x_overlap * float(self.ctx.vrouting_alloc)
        h_contrib = y_overlap * float(self.ctx.hrouting_alloc)
        if self.grid_rows > 0 and self.grid_cols > 0:
            partial_v = (y_overlap > 0) & ((y_overlap - self.grid_h).abs() > 1.0e-5)
            partial_h = (x_overlap > 0) & ((x_overlap - self.grid_w).abs() > 1.0e-5)
            grid_rows = torch.arange(self.grid_rows, device=self.device).repeat_interleave(self.grid_cols)
            grid_cols = torch.arange(self.grid_cols, device=self.device).repeat(self.grid_rows)
            top_rows = torch.floor((macro_boxes[:, :, 3] / max(self.grid_h, 1.0e-9))).to(torch.long).clamp(0, self.grid_rows - 1)
            right_cols = torch.floor((macro_boxes[:, :, 2] / max(self.grid_w, 1.0e-9))).to(torch.long).clamp(0, self.grid_cols - 1)
            top_mask = grid_rows.view(1, 1, -1) == top_rows.unsqueeze(2)
            right_mask = grid_cols.view(1, 1, -1) == right_cols.unsqueeze(2)
            vertical_span = (
                torch.floor((macro_boxes[:, :, 3] / max(self.grid_h, 1.0e-9))).to(torch.long)
                != torch.floor((macro_boxes[:, :, 1] / max(self.grid_h, 1.0e-9))).to(torch.long)
            ).unsqueeze(2)
            horizontal_span = (
                torch.floor((macro_boxes[:, :, 2] / max(self.grid_w, 1.0e-9))).to(torch.long)
                != torch.floor((macro_boxes[:, :, 0] / max(self.grid_w, 1.0e-9))).to(torch.long)
            ).unsqueeze(2)
            v_contrib = torch.where(vertical_span & top_mask & partial_v, torch.zeros_like(v_contrib), v_contrib)
            h_contrib = torch.where(horizontal_span & right_mask & partial_h, torch.zeros_like(h_contrib), h_contrib)
        v = v_contrib.sum(dim=1).reshape(batch, self.grid_rows, self.grid_cols) / grid_v_routes
        h = h_contrib.sum(dim=1).reshape(batch, self.grid_rows, self.grid_cols) / grid_h_routes
        return v, h

    def _smooth_v(self, v):
        r = max(int(self.ctx.smooth_range), 0)
        if r == 0:
            return v
        out = torch.zeros_like(v)
        for col in range(self.grid_cols):
            lp = max(0, col - r)
            rp = min(self.grid_cols - 1, col + r)
            out[:, :, lp : rp + 1] += v[:, :, col : col + 1] / float(rp - lp + 1)
        return out

    def _smooth_h(self, h):
        r = max(int(self.ctx.smooth_range), 0)
        if r == 0:
            return h
        out = torch.zeros_like(h)
        for row in range(self.grid_rows):
            lp = max(0, row - r)
            rp = min(self.grid_rows - 1, row + r)
            out[:, lp : rp + 1, :] += h[:, row : row + 1, :] / float(rp - lp + 1)
        return out

    def _grid_location(self, xy):
        col = torch.floor(xy[:, :, 0] / max(self.grid_w, 1.0e-9)).to(torch.long).clamp(0, self.grid_cols - 1)
        row = torch.floor(xy[:, :, 1] / max(self.grid_h, 1.0e-9)).to(torch.long).clamp(0, self.grid_rows - 1)
        return col, row

    def _macro_boxes(self, placements, sizes=None):
        if sizes is None:
            sizes = self.sizes
        half = sizes / 2
        return torch.stack(
            [
                placements[:, :, 0] - half[:, 0].unsqueeze(0),
                placements[:, :, 1] - half[:, 1].unsqueeze(0),
                placements[:, :, 0] + half[:, 0].unsqueeze(0),
                placements[:, :, 1] + half[:, 1].unsqueeze(0),
            ],
            dim=2,
        )

    def _make_grid_boxes(self):
        boxes = []
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                boxes.append((col * self.grid_w, row * self.grid_h, (col + 1) * self.grid_w, (row + 1) * self.grid_h))
        return torch.tensor(boxes, dtype=torch.float32, device=self.device)

    def _make_net_pin_ranges(self):
        ids = self.ctx.net_pin_net_id.detach().cpu().tolist()
        ranges = []
        start = 0
        while start < len(ids):
            net_id = ids[start]
            end = start + 1
            while end < len(ids) and ids[end] == net_id:
                end += 1
            if end - start >= 2:
                weight = float(self.ctx.net_weights[net_id].detach().cpu().item())
                ranges.append((start, end, weight))
            start = end
        return ranges
