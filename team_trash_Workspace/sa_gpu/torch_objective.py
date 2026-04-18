from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

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
    ):
        self.ctx = context
        self.benchmark = context.benchmark
        self.device = context.device
        self.overlap_weight = float(overlap_weight)
        self.boundary_weight = float(boundary_weight)
        self.gap = float(gap)
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
        pair_area = area[:, tri]
        count = (pair_area > self.gap).sum(dim=1)
        total = pair_area.sum(dim=1)
        max_area = pair_area.max(dim=1).values if pair_area.shape[1] else torch.zeros(batch, device=self.device)
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

        cols = torch.arange(self.grid_cols, device=self.device).view(1, 1, self.grid_cols)
        rows = torch.arange(self.grid_rows, device=self.device).view(1, 1, self.grid_rows)
        h = torch.zeros((batch, self.grid_rows, self.grid_cols), dtype=torch.float32, device=self.device)
        v = torch.zeros_like(h)
        chunk_size = 512
        for start in range(0, pair_count, chunk_size):
            end = min(start + chunk_size, pair_count)
            src_col_c = src_col[:, start:end]
            dst_col_c = dst_col[:, start:end]
            src_row_c = src_row[:, start:end]
            dst_row_c = dst_row[:, start:end]
            col_min = torch.minimum(src_col_c, dst_col_c).unsqueeze(-1)
            col_max = torch.maximum(src_col_c, dst_col_c).unsqueeze(-1)
            row_min = torch.minimum(src_row_c, dst_row_c).unsqueeze(-1)
            row_max = torch.maximum(src_row_c, dst_row_c).unsqueeze(-1)
            h_cols = (cols >= col_min) & (cols < col_max)
            v_rows = (rows >= row_min) & (rows < row_max)

            weights = ctx.routing_weights[start:end].view(1, end - start, 1, 1)
            h_row_onehot = F.one_hot(src_row_c, num_classes=self.grid_rows).to(torch.float32)
            v_col_onehot = F.one_hot(dst_col_c, num_classes=self.grid_cols).to(torch.float32)
            h += (h_row_onehot.unsqueeze(-1) * h_cols.unsqueeze(2).to(torch.float32) * weights).sum(dim=1)
            v += (v_rows.unsqueeze(-1).to(torch.float32) * v_col_onehot.unsqueeze(2) * weights).sum(dim=1)

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
        intersects = (x_overlap > 0) & (y_overlap > 0)
        row_ids = torch.arange(self.grid_rows, device=self.device).repeat_interleave(self.grid_cols)
        col_ids = torch.arange(self.grid_cols, device=self.device).repeat(self.grid_rows)
        left = macro_boxes[:, :, 0]
        right = macro_boxes[:, :, 2]
        bottom = macro_boxes[:, :, 1]
        top = macro_boxes[:, :, 3]
        bl_col = torch.floor(left / max(self.grid_w, 1.0e-9)).to(torch.long).clamp(0, self.grid_cols - 1)
        ur_col = torch.floor(right / max(self.grid_w, 1.0e-9)).to(torch.long).clamp(0, self.grid_cols - 1)
        bl_row = torch.floor(bottom / max(self.grid_h, 1.0e-9)).to(torch.long).clamp(0, self.grid_rows - 1)
        ur_row = torch.floor(top / max(self.grid_h, 1.0e-9)).to(torch.long).clamp(0, self.grid_rows - 1)
        bottom_partial = y_overlap.gather(2, (bl_row * self.grid_cols + bl_col).unsqueeze(2)).squeeze(2)
        top_partial = y_overlap.gather(2, (ur_row * self.grid_cols + bl_col).unsqueeze(2)).squeeze(2)
        left_partial = x_overlap.gather(2, (bl_row * self.grid_cols + bl_col).unsqueeze(2)).squeeze(2)
        right_partial = x_overlap.gather(2, (bl_row * self.grid_cols + ur_col).unsqueeze(2)).squeeze(2)
        partial_v = (ur_row != bl_row) & (
            ((bottom_partial - self.grid_h).abs() > 1.0e-5)
            | ((top_partial - self.grid_h).abs() > 1.0e-5)
        )
        partial_h = (ur_col != bl_col) & (
            ((left_partial - self.grid_w).abs() > 1.0e-5)
            | ((right_partial - self.grid_w).abs() > 1.0e-5)
        )
        v_keep = intersects & ~(partial_v.unsqueeze(2) & (row_ids.view(1, 1, -1) == ur_row.unsqueeze(2)))
        h_keep = intersects & ~(partial_h.unsqueeze(2) & (col_ids.view(1, 1, -1) == ur_col.unsqueeze(2)))
        v = (
            (x_overlap * v_keep.to(x_overlap.dtype) * float(self.ctx.vrouting_alloc))
            .sum(dim=1)
            .reshape(batch, self.grid_rows, self.grid_cols)
            / grid_v_routes
        )
        h = (
            (y_overlap * h_keep.to(y_overlap.dtype) * float(self.ctx.hrouting_alloc))
            .sum(dim=1)
            .reshape(batch, self.grid_rows, self.grid_cols)
            / grid_h_routes
        )
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
