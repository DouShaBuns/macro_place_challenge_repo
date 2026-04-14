from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from macro_place.benchmark import Benchmark
from macro_place.loader import load_benchmark, load_benchmark_from_dir


NG45_BENCHMARK_DIRS = {
    "ariane133": "external/MacroPlacement/Flows/NanGate45/ariane133/netlist/output_CT_Grouping",
    "ariane133_ng45": "external/MacroPlacement/Flows/NanGate45/ariane133/netlist/output_CT_Grouping",
    "ariane136": "external/MacroPlacement/Flows/NanGate45/ariane136/netlist/output_CT_Grouping",
    "ariane136_ng45": "external/MacroPlacement/Flows/NanGate45/ariane136/netlist/output_CT_Grouping",
    "mempool_tile": "external/MacroPlacement/Flows/NanGate45/mempool_tile/netlist/output_CT_Grouping",
    "mempool_tile_ng45": "external/MacroPlacement/Flows/NanGate45/mempool_tile/netlist/output_CT_Grouping",
    "nvdla": "external/MacroPlacement/Flows/NanGate45/nvdla/netlist/output_CT_Grouping",
    "nvdla_ng45": "external/MacroPlacement/Flows/NanGate45/nvdla/netlist/output_CT_Grouping",
}


@dataclass
class BenchmarkContext:
    benchmark: Benchmark
    device: torch.device
    net_pin_parent: torch.Tensor
    net_pin_offset: torch.Tensor
    net_pin_port_pos: torch.Tensor
    net_pin_is_port: torch.Tensor
    net_pin_net_id: torch.Tensor
    net_weights: torch.Tensor
    routing_src_parent: torch.Tensor
    routing_src_offset: torch.Tensor
    routing_src_port_pos: torch.Tensor
    routing_src_is_port: torch.Tensor
    routing_dst_parent: torch.Tensor
    routing_dst_offset: torch.Tensor
    routing_dst_port_pos: torch.Tensor
    routing_dst_is_port: torch.Tensor
    routing_weights: torch.Tensor
    num_nets: int
    wirelength_norm_net_count: float
    smooth_range: int
    hrouting_alloc: float
    vrouting_alloc: float


def load_plc_for_benchmark(name: str):
    ibm_dir = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if ibm_dir.exists():
        _, plc = load_benchmark_from_dir(ibm_dir.as_posix())
        return plc

    ng45_dir = NG45_BENCHMARK_DIRS.get(name)
    if ng45_dir is not None:
        netlist = Path(ng45_dir) / "netlist.pb.txt"
        initial = Path(ng45_dir) / "initial.plc"
        if netlist.exists():
            _, plc = load_benchmark(netlist.as_posix(), initial.as_posix(), name=name)
            return plc

    return None


def build_benchmark_context(
    benchmark: Benchmark, device: torch.device | str | None = None
) -> BenchmarkContext:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    plc = load_plc_for_benchmark(benchmark.name)
    if plc is None:
        return _build_context_from_benchmark_nets(benchmark, device)

    plc_macro_to_bench: Dict[int, int] = {}
    for bench_idx, plc_idx in enumerate(benchmark.hard_macro_indices):
        plc_macro_to_bench[int(plc_idx)] = bench_idx
    for soft_offset, plc_idx in enumerate(benchmark.soft_macro_indices):
        plc_macro_to_bench[int(plc_idx)] = benchmark.num_hard_macros + soft_offset

    macro_name_to_bench: Dict[str, int] = {}
    for plc_idx, bench_idx in plc_macro_to_bench.items():
        macro_name_to_bench[plc.modules_w_pins[plc_idx].get_name()] = bench_idx

    def pin_record(pin_name: str) -> Tuple[int, Tuple[float, float], Tuple[float, float], bool]:
        pin_idx = plc.mod_name_to_indices[pin_name]
        pin = plc.modules_w_pins[pin_idx]
        if pin.get_type() == "PORT":
            x, y = pin.get_pos()
            return -1, (0.0, 0.0), (float(x), float(y)), True

        parent_name = pin_name.split("/")[0]
        if hasattr(pin, "get_macro_name"):
            parent_name = pin.get_macro_name()
        parent_bench_idx = macro_name_to_bench.get(parent_name, -1)
        if hasattr(pin, "get_offset"):
            ox, oy = pin.get_offset()
        else:
            ox, oy = getattr(pin, "x_offset", 0.0), getattr(pin, "y_offset", 0.0)
        return parent_bench_idx, (float(ox), float(oy)), (0.0, 0.0), False

    net_pin_parent: List[int] = []
    net_pin_offset: List[Tuple[float, float]] = []
    net_pin_port_pos: List[Tuple[float, float]] = []
    net_pin_is_port: List[bool] = []
    net_pin_net_id: List[int] = []
    net_weights: List[float] = []
    routing_src_parent: List[int] = []
    routing_src_offset: List[Tuple[float, float]] = []
    routing_src_port_pos: List[Tuple[float, float]] = []
    routing_src_is_port: List[bool] = []
    routing_dst_parent: List[int] = []
    routing_dst_offset: List[Tuple[float, float]] = []
    routing_dst_port_pos: List[Tuple[float, float]] = []
    routing_dst_is_port: List[bool] = []
    routing_weights: List[float] = []

    for net_id, (driver_name, sink_names) in enumerate(plc.nets.items()):
        driver_idx = plc.mod_name_to_indices[driver_name]
        driver = plc.modules_w_pins[driver_idx]
        weight = float(driver.get_weight() if hasattr(driver, "get_weight") else 1.0)
        net_weights.append(weight)

        records = [pin_record(pin_name) for pin_name in [driver_name] + list(sink_names)]
        for parent, offset, port_pos, is_port in records:
            if parent < 0 and not is_port:
                continue
            net_pin_parent.append(parent)
            net_pin_offset.append(offset)
            net_pin_port_pos.append(port_pos)
            net_pin_is_port.append(is_port)
            net_pin_net_id.append(net_id)

        src = records[0]
        for dst in records[1:]:
            if (src[0] < 0 and not src[3]) or (dst[0] < 0 and not dst[3]):
                continue
            routing_src_parent.append(src[0])
            routing_src_offset.append(src[1])
            routing_src_port_pos.append(src[2])
            routing_src_is_port.append(src[3])
            routing_dst_parent.append(dst[0])
            routing_dst_offset.append(dst[1])
            routing_dst_port_pos.append(dst[2])
            routing_dst_is_port.append(dst[3])
            routing_weights.append(weight)

    hrouting_alloc, vrouting_alloc = 0.0, 0.0
    if hasattr(plc, "get_macro_routing_allocation"):
        hrouting_alloc, vrouting_alloc = plc.get_macro_routing_allocation()

    return BenchmarkContext(
        benchmark=benchmark,
        device=device,
        net_pin_parent=_long(net_pin_parent, device),
        net_pin_offset=_float2(net_pin_offset, device),
        net_pin_port_pos=_float2(net_pin_port_pos, device),
        net_pin_is_port=_bool(net_pin_is_port, device),
        net_pin_net_id=_long(net_pin_net_id, device),
        net_weights=torch.tensor(net_weights, dtype=torch.float32, device=device),
        routing_src_parent=_long(routing_src_parent, device),
        routing_src_offset=_float2(routing_src_offset, device),
        routing_src_port_pos=_float2(routing_src_port_pos, device),
        routing_src_is_port=_bool(routing_src_is_port, device),
        routing_dst_parent=_long(routing_dst_parent, device),
        routing_dst_offset=_float2(routing_dst_offset, device),
        routing_dst_port_pos=_float2(routing_dst_port_pos, device),
        routing_dst_is_port=_bool(routing_dst_is_port, device),
        routing_weights=torch.tensor(routing_weights, dtype=torch.float32, device=device),
        num_nets=len(net_weights),
        wirelength_norm_net_count=float(getattr(plc, "net_cnt", len(net_weights)) or len(net_weights) or 1),
        smooth_range=int(getattr(plc, "smooth_range", 0)),
        hrouting_alloc=float(hrouting_alloc),
        vrouting_alloc=float(vrouting_alloc),
    )


def _build_context_from_benchmark_nets(benchmark: Benchmark, device: torch.device) -> BenchmarkContext:
    parents, offsets, ports, is_ports, net_ids, weights = [], [], [], [], [], []
    src_parent, src_offset, src_port, src_is_port = [], [], [], []
    dst_parent, dst_offset, dst_port, dst_is_port, route_weights = [], [], [], [], []

    for net_id, nodes in enumerate(benchmark.net_nodes):
        if nodes.numel() == 0:
            continue
        weight = float(benchmark.net_weights[net_id].item()) if net_id < benchmark.net_weights.numel() else 1.0
        weights.append(weight)
        node_list = [int(x) for x in nodes.tolist()]
        for node in node_list:
            if node < benchmark.num_macros:
                parents.append(node)
                offsets.append((0.0, 0.0))
                ports.append((0.0, 0.0))
                is_ports.append(False)
                net_ids.append(net_id)
        src = node_list[0]
        if src >= benchmark.num_macros:
            continue
        for dst in node_list[1:]:
            if dst >= benchmark.num_macros:
                continue
            src_parent.append(src)
            src_offset.append((0.0, 0.0))
            src_port.append((0.0, 0.0))
            src_is_port.append(False)
            dst_parent.append(dst)
            dst_offset.append((0.0, 0.0))
            dst_port.append((0.0, 0.0))
            dst_is_port.append(False)
            route_weights.append(weight)

    return BenchmarkContext(
        benchmark=benchmark,
        device=device,
        net_pin_parent=_long(parents, device),
        net_pin_offset=_float2(offsets, device),
        net_pin_port_pos=_float2(ports, device),
        net_pin_is_port=_bool(is_ports, device),
        net_pin_net_id=_long(net_ids, device),
        net_weights=torch.tensor(weights, dtype=torch.float32, device=device),
        routing_src_parent=_long(src_parent, device),
        routing_src_offset=_float2(src_offset, device),
        routing_src_port_pos=_float2(src_port, device),
        routing_src_is_port=_bool(src_is_port, device),
        routing_dst_parent=_long(dst_parent, device),
        routing_dst_offset=_float2(dst_offset, device),
        routing_dst_port_pos=_float2(dst_port, device),
        routing_dst_is_port=_bool(dst_is_port, device),
        routing_weights=torch.tensor(route_weights, dtype=torch.float32, device=device),
        num_nets=len(weights),
        wirelength_norm_net_count=float(max(len(weights), 1)),
        smooth_range=0,
        hrouting_alloc=0.0,
        vrouting_alloc=0.0,
    )


def _long(values: List[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.long, device=device)


def _float2(values: List[Tuple[float, float]], device: torch.device) -> torch.Tensor:
    if not values:
        return torch.zeros((0, 2), dtype=torch.float32, device=device)
    return torch.tensor(values, dtype=torch.float32, device=device)


def _bool(values: List[bool], device: torch.device) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.bool, device=device)
