#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from macro_place.benchmark import Benchmark
from macro_place.loader import load_benchmark_from_dir
from macro_place.utils import visualize_placement

try:
    from team_trash_Workspace.dreamplace_gpu.benchmark_context import NG45_BENCHMARK_DIRS
except Exception:
    NG45_BENCHMARK_DIRS = {}


def _load_benchmark_pt(name: str) -> Benchmark:
    path = REPO_ROOT / "benchmarks" / "processed" / "public" / f"{name}.pt"
    if not path.exists():
        raise FileNotFoundError(f"benchmark tensor not found: {path}")
    return Benchmark.load(str(path))


def _default_netlist_dir(name: str) -> Path | None:
    text = NG45_BENCHMARK_DIRS.get(name)
    if text is None:
        return None
    path = REPO_ROOT / text
    return path if path.exists() else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a saved placement tensor to PNG.")
    parser.add_argument("--placement", type=Path, required=True, help="Path to saved placement tensor [.pt].")
    parser.add_argument("--benchmark", required=True, help="Benchmark name, e.g. ariane136_ng45.")
    parser.add_argument("--out", type=Path, default=None, help="PNG output path.")
    parser.add_argument(
        "--netlist-dir",
        type=Path,
        default=None,
        help="Optional netlist/output_CT_Grouping directory for richer density/congestion overlays.",
    )
    args = parser.parse_args()

    placement_path = args.placement.resolve()
    if not placement_path.exists():
        raise FileNotFoundError(f"placement tensor not found: {placement_path}")

    benchmark = _load_benchmark_pt(args.benchmark)
    placement = torch.load(placement_path, map_location="cpu", weights_only=True)
    if placement.shape != benchmark.macro_positions.shape:
        raise ValueError(
            f"placement shape {tuple(placement.shape)} does not match benchmark {tuple(benchmark.macro_positions.shape)}"
        )

    netlist_dir = args.netlist_dir.resolve() if args.netlist_dir else _default_netlist_dir(args.benchmark)
    plc = None
    if netlist_dir is not None and netlist_dir.exists():
        _, plc = load_benchmark_from_dir(netlist_dir.as_posix())

    out_path = args.out or (REPO_ROOT / "output" / "rendered_placements" / f"{args.benchmark}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    visualize_placement(placement, benchmark, save_path=str(out_path), plc=plc)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
