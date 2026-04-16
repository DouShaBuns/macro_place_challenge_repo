from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from benchmark_context import build_benchmark_context  # noqa: E402
from legalize import clamp_placement, legalize_initial  # noqa: E402
from parallel_runner import _load_benchmark  # noqa: E402
from sa_optimizer import SAConfig, SAOptimizer  # noqa: E402
from torch_objective import TorchProxyCostEvaluator  # noqa: E402

from macro_place.objective import compute_proxy_cost  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare torch search proxy against official proxy cost.")
    parser.add_argument("--benchmarks", nargs="+", default=["ibm01", "ibm10", "ibm17"])
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--candidate-batch", type=int, default=16)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    summaries = []
    for name in args.benchmarks:
        summary = calibrate_one(
            name=name,
            samples=max(1, int(args.samples)),
            candidate_batch=max(1, int(args.candidate_batch)),
            steps=max(1, int(args.steps)),
            device=device,
        )
        summaries.append(summary)
        print_summary(summary)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for summary in summaries:
                f.write(json.dumps(summary, sort_keys=True) + "\n")
        print(f"Wrote {out}")


def calibrate_one(name: str, samples: int, candidate_batch: int, steps: int, device: str) -> dict:
    benchmark, plc = _load_benchmark(name)
    torch_device = torch.device(device)
    ctx = build_benchmark_context(benchmark, torch_device)
    evaluator = TorchProxyCostEvaluator(ctx)

    config = SAConfig(seeds=(1,), iters=max(steps, 2), candidate_batch=candidate_batch)
    optimizer = SAOptimizer(config, device=torch_device)

    init = legalize_initial(benchmark).to(torch_device, dtype=torch.float32)
    init = clamp_placement(init, benchmark).to(torch_device)
    state = init.unsqueeze(0)
    placements = [init.detach().cpu()]

    for step in range(steps):
        temp = optimizer._temperature(step)
        candidates = optimizer._generate_candidates(state, benchmark, config.seeds, step, temp)
        flat = candidates.reshape(-1, benchmark.num_macros, 2)
        placements.extend(flat.detach().cpu())
        costs = evaluator.evaluate_batch(flat)
        best_idx = int(torch.argmin(costs.search_score).item())
        state = flat[best_idx].view(1, benchmark.num_macros, 2)
        if len(placements) >= samples:
            break

    placements = torch.stack(placements[:samples], dim=0)
    torch_cost = evaluator.evaluate_batch(placements.to(torch_device))
    official_rows = [compute_proxy_cost(placement, benchmark, plc) for placement in placements]

    torch_wl = torch_cost.wirelength_cost.detach().cpu().tolist()
    torch_den = torch_cost.density_cost.detach().cpu().tolist()
    torch_cong = torch_cost.congestion_cost.detach().cpu().tolist()
    torch_current = torch_cost.official_proxy.detach().cpu().tolist()
    torch_unweighted_components = [
        float(w + d + c) for w, d, c in zip(torch_wl, torch_den, torch_cong)
    ]

    official_wl = [float(row["wirelength_cost"]) for row in official_rows]
    official_den = [float(row["density_cost"]) for row in official_rows]
    official_cong = [float(row["congestion_cost"]) for row in official_rows]
    official_proxy = [float(row["proxy_cost"]) for row in official_rows]

    return {
        "name": name,
        "samples": len(placements),
        "device": str(device),
        "torch_legal_count": int(torch_cost.is_legal.detach().cpu().sum().item()),
        "official_zero_overlap_count": sum(1 for row in official_rows if int(row["overlap_count"]) == 0),
        "ratios": {
            "wirelength_torch_over_official_median": median_ratio(torch_wl, official_wl),
            "density_torch_over_official_median": median_ratio(torch_den, official_den),
            "congestion_torch_over_official_median": median_ratio(torch_cong, official_cong),
        },
        "pearson": {
            "current_torch_proxy_vs_official": pearson(torch_current, official_proxy),
            "unweighted_torch_components_vs_official": pearson(torch_unweighted_components, official_proxy),
            "wirelength": pearson(torch_wl, official_wl),
            "density": pearson(torch_den, official_den),
            "congestion": pearson(torch_cong, official_cong),
        },
        "spearman": {
            "current_torch_proxy_vs_official": spearman(torch_current, official_proxy),
            "unweighted_torch_components_vs_official": spearman(torch_unweighted_components, official_proxy),
            "wirelength": spearman(torch_wl, official_wl),
            "density": spearman(torch_den, official_den),
            "congestion": spearman(torch_cong, official_cong),
        },
        "means": {
            "torch_wirelength": mean(torch_wl),
            "official_wirelength": mean(official_wl),
            "torch_density": mean(torch_den),
            "official_density": mean(official_den),
            "torch_congestion": mean(torch_cong),
            "official_congestion": mean(official_cong),
            "torch_current_proxy": mean(torch_current),
            "torch_unweighted_components": mean(torch_unweighted_components),
            "official_proxy": mean(official_proxy),
        },
    }


def print_summary(summary: dict) -> None:
    name = summary["name"]
    print(f"\n{name}: samples={summary['samples']} device={summary['device']}")
    print(
        "  legality: "
        f"torch_legal={summary['torch_legal_count']} "
        f"official_zero_overlap={summary['official_zero_overlap_count']}"
    )
    print("  median torch/official ratios:")
    for key, value in summary["ratios"].items():
        print(f"    {key}: {value:.4f}")
    print("  spearman rank corr:")
    for key, value in summary["spearman"].items():
        print(f"    {key}: {value:.4f}")
    print("  pearson corr:")
    for key, value in summary["pearson"].items():
        print(f"    {key}: {value:.4f}")
    print("  means:")
    for key, value in summary["means"].items():
        print(f"    {key}: {value:.4f}")


def median_ratio(xs: list[float], ys: list[float]) -> float:
    ratios = [float(x) / float(y) for x, y in zip(xs, ys) if abs(float(y)) > 1.0e-12]
    if not ratios:
        return float("nan")
    ratios.sort()
    mid = len(ratios) // 2
    if len(ratios) % 2:
        return ratios[mid]
    return 0.5 * (ratios[mid - 1] + ratios[mid])


def mean(values: list[float]) -> float:
    return sum(float(v) for v in values) / max(len(values), 1)


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    mx = mean(xs)
    my = mean(ys)
    dx = [float(x) - mx for x in xs]
    dy = [float(y) - my for y in ys]
    denom = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    if denom <= 1.0e-12:
        return float("nan")
    return sum(x * y for x, y in zip(dx, dy)) / denom


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(ranks(xs), ranks(ys))


def ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(float(v) for v in values), key=lambda item: item[1])
    out = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        rank = 0.5 * (i + j - 1)
        for k in range(i, j):
            out[indexed[k][0]] = rank
        i = j
    return out


if __name__ == "__main__":
    main()
