from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from placer import SAGPUPlacer  # noqa: E402

from macro_place.evaluate import IBM_BENCHMARKS, NG45_BENCHMARKS  # noqa: E402
from macro_place.loader import load_benchmark, load_benchmark_from_dir  # noqa: E402
from macro_place.objective import compute_proxy_cost  # noqa: E402
from macro_place.utils import validate_placement  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel GPU batched SA runner.")
    parser.add_argument("--all", action="store_true", help="Run all IBM benchmarks.")
    parser.add_argument("--ng45", action="store_true", help="Run NG45 benchmarks.")
    parser.add_argument("--benchmarks", nargs="*", default=None, help="Explicit benchmark names.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45])
    parser.add_argument("--candidate-batch", type=int, default=16)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--devices", nargs="*", default=None, help="Devices, e.g. cuda:0 cuda:1 or cpu.")
    parser.add_argument("--warm-start-path", default=None, help="Optional torch-saved placement used as the SA start.")
    parser.add_argument("--out", default="team_trash_Workspace/sa_gpu/results/latest.jsonl")
    args = parser.parse_args()
    if args.benchmarks:
        benchmarks = args.benchmarks
    elif args.ng45:
        benchmarks = list(NG45_BENCHMARKS.keys())
    elif args.all:
        benchmarks = IBM_BENCHMARKS
    else:
        benchmarks = ["ibm01"]

    devices = args.devices or ["cuda:0"]
    if devices == ["cuda:0"]:
        try:
            import torch

            if not torch.cuda.is_available():
                devices = ["cpu"]
        except Exception:
            devices = ["cpu"]

    jobs = [
        {
            "name": name,
            "seeds": tuple(args.seeds),
            "candidate_batch": args.candidate_batch,
            "iters": args.iters,
            "device": devices[idx % len(devices)],
            "warm_start_path": args.warm_start_path,
        }
        for idx, name in enumerate(benchmarks)
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with out.open("w", encoding="utf-8") as f:
        if max(1, int(args.workers)) == 1:
            for job in jobs:
                result = _run_one_safe(job)
                results.append(result)
                f.write(json.dumps(result, sort_keys=True) + "\n")
                f.flush()
                _print_result(result)
        else:
            with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
                futures = {pool.submit(_run_one, job): job for job in jobs}
                for fut in as_completed(futures):
                    job = futures[fut]
                    try:
                        result = fut.result()
                    except Exception:
                        result = _error_result(job, traceback.format_exc())
                    results.append(result)
                    f.write(json.dumps(result, sort_keys=True) + "\n")
                    f.flush()
                    _print_result(result)
    results.sort(key=lambda item: item["name"])
    with out.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, sort_keys=True) + "\n")
    print(f"Wrote {out}")


def _print_result(result: dict) -> None:
    if "error" in result:
        print(
            f"{result['name']:>13} FAILED [{result['runtime']:.2f}s] "
            f"device={result['device']} error={result['error'].splitlines()[-1]}"
        )
        return
    status = "VALID" if result["valid"] and result["overlaps"] == 0 else "INVALID"
    print(
        f"{result['name']:>13} proxy={result['proxy_cost']:.4f} "
        f"wl={result['wirelength']:.3f} den={result['density']:.3f} "
        f"cong={result['congestion']:.3f} overlaps={result['overlaps']} "
        f"{status} [{result['runtime']:.2f}s] device={result['device']}"
    )


def _run_one_safe(job: dict) -> dict:
    try:
        return _run_one(job)
    except Exception:
        return _error_result(job, traceback.format_exc())


def _error_result(job: dict, error: str) -> dict:
    return {
        "name": job["name"],
        "error": error,
        "runtime": 0.0,
        "device": str(job["device"]),
        "seeds": list(job["seeds"]),
        "iters": int(job["iters"]),
        "candidate_batch": int(job["candidate_batch"]),
        "warm_start_path": job.get("warm_start_path"),
    }


def _run_one(job: dict) -> dict:
    benchmark, plc = _load_benchmark(job["name"])
    placer = SAGPUPlacer(
        seeds=tuple(job["seeds"]),
        iters=int(job["iters"]),
        candidate_batch=int(job["candidate_batch"]),
        device=str(job["device"]),
        warm_start_path=job.get("warm_start_path"),
    )
    start = time.time()
    placement = placer.place(benchmark)
    runtime = time.time() - start
    valid, violations = validate_placement(placement, benchmark)
    costs = compute_proxy_cost(placement, benchmark, plc)
    return {
        "name": job["name"],
        "proxy_cost": float(costs["proxy_cost"]),
        "wirelength": float(costs["wirelength_cost"]),
        "density": float(costs["density_cost"]),
        "congestion": float(costs["congestion_cost"]),
        "overlaps": int(costs["overlap_count"]),
        "valid": bool(valid),
        "violations": violations,
        "runtime": float(runtime),
        "device": str(job["device"]),
        "seeds": list(job["seeds"]),
        "iters": int(job["iters"]),
        "candidate_batch": int(job["candidate_batch"]),
        "warm_start_path": job.get("warm_start_path"),
    }


def _load_benchmark(name: str):
    if name in NG45_BENCHMARKS:
        base = NG45_BENCHMARKS[name]
        return load_benchmark(f"{base}/netlist.pb.txt", f"{base}/initial.plc", name=name)
    return load_benchmark_from_dir(f"external/MacroPlacement/Testcases/ICCAD04/{name}")


if __name__ == "__main__":
    main()
