from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from placer import DreamPlaceGPUPlacer  # noqa: E402

from macro_place.evaluate import IBM_BENCHMARKS, NG45_BENCHMARKS  # noqa: E402
from macro_place.loader import load_benchmark, load_benchmark_from_dir  # noqa: E402
from macro_place.objective import compute_proxy_cost  # noqa: E402
from macro_place.utils import validate_placement  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="DREAMPlace-style analytical placement runner.")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--ng45", action="store_true")
    parser.add_argument("--benchmarks", nargs="*", default=None)
    parser.add_argument("--out", default="team_trash_Workspace/dreamplace_gpu/results/latest.jsonl")
    args = parser.parse_args()

    if args.benchmarks:
        benchmarks = args.benchmarks
    elif args.ng45:
        benchmarks = list(NG45_BENCHMARKS.keys())
    elif args.all:
        benchmarks = IBM_BENCHMARKS
    else:
        benchmarks = ["ibm01"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with out.open("w", encoding="utf-8") as f:
        for name in benchmarks:
            try:
                result = _run_one(name)
            except Exception:
                result = {"name": name, "error": traceback.format_exc(), "runtime": 0.0}
            results.append(result)
            f.write(json.dumps(result, sort_keys=True) + "\n")
            f.flush()
            _print_result(result)
    results.sort(key=lambda row: row["name"])
    with out.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, sort_keys=True) + "\n")
    print(f"Wrote {out}")


def _run_one(name: str) -> dict:
    benchmark, plc = _load_benchmark(name)
    placer = DreamPlaceGPUPlacer()
    start = time.time()
    placement = placer.place(benchmark)
    runtime = time.time() - start
    valid, violations = validate_placement(placement, benchmark)
    costs = compute_proxy_cost(placement, benchmark, plc)
    return {
        "name": name,
        "proxy_cost": float(costs["proxy_cost"]),
        "wirelength": float(costs["wirelength_cost"]),
        "density": float(costs["density_cost"]),
        "congestion": float(costs["congestion_cost"]),
        "overlaps": int(costs["overlap_count"]),
        "valid": bool(valid),
        "violations": violations,
        "runtime": float(runtime),
    }


def _load_benchmark(name: str):
    if name in NG45_BENCHMARKS:
        base = NG45_BENCHMARKS[name]
        return load_benchmark(f"{base}/netlist.pb.txt", f"{base}/initial.plc", name=name)
    return load_benchmark_from_dir(f"external/MacroPlacement/Testcases/ICCAD04/{name}")


def _print_result(result: dict) -> None:
    if "error" in result:
        print(f"{result['name']:>13} FAILED {result['error'].splitlines()[-1]}")
        return
    status = "VALID" if result["valid"] and result["overlaps"] == 0 else "INVALID"
    print(
        f"{result['name']:>13} proxy={result['proxy_cost']:.4f} "
        f"wl={result['wirelength']:.3f} den={result['density']:.3f} "
        f"cong={result['congestion']:.3f} overlaps={result['overlaps']} "
        f"{status} [{result['runtime']:.2f}s]"
    )


if __name__ == "__main__":
    main()
