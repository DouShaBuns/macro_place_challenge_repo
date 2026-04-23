from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from optimizer import PauseRequested  # noqa: E402
from placer import DreamPlaceGPUPlacer  # noqa: E402

from macro_place.evaluate import IBM_BENCHMARKS  # noqa: E402
from macro_place.loader import load_benchmark, load_benchmark_from_dir  # noqa: E402
from macro_place.objective import compute_proxy_cost  # noqa: E402
from macro_place.utils import validate_placement  # noqa: E402
from benchmark_context import NG45_BENCHMARK_DIRS  # noqa: E402


NG45_BENCHMARKS = {name: path for name, path in NG45_BENCHMARK_DIRS.items() if name.endswith("_ng45")}

_ORFS_MODULE = None


def _heartbeat_path() -> Path | None:
    text = os.getenv("DP_STAGE_HEARTBEAT_PATH")
    return Path(text) if text else None


def _heartbeat_update(**fields) -> None:
    path = _heartbeat_path()
    if path is None:
        return
    payload = {"pid": os.getpid(), "updated_at": time.time()}
    if path.exists():
        try:
            payload.update(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            pass
    payload.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _repo_root() -> Path:
    return _HERE.parents[1]


def _load_orfs_module():
    global _ORFS_MODULE
    if _ORFS_MODULE is not None:
        return _ORFS_MODULE
    path = _repo_root() / "scripts" / "evaluate_with_orfs.py"
    spec = importlib.util.spec_from_file_location("dreamplace_orfs_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load ORFS helper from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _ORFS_MODULE = module
    return module


def _orfs_enabled() -> bool:
    return _env_bool("DP_RUNNER_ORFS_VALIDATE", False)


def _set_orfs_env(args) -> None:
    os.environ["DP_RUNNER_ORFS_VALIDATE"] = "1" if args.orfs_validate else "0"
    os.environ["DP_RUNNER_ORFS_ROOT"] = str(Path(args.orfs_root))
    os.environ["DP_RUNNER_ORFS_OUTPUT_DIR"] = str(Path(args.orfs_output_dir))
    os.environ["DP_RUNNER_ORFS_NO_DOCKER"] = "1" if args.orfs_no_docker else "0"
    os.environ["DP_RUNNER_ORFS_SKIP_SYNTHESIS"] = "1" if args.orfs_skip_synthesis else "0"


def _orfs_benchmark_name(name: str) -> str:
    if name in NG45_BENCHMARKS and "_ng45" not in name and "_asap7" not in name:
        return f"{name}_ng45"
    return name


def _maybe_run_orfs(name: str, result: dict) -> dict:
    if not _orfs_enabled() or "error" in result or result.get("paused"):
        return result
    benchmark_name = _orfs_benchmark_name(name)
    if benchmark_name not in {"ariane133_ng45", "ariane136_ng45", "bp_quad_ng45", "nvdla_ng45", "mempool_tile_ng45", "ariane136_asap7", "nvdla_asap7", "mempool_tile_asap7"}:
        return result
    placement_path = Path(result.get("placement_paths", {}).get("placement_pt", ""))
    if not placement_path.exists():
        result["orfs"] = {"error": f"missing placement tensor at {placement_path}"}
        return result
    module = _load_orfs_module()
    output_root = Path(os.getenv("DP_RUNNER_ORFS_OUTPUT_DIR", "output/orfs_evaluation"))
    output_dir = output_root / benchmark_name
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    _heartbeat_update(name=name, current_stage="orfs_validate", placement_path=str(placement_path))
    eval_result = module.evaluate_benchmark(
        benchmark_name=benchmark_name,
        orfs_root=Path(os.getenv("DP_RUNNER_ORFS_ROOT", "../OpenROAD-flow-scripts")),
        output_dir=output_dir,
        use_docker=not _env_bool("DP_RUNNER_ORFS_NO_DOCKER", False),
        skip_synthesis=_env_bool("DP_RUNNER_ORFS_SKIP_SYNTHESIS", False),
        placement_path=placement_path,
    )
    elapsed = time.time() - start
    result["orfs"] = eval_result.get("orfs", eval_result)
    result["orfs_proxy_cost"] = float(eval_result.get("proxy_cost", result.get("proxy_cost", 0.0)))
    result["orfs_runtime"] = float(elapsed)
    result["orfs_output_dir"] = str(output_dir)
    profile = result.setdefault("profile", {})
    profile["orfs_validate"] = float(elapsed)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="DREAMPlace-style analytical placement runner.")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--ng45", action="store_true")
    parser.add_argument("--benchmarks", nargs="*", default=None)
    parser.add_argument("--out", default="team_trash_Workspace/dreamplace_gpu/results/latest.jsonl")
    parser.add_argument(
        "--jobs",
        type=int,
        default=int(os.getenv("DP_RUNNER_JOBS", "1")),
        help="Number of benchmark worker processes. Use cautiously on a single GPU.",
    )
    parser.add_argument(
        "--schedule",
        choices=("input", "large-first", "small-first"),
        default=os.getenv("DP_RUNNER_SCHEDULE", "input"),
        help="Benchmark launch order for dynamic parallel runs.",
    )
    parser.add_argument(
        "--isolate",
        action="store_true",
        default=os.getenv("DP_RUNNER_ISOLATE", "0") != "0",
        help="Run each benchmark in a fresh Python subprocess so CUDA memory is released after each case.",
    )
    parser.add_argument(
        "--orfs-validate",
        action="store_true",
        default=_env_bool("DP_RUNNER_ORFS_VALIDATE", False),
        help="After each NG45/ASAP7 placement finishes, run ORFS and record WNS/TNS/Area in the same JSONL row.",
    )
    parser.add_argument(
        "--orfs-root",
        default=os.getenv("DP_RUNNER_ORFS_ROOT", "../OpenROAD-flow-scripts"),
        help="Path to OpenROAD-flow-scripts.",
    )
    parser.add_argument(
        "--orfs-output-dir",
        default=os.getenv("DP_RUNNER_ORFS_OUTPUT_DIR", "output/orfs_evaluation"),
        help="Directory for ORFS configs, logs, and reports.",
    )
    parser.add_argument(
        "--orfs-no-docker",
        action="store_true",
        default=_env_bool("DP_RUNNER_ORFS_NO_DOCKER", False),
        help="Run ORFS without docker_shell.",
    )
    parser.add_argument(
        "--orfs-skip-synthesis",
        action="store_true",
        default=_env_bool("DP_RUNNER_ORFS_SKIP_SYNTHESIS", True),
        help="Reuse the provided synthesized netlist inside ORFS.",
    )
    args = parser.parse_args()
    _set_orfs_env(args)

    if args.benchmarks:
        benchmarks = args.benchmarks
    elif args.ng45:
        benchmarks = sorted(NG45_BENCHMARKS.keys())
    elif args.all:
        benchmarks = IBM_BENCHMARKS
    else:
        benchmarks = ["ibm01"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    jobs = max(1, int(args.jobs))
    if args.isolate and os.getenv("DP_RUNNER_CHILD") != "1":
        _run_isolated(benchmarks, out, jobs, args.schedule)
        return
    if jobs > 1:
        _run_parallel(benchmarks, out, jobs, args.schedule)
        return

    results = []
    with out.open("w", encoding="utf-8") as f:
        for name in benchmarks:
            try:
                result = _run_one(name)
            except Exception:
                result = {"name": name, "error": traceback.format_exc(), "runtime": 0.0}
            result = _maybe_run_orfs(name, result)
            results.append(result)
            f.write(json.dumps(result, sort_keys=True) + "\n")
            f.flush()
            _print_result(result)
    results.sort(key=lambda row: row["name"])
    with out.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, sort_keys=True) + "\n")
    print(f"Wrote {out}")


def _run_parallel(benchmarks: list[str], out: Path, jobs: int, schedule: str) -> None:
    pending = _scheduled_benchmarks(benchmarks, schedule)
    results = []
    with out.open("w", encoding="utf-8") as f:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {}
            while pending or futures:
                while pending and len(futures) < jobs:
                    name = pending.pop(0)
                    futures[pool.submit(_run_one_safe, name)] = name
                    print(f"[runner] launched {name} active={len(futures)}/{jobs} pending={len(pending)}")
                for future in as_completed(futures):
                    futures.pop(future)
                    result = future.result()
                    result = _maybe_run_orfs(result.get("name", name), result)
                    results.append(result)
                    f.write(json.dumps(result, sort_keys=True) + "\n")
                    f.flush()
                    _print_result(result)
                    break
    results.sort(key=lambda row: row["name"])
    with out.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, sort_keys=True) + "\n")
    print(f"Wrote {out}")


def _run_isolated(benchmarks: list[str], out: Path, jobs: int, schedule: str) -> None:
    pending = _scheduled_benchmarks(benchmarks, schedule)
    results = []
    tmp_dir = out.parent / f".{out.stem}.isolated"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    active: dict[subprocess.Popen, tuple[str, Path, float]] = {}
    with out.open("w", encoding="utf-8") as f:
        while pending or active:
            while pending and len(active) < jobs:
                name = pending.pop(0)
                child_out = tmp_dir / f"{name}.jsonl"
                env = os.environ.copy()
                env["DP_RUNNER_CHILD"] = "1"
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--benchmarks",
                    name,
                    "--out",
                    str(child_out),
                ]
                start = time.time()
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                )
                active[process] = (name, child_out, start)
                print(f"[runner] launched isolated {name} pid={process.pid} active={len(active)}/{jobs} pending={len(pending)}")
            time.sleep(0.5)
            for process, (name, child_out, start) in list(active.items()):
                if process.poll() is None:
                    continue
                active.pop(process)
                stdout, stderr = process.communicate()
                if stdout.strip():
                    print(stdout.rstrip())
                if stderr.strip():
                    print(stderr.rstrip(), file=sys.stderr)
                result = _read_child_result(name, child_out, process.returncode, stderr, time.time() - start)
                result = _maybe_run_orfs(name, result)
                results.append(result)
                f.write(json.dumps(result, sort_keys=True) + "\n")
                f.flush()
                _print_result(result)
                break
    results.sort(key=lambda row: row["name"])
    with out.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, sort_keys=True) + "\n")
    print(f"Wrote {out}")


def _read_child_result(name: str, child_out: Path, returncode: int | None, stderr: str, wall_time: float) -> dict:
    if child_out.exists():
        rows = [json.loads(line) for line in child_out.read_text(encoding="utf-8").splitlines() if line.strip()]
        if rows:
            result = rows[-1]
            result["isolated_wall_time"] = float(wall_time)
            result["isolated_returncode"] = int(returncode or 0)
            return result
    return {
        "name": name,
        "error": stderr or f"isolated child exited with return code {returncode}",
        "runtime": 0.0,
        "isolated_wall_time": float(wall_time),
        "isolated_returncode": int(returncode or 0),
    }


def _scheduled_benchmarks(benchmarks: list[str], schedule: str) -> list[str]:
    if schedule == "input":
        return list(benchmarks)
    keyed = []
    for index, name in enumerate(benchmarks):
        try:
            benchmark, _ = _load_benchmark(name)
            size = (int(benchmark.num_hard_macros), int(benchmark.num_macros))
        except Exception:
            size = (0, 0)
        keyed.append((size, index, name))
    reverse = schedule == "large-first"
    return [name for _, _, name in sorted(keyed, key=lambda item: (item[0], -item[1]), reverse=reverse)]


def _run_one_safe(name: str) -> dict:
    try:
        return _run_one(name)
    except Exception:
        return {"name": name, "error": traceback.format_exc(), "runtime": 0.0}


def _run_one(name: str) -> dict:
    benchmark, plc = _load_benchmark(name)
    _heartbeat_update(name=name, current_stage="place", benchmark=name, device=os.getenv("DP_DEVICE", "cpu"))
    placer = DreamPlaceGPUPlacer()
    start = time.time()
    try:
        placement = placer.place(benchmark)
    except PauseRequested as exc:
        runtime = time.time() - start
        result = {
            "name": name,
            "paused": True,
            "runtime": float(runtime),
            "checkpoint_path": exc.checkpoint_path,
        }
        _heartbeat_update(name=name, current_stage="paused", paused=True, checkpoint_path=exc.checkpoint_path)
        return result
    runtime = time.time() - start
    _heartbeat_update(name=name, current_stage="official_validate", runner_place=runtime)
    validation_start = time.time()
    valid, violations = validate_placement(placement, benchmark)
    validation_runtime = time.time() - validation_start
    _heartbeat_update(name=name, current_stage="official_cost", official_validate=validation_runtime)
    cost_start = time.time()
    costs = compute_proxy_cost(placement, benchmark, plc)
    official_cost_runtime = time.time() - cost_start
    profile = dict(getattr(placer, "last_profile", {}))
    profile["official_validate"] = float(validation_runtime)
    profile["official_cost"] = float(official_cost_runtime)
    profile["runner_place"] = float(runtime)
    placement_paths = dict(getattr(placer, "last_placement_paths", {}))
    result = {
        "name": name,
        "proxy_cost": float(costs["proxy_cost"]),
        "wirelength": float(costs["wirelength_cost"]),
        "density": float(costs["density_cost"]),
        "congestion": float(costs["congestion_cost"]),
        "overlaps": int(costs["overlap_count"]),
        "valid": bool(valid),
        "violations": violations,
        "runtime": float(runtime),
        "profile": profile,
        "placement_paths": placement_paths,
    }
    _heartbeat_update(name=name, current_stage="done", result_proxy=result["proxy_cost"], valid=result["valid"])
    return result


def _load_benchmark(name: str):
    if name in NG45_BENCHMARKS:
        base = NG45_BENCHMARKS[name]
        return load_benchmark(f"{base}/netlist.pb.txt", f"{base}/initial.plc", name=name)
    return load_benchmark_from_dir(f"external/MacroPlacement/Testcases/ICCAD04/{name}")


def _print_result(result: dict) -> None:
    if result.get("paused"):
        print(f"{result['name']:>13} PAUSED checkpoint={result.get('checkpoint_path', '')}")
        return
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
    orfs = result.get("orfs")
    if isinstance(orfs, dict) and orfs:
        if "error" in orfs:
            print(f"{'':>13} ORFS failed: {orfs['error']}")
        else:
            print(
                f"{'':>13} ORFS wns={float(orfs.get('wns', 0.0)):.2f} "
                f"tns={float(orfs.get('tns', 0.0)):.2f} "
                f"area={float(orfs.get('area', 0.0)) / 1e6:.3f} "
                f"[{float(result.get('orfs_runtime', 0.0)):.2f}s]"
            )


if __name__ == "__main__":
    main()
