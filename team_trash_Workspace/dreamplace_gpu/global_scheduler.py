from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from macro_place.evaluate import IBM_BENCHMARKS, NG45_BENCHMARKS  # noqa: E402
from macro_place.loader import load_benchmark, load_benchmark_from_dir  # noqa: E402


@dataclass(frozen=True)
class BenchmarkInfo:
    name: str
    hard_macros: int
    total_macros: int
    grid_cells: int

    @property
    def size_key(self) -> tuple[int, int, int]:
        return (self.hard_macros, self.total_macros, self.grid_cells)


@dataclass
class ActiveRun:
    info: BenchmarkInfo
    name: str
    device: str
    launch_kind: str
    process: subprocess.Popen
    out_path: Path
    log_path: Path
    heartbeat_path: Path
    pause_path: Path
    checkpoint_path: Path
    start_time: float


def main() -> None:
    parser = argparse.ArgumentParser(description="Resource-aware DreamPlace benchmark scheduler.")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--ng45", action="store_true")
    parser.add_argument("--benchmarks", nargs="*", default=None)
    parser.add_argument("--out", default="team_trash_Workspace/dreamplace_gpu/results/scheduled.jsonl")
    parser.add_argument("--work-dir", default="team_trash_Workspace/dreamplace_gpu/results/scheduled_work")
    parser.add_argument("--gpu-device", default=os.getenv("DP_SCHED_GPU_DEVICE", "cuda:0"))
    parser.add_argument("--max-gpu-jobs", type=int, default=int(os.getenv("DP_SCHED_MAX_GPU_JOBS", "1")))
    parser.add_argument("--max-cpu-jobs", type=int, default=int(os.getenv("DP_SCHED_MAX_CPU_JOBS", "0")))
    parser.add_argument(
        "--cpu-hard-limit",
        type=int,
        default=int(os.getenv("DP_SCHED_CPU_HARD_LIMIT", "320")),
        help="Benchmarks with at most this many hard macros may run on CPU.",
    )
    parser.add_argument(
        "--cpu-load-limit",
        type=float,
        default=float(os.getenv("DP_SCHED_CPU_LOAD_LIMIT", str(max((os.cpu_count() or 2) - 2, 1)))),
        help="Do not launch new CPU jobs when 1-minute load is above this value.",
    )
    parser.add_argument("--min-free-gpu-mib", type=int, default=int(os.getenv("DP_SCHED_MIN_FREE_GPU_MIB", "12000")))
    parser.add_argument(
        "--max-gpu-burst-jobs",
        type=int,
        default=int(os.getenv("DP_SCHED_MAX_GPU_BURST_JOBS", "2")),
        help="Maximum concurrent GPU jobs when the active GPU job is under-utilizing the device.",
    )
    parser.add_argument(
        "--gpu-idle-util-threshold",
        type=float,
        default=float(os.getenv("DP_SCHED_GPU_IDLE_UTIL_THRESHOLD", "35")),
        help="Allow opportunistic GPU backfill when utilization is at or below this percentage.",
    )
    parser.add_argument(
        "--gpu-backfill-free-mib",
        type=int,
        default=int(os.getenv("DP_SCHED_GPU_BACKFILL_FREE_MIB", "8000")),
        help="Minimum free GPU memory required before launching an extra backfill job.",
    )
    parser.add_argument(
        "--gpu-evict-util-threshold",
        type=float,
        default=float(os.getenv("DP_SCHED_GPU_EVICT_UTIL_THRESHOLD", "70")),
        help="Evict a GPU backfill worker when utilization rises above this threshold and a primary slot is needed.",
    )
    parser.add_argument(
        "--backfill-grace-seconds",
        type=float,
        default=float(os.getenv("DP_SCHED_BACKFILL_GRACE_SECONDS", "20")),
        help="Do not evict a newly launched backfill worker before this age unless memory pressure is high.",
    )
    parser.add_argument(
        "--pause-timeout-seconds",
        type=float,
        default=float(os.getenv("DP_SCHED_PAUSE_TIMEOUT_SECONDS", "120")),
        help="Wait this long for a paused backfill worker to stop at a checkpoint before forcing termination.",
    )
    parser.add_argument("--poll-interval", type=float, default=float(os.getenv("DP_SCHED_POLL_INTERVAL", "10")))
    parser.add_argument("--schedule", choices=("large-first", "small-first", "input"), default="large-first")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    names = _select_benchmarks(args)
    infos = [_benchmark_info(name) for name in names]
    infos = _order(infos, args.schedule)
    cpu_queue = [item for item in infos if item.hard_macros <= args.cpu_hard_limit]
    gpu_queue = [item for item in infos if item.hard_macros > args.cpu_hard_limit]

    if args.dry_run:
        print("[scheduler] cpu_queue=" + ", ".join(item.name for item in cpu_queue))
        print("[scheduler] gpu_queue=" + ", ".join(item.name for item in gpu_queue))
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    active: list[ActiveRun] = []

    with out.open("w", encoding="utf-8") as result_file:
        while cpu_queue or gpu_queue or active:
            _collect_finished(active, results, result_file, gpu_queue, cpu_queue, args)
            cpu_load = _loadavg_1m()
            gpu = _gpu_state()
            _maybe_evict_backfill(active, gpu_queue, cpu_queue, args, gpu)
            _collect_finished(active, results, result_file, gpu_queue, cpu_queue, args)
            gpu = _gpu_state()

            while True:
                gpu_limit = _gpu_launch_limit(active, args.gpu_device, args, gpu)
                if _active_count(active, args.gpu_device) >= gpu_limit:
                    break
                job = _pick_gpu_job(gpu_queue, cpu_queue, active, args, gpu)
                if job is None:
                    break
                if gpu is not None and gpu["free_mib"] < _required_free_mib(active, args.gpu_device, args, gpu):
                    _restore_job(job, gpu_queue, cpu_queue, args)
                    break
                launch_kind = "primary" if _active_count(active, args.gpu_device) < max(int(args.max_gpu_jobs), 0) else "backfill"
                active.append(_launch(job, args.gpu_device, work_dir, launch_kind=launch_kind))
                gpu = _gpu_state()

            while cpu_queue and _active_count(active, "cpu") < max(args.max_cpu_jobs, 0):
                if cpu_load is not None and cpu_load > args.cpu_load_limit:
                    break
                active.append(_launch(cpu_queue.pop(0), "cpu", work_dir, launch_kind="cpu"))
                cpu_load = _loadavg_1m()

            if not active and cpu_queue and args.max_cpu_jobs <= 0:
                # CPU queue is disabled; move remaining small jobs to GPU so work can finish.
                gpu_queue = cpu_queue + gpu_queue
                cpu_queue = []
                continue

            time.sleep(max(float(args.poll_interval), 1.0))

    results.sort(key=lambda row: row.get("name", ""))
    with out.open("w", encoding="utf-8") as result_file:
        for row in results:
            result_file.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"Wrote {out}")


def _select_benchmarks(args) -> list[str]:
    if args.benchmarks:
        return list(args.benchmarks)
    if args.ng45:
        return list(NG45_BENCHMARKS.keys())
    if args.all:
        return list(IBM_BENCHMARKS)
    return ["ibm01"]


def _order(infos: list[BenchmarkInfo], schedule: str) -> list[BenchmarkInfo]:
    if schedule == "input":
        return list(infos)
    return sorted(infos, key=lambda item: item.size_key, reverse=schedule == "large-first")


def _benchmark_info(name: str) -> BenchmarkInfo:
    benchmark, _ = _load_benchmark(name)
    return BenchmarkInfo(
        name=name,
        hard_macros=int(benchmark.num_hard_macros),
        total_macros=int(benchmark.num_macros),
        grid_cells=int(benchmark.grid_rows) * int(benchmark.grid_cols),
    )


def _load_benchmark(name: str):
    if name in NG45_BENCHMARKS:
        base = NG45_BENCHMARKS[name]
        return load_benchmark(f"{base}/netlist.pb.txt", f"{base}/initial.plc", name=name)
    return load_benchmark_from_dir(f"external/MacroPlacement/Testcases/ICCAD04/{name}")


def _launch(info: BenchmarkInfo, device: str, work_dir: Path, launch_kind: str) -> ActiveRun:
    out_path = work_dir / f"{info.name}_{device.replace(':', '')}.jsonl"
    log_path = work_dir / f"{info.name}_{device.replace(':', '')}.log"
    heartbeat_path = work_dir / f"{info.name}_{device.replace(':', '')}.heartbeat.json"
    checkpoint_path = _checkpoint_path(info.name, device, work_dir)
    pause_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".pause")
    env = os.environ.copy()
    env["DP_DEVICE"] = device
    env["DP_STAGE_HEARTBEAT_PATH"] = str(heartbeat_path)
    env.setdefault("DP_CHECKPOINT_BACKEND", "shm")
    env["DP_CHECKPOINT_PATH"] = str(checkpoint_path)
    env["DP_PAUSE_REQUEST_PATH"] = str(pause_path)
    if pause_path.exists():
        pause_path.unlink()
    cmd = [
        sys.executable,
        "-u",
        str(_HERE / "parallel_runner.py"),
        "--benchmarks",
        info.name,
        "--out",
        str(out_path),
    ]
    log = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    log.close()
    print(
        f"[scheduler] launched {info.name} device={device} kind={launch_kind} "
        f"hard={info.hard_macros} pid={process.pid}"
    )
    return ActiveRun(info, info.name, device, launch_kind, process, out_path, log_path, heartbeat_path, pause_path, checkpoint_path, time.time())


def _collect_finished(active: list[ActiveRun], results: list[dict], result_file, gpu_queue: list[BenchmarkInfo], cpu_queue: list[BenchmarkInfo], args) -> None:
    still_active = []
    for run in active:
        rc = run.process.poll()
        if rc is None:
            still_active.append(run)
            continue
        rows = _read_result_rows(run.out_path)
        if rows:
            row = rows[-1]
            row["scheduled_device"] = run.device
            row["scheduled_launch_kind"] = run.launch_kind
            row["scheduled_pid"] = run.process.pid
            row["scheduled_wall_time"] = time.time() - run.start_time
        else:
            row = {
                "name": run.name,
                "scheduled_device": run.device,
                "scheduled_launch_kind": run.launch_kind,
                "scheduled_pid": run.process.pid,
                "scheduled_wall_time": time.time() - run.start_time,
                "runtime": 0.0,
                "error": f"worker exited {rc}; see {run.log_path}",
            }
        if row.get("paused"):
            _restore_job(run.info, gpu_queue, cpu_queue, args)
            print(f"[scheduler] paused {run.name} device={run.device} rc={rc} checkpoint={run.checkpoint_path}")
            continue
        results.append(row)
        result_file.write(json.dumps(row, sort_keys=True) + "\n")
        result_file.flush()
        status = "FAILED" if "error" in row else "done"
        print(f"[scheduler] {status} {run.name} device={run.device} rc={rc}")
    active[:] = still_active


def _read_result_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _active_count(active: list[ActiveRun], device: str) -> int:
    return sum(1 for run in active if run.device == device)


def _gpu_launch_limit(active: list[ActiveRun], gpu_device: str, args, gpu: dict[str, float] | None) -> int:
    base = max(int(args.max_gpu_jobs), 0)
    burst = max(int(args.max_gpu_burst_jobs), base)
    active_gpu = _active_count(active, gpu_device)
    if active_gpu < base:
        return base
    if active_gpu >= burst:
        return burst
    if gpu is None:
        return base
    if float(gpu["util"]) > float(args.gpu_idle_util_threshold):
        return base
    if float(gpu["free_mib"]) < float(args.gpu_backfill_free_mib):
        return base
    return burst


def _required_free_mib(active: list[ActiveRun], gpu_device: str, args, gpu: dict[str, float] | None) -> float:
    if gpu is None:
        return 0.0
    if _active_count(active, gpu_device) < max(int(args.max_gpu_jobs), 0):
        return float(args.min_free_gpu_mib)
    return float(args.gpu_backfill_free_mib)


def _pick_gpu_job(
    gpu_queue: list[BenchmarkInfo],
    cpu_queue: list[BenchmarkInfo],
    active: list[ActiveRun],
    args,
    gpu: dict[str, float] | None,
) -> BenchmarkInfo | None:
    active_gpu = _active_count(active, args.gpu_device)
    base = max(int(args.max_gpu_jobs), 0)
    if active_gpu < base:
        if gpu_queue:
            return gpu_queue.pop(0)
        if cpu_queue:
            return cpu_queue.pop(0)
        return None
    if gpu is None:
        return None
    if float(gpu["util"]) > float(args.gpu_idle_util_threshold):
        return None
    if float(gpu["free_mib"]) < float(args.gpu_backfill_free_mib):
        return None
    if cpu_queue:
        return _pop_smallest(cpu_queue)
    if gpu_queue:
        return _pop_smallest(gpu_queue)
    return None


def _pop_smallest(queue: list[BenchmarkInfo]) -> BenchmarkInfo:
    index = min(range(len(queue)), key=lambda i: queue[i].size_key)
    return queue.pop(index)


def _restore_job(job: BenchmarkInfo, gpu_queue: list[BenchmarkInfo], cpu_queue: list[BenchmarkInfo], args) -> None:
    if job.hard_macros <= int(args.cpu_hard_limit):
        cpu_queue.insert(0, job)
    else:
        gpu_queue.insert(0, job)


def _checkpoint_root(work_dir: Path) -> Path:
    backend = os.getenv("DP_CHECKPOINT_BACKEND", "shm").strip().lower() or "shm"
    if backend == "disk":
        root = work_dir / "_checkpoints"
    else:
        root_text = os.getenv("DP_CHECKPOINT_DIR")
        if root_text:
            root = Path(root_text)
        elif Path("/dev/shm").exists():
            root = Path("/dev/shm") / "dreamplace_gpu"
        else:
            root = Path("/tmp") / "dreamplace_gpu_shm"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _checkpoint_path(name: str, device: str, work_dir: Path) -> Path:
    safe_device = device.replace(":", "")
    return _checkpoint_root(work_dir) / f"{name}_{safe_device}.ckpt.pt"


def _read_heartbeat(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _stage_is_gpu_heavy(stage: str | None) -> bool:
    if not stage:
        return False
    return stage in {
        "analytical",
        "soft_relax",
        "official_refine",
        "sa_refine",
        "place",
    }


def _backfill_victim(active: list[ActiveRun]) -> ActiveRun | None:
    backfills = [run for run in active if run.device.startswith("cuda") and run.launch_kind == "backfill"]
    if not backfills:
        return None
    return max(backfills, key=lambda run: run.start_time)


def _maybe_evict_backfill(
    active: list[ActiveRun],
    gpu_queue: list[BenchmarkInfo],
    cpu_queue: list[BenchmarkInfo],
    args,
    gpu: dict[str, float] | None,
) -> None:
    base = max(int(args.max_gpu_jobs), 0)
    if _active_count(active, args.gpu_device) <= base:
        return
    victim = _backfill_victim(active)
    if victim is None:
        return
    pending_primary = bool(gpu_queue)
    if not pending_primary and gpu is None:
        return
    age = time.time() - victim.start_time
    primary_runs = [run for run in active if run.device == args.gpu_device and run.launch_kind == "primary"]
    primary_heavy = any(_stage_is_gpu_heavy((_read_heartbeat(run.heartbeat_path) or {}).get("current_stage")) for run in primary_runs)
    mem_pressure = gpu is not None and float(gpu["free_mib"]) < float(args.min_free_gpu_mib)
    util_pressure = gpu is not None and float(gpu["util"]) >= float(args.gpu_evict_util_threshold)
    if not (mem_pressure or (pending_primary and util_pressure) or (primary_heavy and util_pressure)):
        return
    if age < float(args.backfill_grace_seconds) and not mem_pressure:
        return
    _request_pause_or_kill(victim, active, gpu_queue, cpu_queue, args)


def _request_pause_or_kill(run: ActiveRun, active: list[ActiveRun], gpu_queue: list[BenchmarkInfo], cpu_queue: list[BenchmarkInfo], args) -> None:
    print(f"[scheduler] requesting pause for backfill {run.name} pid={run.process.pid} due to GPU pressure")
    try:
        run.pause_path.parent.mkdir(parents=True, exist_ok=True)
        run.pause_path.write_text("pause\n", encoding="utf-8")
    except Exception:
        pass
    deadline = time.time() + max(float(args.pause_timeout_seconds), 1.0)
    while time.time() < deadline:
        if run.process.poll() is not None:
            return
        time.sleep(1.0)
    print(f"[scheduler] force killing backfill {run.name} pid={run.process.pid} after pause timeout")
    try:
        run.process.terminate()
        run.process.wait(timeout=5)
    except Exception:
        try:
            run.process.kill()
            run.process.wait(timeout=5)
        except Exception:
            pass
    if run in active:
        active.remove(run)
    _restore_job(run.info, gpu_queue, cpu_queue, args)


def _loadavg_1m() -> float | None:
    try:
        return float(os.getloadavg()[0])
    except OSError:
        return None


def _gpu_state() -> dict[str, float] | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None
    if not output:
        return None
    util, used, total = [float(item.strip()) for item in output.splitlines()[0].split(",")[:3]]
    return {"util": util, "used_mib": used, "total_mib": total, "free_mib": total - used}


if __name__ == "__main__":
    main()
