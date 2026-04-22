from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import types


ROOT = Path(__file__).resolve().parents[3]
DREAMPLACE_GPU = ROOT / "team_trash_Workspace" / "dreamplace_gpu"
if str(DREAMPLACE_GPU) not in sys.path:
    sys.path.insert(0, str(DREAMPLACE_GPU))


def _load_scheduler_module():
    macro_place = types.ModuleType("macro_place")
    evaluate = types.ModuleType("macro_place.evaluate")
    evaluate.IBM_BENCHMARKS = []
    evaluate.NG45_BENCHMARKS = {}
    loader = types.ModuleType("macro_place.loader")
    loader.load_benchmark = lambda *args, **kwargs: (_ for _ in ()).throw(NotImplementedError)
    loader.load_benchmark_from_dir = lambda *args, **kwargs: (_ for _ in ()).throw(NotImplementedError)
    sys.modules.setdefault("macro_place", macro_place)
    sys.modules["macro_place.evaluate"] = evaluate
    sys.modules["macro_place.loader"] = loader
    spec = importlib.util.spec_from_file_location("global_scheduler_test", DREAMPLACE_GPU / "global_scheduler.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SCHEDULER = _load_scheduler_module()
ActiveRun = SCHEDULER.ActiveRun
BenchmarkInfo = SCHEDULER.BenchmarkInfo
_gpu_launch_limit = SCHEDULER._gpu_launch_limit
_pick_gpu_job = SCHEDULER._pick_gpu_job
_stage_is_gpu_heavy = SCHEDULER._stage_is_gpu_heavy
_collect_finished = SCHEDULER._collect_finished


def _args(**overrides):
    data = {
        "gpu_device": "cuda:0",
        "max_gpu_jobs": 1,
        "max_gpu_burst_jobs": 2,
        "gpu_idle_util_threshold": 35.0,
        "gpu_backfill_free_mib": 8000,
        "min_free_gpu_mib": 12000,
        "cpu_hard_limit": 220,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def _info(name: str, hard: int, total: int, grid: int) -> BenchmarkInfo:
    return BenchmarkInfo(name=name, hard_macros=hard, total_macros=total, grid_cells=grid)


def test_gpu_launch_limit_allows_backfill_when_gpu_is_idle():
    args = _args()
    active = [
        ActiveRun(
            info=_info("ibm17", 540, 790, 1024),
            name="ibm17",
            device="cuda:0",
            launch_kind="primary",
            process=None,
            out_path=Path("/tmp/ibm17.jsonl"),
            log_path=Path("/tmp/ibm17.log"),
            heartbeat_path=Path("/tmp/ibm17.heartbeat.json"),
            pause_path=Path("/tmp/ibm17.pause"),
            checkpoint_path=Path("/tmp/ibm17.ckpt.pt"),
            start_time=0.0,
        )
    ]

    limit = _gpu_launch_limit(active, args.gpu_device, args, {"util": 12.0, "free_mib": 12000.0})

    assert limit == 2


def test_pick_gpu_job_prefers_small_cpu_job_for_backfill():
    args = _args()
    active = [
        ActiveRun(
            info=_info("ibm17", 540, 790, 1024),
            name="ibm17",
            device="cuda:0",
            launch_kind="primary",
            process=None,
            out_path=Path("/tmp/ibm17.jsonl"),
            log_path=Path("/tmp/ibm17.log"),
            heartbeat_path=Path("/tmp/ibm17.heartbeat.json"),
            pause_path=Path("/tmp/ibm17.pause"),
            checkpoint_path=Path("/tmp/ibm17.ckpt.pt"),
            start_time=0.0,
        )
    ]
    cpu_queue = [
        _info("ibm06", 120, 180, 256),
        _info("ibm01", 40, 90, 64),
    ]
    gpu_queue = [_info("ibm12", 600, 840, 2048)]

    picked = _pick_gpu_job(
        gpu_queue,
        cpu_queue,
        active,
        args,
        {"util": 18.0, "free_mib": 14000.0},
    )

    assert picked is not None
    assert picked.name == "ibm01"
    assert [item.name for item in cpu_queue] == ["ibm06"]
    assert [item.name for item in gpu_queue] == ["ibm12"]


def test_pick_gpu_job_keeps_single_gpu_job_when_device_is_busy():
    args = _args()
    active = [
        ActiveRun(
            info=_info("ibm17", 540, 790, 1024),
            name="ibm17",
            device="cuda:0",
            launch_kind="primary",
            process=None,
            out_path=Path("/tmp/ibm17.jsonl"),
            log_path=Path("/tmp/ibm17.log"),
            heartbeat_path=Path("/tmp/ibm17.heartbeat.json"),
            pause_path=Path("/tmp/ibm17.pause"),
            checkpoint_path=Path("/tmp/ibm17.ckpt.pt"),
            start_time=0.0,
        )
    ]

    picked = _pick_gpu_job(
        [_info("ibm12", 600, 840, 2048)],
        [_info("ibm01", 40, 90, 64)],
        active,
        args,
        {"util": 91.0, "free_mib": 18000.0},
    )

    assert picked is None


def test_stage_classifier_marks_gpu_heavy_phases_only():
    assert _stage_is_gpu_heavy("analytical") is True
    assert _stage_is_gpu_heavy("soft_relax") is True
    assert _stage_is_gpu_heavy("official_cost") is False
    assert _stage_is_gpu_heavy("official_validate") is False


def test_collect_finished_requeues_paused_result(tmp_path):
    class FakeProcess:
        pid = 1234

        def poll(self):
            return 0

    args = _args()
    out_path = tmp_path / "ibm08_cuda0.jsonl"
    out_path.write_text('{"name":"ibm08","paused":true,"checkpoint_path":"/dev/shm/dreamplace_gpu/ibm08.ckpt.pt"}\n', encoding="utf-8")
    active = [
        ActiveRun(
            info=_info("ibm08", 301, 859, 1292),
            name="ibm08",
            device="cuda:0",
            launch_kind="backfill",
            process=FakeProcess(),
            out_path=out_path,
            log_path=tmp_path / "ibm08.log",
            heartbeat_path=tmp_path / "ibm08.heartbeat.json",
            pause_path=tmp_path / "ibm08.pause",
            checkpoint_path=tmp_path / "ibm08.ckpt.pt",
            start_time=0.0,
        )
    ]
    results = []
    cpu_queue = []
    gpu_queue = []
    with (tmp_path / "main.jsonl").open("w", encoding="utf-8") as result_file:
        _collect_finished(active, results, result_file, gpu_queue, cpu_queue, args)
    assert not active
    assert not results
    assert gpu_queue and gpu_queue[0].name == "ibm08"
