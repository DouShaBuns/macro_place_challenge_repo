"""DREAMPlace 风格 GPU 解析式 placer 入口。

Pipeline：global placement (Adam + WL + eDensity) → legalize (spiral search) → detail (hill-climb).

评测器通过反射加载 placer.py 里的第一个带 `place` 方法的类并无参数实例化它。
所以 DreamPlacer 必须定义在这个文件里，且不能要求构造参数。配置全走环境变量。
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from macro_place.benchmark import Benchmark  # noqa: E402
from macro_place.loader import load_benchmark, load_benchmark_from_dir  # noqa: E402

from detail import hill_climb  # noqa: E402
from global_place import GlobalConfig, run_global_place  # noqa: E402
from legalize import legalize  # noqa: E402


NG45_DIRS = {
    "ariane133": "external/MacroPlacement/Flows/NanGate45/ariane133/netlist/output_CT_Grouping",
    "ariane133_ng45": "external/MacroPlacement/Flows/NanGate45/ariane133/netlist/output_CT_Grouping",
    "ariane136": "external/MacroPlacement/Flows/NanGate45/ariane136/netlist/output_CT_Grouping",
    "ariane136_ng45": "external/MacroPlacement/Flows/NanGate45/ariane136/netlist/output_CT_Grouping",
    "mempool_tile": "external/MacroPlacement/Flows/NanGate45/mempool_tile/netlist/output_CT_Grouping",
    "mempool_tile_ng45": "external/MacroPlacement/Flows/NanGate45/mempool_tile/netlist/output_CT_Grouping",
    "nvdla": "external/MacroPlacement/Flows/NanGate45/nvdla/netlist/output_CT_Grouping",
    "nvdla_ng45": "external/MacroPlacement/Flows/NanGate45/nvdla/netlist/output_CT_Grouping",
}


def _load_plc_for(name: str):
    """Detail placement 要 PlacementCost 做裁判。找不到就返回 None（跳过 detail）。"""
    ibm_dir = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if ibm_dir.exists():
        _, plc = load_benchmark_from_dir(str(ibm_dir))
        return plc
    ng45_dir = NG45_DIRS.get(name)
    if ng45_dir:
        netlist = Path(ng45_dir) / "netlist.pb.txt"
        initial = Path(ng45_dir) / "initial.plc"
        if netlist.exists():
            _, plc = load_benchmark(netlist.as_posix(), initial.as_posix(), name=name)
            return plc
    return None


def _log(log_path: Optional[str], record: dict) -> None:
    if not log_path:
        return
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _configure_determinism() -> None:
    """强制 deterministic：同输入同输出，避免 GPU atomic 累加顺序抖动。"""
    if os.getenv("DP_DETERMINISTIC", "1") != "1":
        return
    # CUBLAS 要求的 workspace 配置，必须在创建任何 CUDA stream 前设置
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
    except Exception:
        # 部分 op 没有 deterministic 实现时退化为 warn
        torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # 设 torch seed 保底（我们 pipeline 本身无随机采样，但 scatter_reduce 等底层
    # 在 deterministic 路径下有时会用到 RNG 初始化）
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


class DreamPlacer:
    """DREAMPlace 风格解析式 placer。全部配置走 env var。"""

    def __init__(self):
        _configure_determinism()
        # Global placement
        self.iters = int(os.getenv("DP_ITERS", "1500"))
        self.grid_base = int(os.getenv("DP_GRID", "256"))
        self.lr_factor = float(os.getenv("DP_LR_FACTOR", "0.05"))
        self.lambda_init_ratio = float(os.getenv("DP_LAMBDA_INIT_RATIO", "1.0"))
        self.lambda_mult = float(os.getenv("DP_LAMBDA_MULT", "1.20"))
        self.lambda_step = int(os.getenv("DP_LAMBDA_STEP", "50"))
        self.gamma_start = float(os.getenv("DP_GAMMA_START", "4.0"))
        self.gamma_end = float(os.getenv("DP_GAMMA_END", "1.0"))
        self.global_time_budget = float(os.getenv("DP_GLOBAL_TIME", "540"))
        # Detail placement
        self.detail_trials = int(os.getenv("DP_DETAIL_TRIALS", "1000"))
        self.detail_time_budget = float(os.getenv("DP_DETAIL_TIME", "120"))
        # Infra
        self.device = os.getenv("DP_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.log_path = os.getenv("DP_LOG_PATH")
        self.log_every = int(os.getenv("DP_LOG_EVERY", "100"))

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t_total = time.time()
        name = benchmark.name or "unknown"

        # 1. Global placement
        config = GlobalConfig(
            iters=self.iters,
            grid_base=self.grid_base,
            lr_factor=self.lr_factor,
            lambda_init_ratio=self.lambda_init_ratio,
            lambda_mult=self.lambda_mult,
            lambda_step=self.lambda_step,
            gamma_start_factor=self.gamma_start,
            gamma_end_factor=self.gamma_end,
            log_every=self.log_every,
            time_budget=self.global_time_budget,
            log_path=self.log_path,
            diagnostic_history=False,
        )
        t0 = time.time()
        gr = run_global_place(benchmark, config, device=self.device)
        t_global = time.time() - t0
        _log(self.log_path, {
            "name": name, "stage": "global_done",
            "final_step": gr.final_step, "stop_reason": gr.stop_reason,
            "elapsed": t_global,
        })

        # 2. Legalize
        t0 = time.time()
        placement = legalize(gr.placement.clone(), benchmark)
        t_legal = time.time() - t0
        _log(self.log_path, {"name": name, "stage": "legalize_done", "elapsed": t_legal})

        # 3. Detail placement (optional, needs PlacementCost)
        t_detail = 0.0
        detail_diag = None
        if self.detail_trials > 0:
            plc = _load_plc_for(name)
            if plc is not None:
                t0 = time.time()
                placement, detail_diag = hill_climb(
                    placement, benchmark, plc,
                    max_trials=self.detail_trials,
                    time_budget=self.detail_time_budget,
                )
                t_detail = time.time() - t0
                _log(self.log_path, {
                    "name": name, "stage": "detail_done",
                    "elapsed": t_detail, **detail_diag,
                })
            else:
                _log(self.log_path, {
                    "name": name, "stage": "detail_skipped",
                    "reason": "plc_not_found",
                })

        elapsed = time.time() - t_total
        _log(self.log_path, {
            "name": name, "stage": "total",
            "t_global": t_global, "t_legal": t_legal, "t_detail": t_detail,
            "elapsed": elapsed,
        })
        return placement
