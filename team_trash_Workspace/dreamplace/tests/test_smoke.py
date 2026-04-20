"""Smoke test：ibm01 最小配置跑通 pipeline，验证合法性 + proxy 合理。

目标：≤60s CPU 跑完，仅验流程不评分。正式跑分用 sanity_m*.py / parallel_runner.py。

启动（从仓库根）：
    uv run pytest team_trash_Workspace/dreamplace/tests/test_smoke.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

_DREAM = Path(__file__).resolve().parents[1]
if str(_DREAM) not in sys.path:
    sys.path.insert(0, str(_DREAM))


@pytest.fixture(scope="module")
def ibm01():
    from macro_place.loader import load_benchmark_from_dir
    return load_benchmark_from_dir("external/MacroPlacement/Testcases/ICCAD04/ibm01")


def test_dream_placer_smoke(ibm01, monkeypatch):
    """端到端 smoke：输出合法且 proxy < 2.0（远高于阈值，仅防大回归）。"""
    # 最小配置，快速跑通
    monkeypatch.setenv("DP_ITERS", "200")
    monkeypatch.setenv("DP_GRID", "128")
    monkeypatch.setenv("DP_DETAIL_TRIALS", "0")  # 跳过 detail 节省时间
    monkeypatch.setenv("DP_DEVICE", "cpu")

    from placer import DreamPlacer
    from macro_place.objective import compute_proxy_cost

    bench, plc = ibm01
    placer = DreamPlacer()
    out = placer.place(bench)

    # 基础检查
    assert isinstance(out, torch.Tensor)
    assert out.shape == (bench.num_macros, 2), f"shape mismatch: {tuple(out.shape)}"
    assert torch.isfinite(out).all(), "placement 含 NaN/Inf"

    # 合法性 + 分数
    metrics = compute_proxy_cost(out, bench, plc)
    assert int(metrics["overlap_count"]) == 0, (
        f"hard macro 重叠数 = {metrics['overlap_count']}"
    )
    assert float(metrics["proxy_cost"]) < 2.0, (
        f"proxy_cost = {metrics['proxy_cost']}，可能存在回归"
    )
