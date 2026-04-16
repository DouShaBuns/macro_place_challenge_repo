from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
SA_GPU = ROOT / "team_trash_Workspace" / "sa_gpu"
if str(SA_GPU) not in sys.path:
    sys.path.insert(0, str(SA_GPU))

from benchmark_context import build_benchmark_context  # noqa: E402
from legalize import legalize_initial  # noqa: E402
from placer import SAGPUPlacer  # noqa: E402
from sa_optimizer import SAConfig  # noqa: E402
from torch_objective import TorchProxyCostEvaluator  # noqa: E402
from warm_start import prepare_warm_start, build_warm_start_provider  # noqa: E402

from macro_place.loader import load_benchmark_from_dir  # noqa: E402
from macro_place.objective import compute_overlap_metrics  # noqa: E402
from macro_place.utils import validate_placement  # noqa: E402


@pytest.fixture(scope="module")
def ibm01():
    path = ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / "ibm01"
    if not path.exists():
        pytest.skip("TILOS submodule not initialized")
    return load_benchmark_from_dir(path.as_posix())[0]


def test_torch_evaluator_smoke(ibm01):
    ctx = build_benchmark_context(ibm01, "cpu")
    evaluator = TorchProxyCostEvaluator(ctx)
    placements = ibm01.macro_positions.unsqueeze(0).repeat(2, 1, 1)
    costs = evaluator.evaluate_batch(placements)
    assert costs.official_proxy.shape == (2,)
    assert torch.isfinite(costs.wirelength_cost).all()
    assert torch.isfinite(costs.density_cost).all()
    assert torch.isfinite(costs.congestion_cost).all()


def test_overlap_and_legalize(ibm01):
    placement = ibm01.macro_positions.clone()
    if ibm01.num_hard_macros < 2:
        pytest.skip("needs at least two hard macros")
    placement[1] = placement[0]
    assert compute_overlap_metrics(placement, ibm01)["overlap_count"] > 0
    legal = legalize_initial(ibm01)
    assert compute_overlap_metrics(legal, ibm01)["overlap_count"] == 0


def test_torch_overlap_matches_official_strict_positive_overlap(ibm01):
    if ibm01.num_hard_macros < 2:
        pytest.skip("needs at least two hard macros")
    placement = legalize_initial(ibm01)
    sizes = ibm01.macro_sizes
    i, j = 0, 1
    thin_overlap = 1.0e-6
    placement[j, 0] = placement[i, 0] + (sizes[i, 0] + sizes[j, 0]) / 2 - thin_overlap
    placement[j, 1] = placement[i, 1]

    official = compute_overlap_metrics(placement, ibm01)
    ctx = build_benchmark_context(ibm01, "cpu")
    evaluator = TorchProxyCostEvaluator(ctx)
    costs = evaluator.evaluate_batch(placement)

    assert official["overlap_count"] > 0
    assert int(costs.overlap_count.item()) > 0
    assert not bool(costs.is_legal.item())


def test_official_loader_can_find_placer_class():
    placer_path = SA_GPU / "placer.py"
    spec = importlib.util.spec_from_file_location(placer_path.stem, str(placer_path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    classes = [
        value
        for value in vars(mod).values()
        if isinstance(value, type)
        and value.__module__ == placer_path.stem
        and callable(getattr(value, "place", None))
    ]
    assert [cls.__name__ for cls in classes] == ["SAGPUPlacer"]


def test_sa_config_accepts_final_selection_fields():
    config = SAConfig(top_k_final_candidates=4, local_refine_max_trials=10)
    assert config.top_k_final_candidates == 4
    assert config.local_refine_max_trials == 10


def test_placer_reads_final_selection_env(monkeypatch):
    monkeypatch.setenv("SA_GPU_TOP_K_FINAL_CANDIDATES", "5")
    monkeypatch.setenv("SA_GPU_LOCAL_REFINE_MAX_TRIALS", "11")
    placer = SAGPUPlacer(seeds=(1,), iters=0, candidate_batch=1, device="cpu")
    assert placer.config.top_k_final_candidates == 5
    assert placer.config.local_refine_max_trials == 11


def test_warm_start_accepts_hard_only_tensor(ibm01):
    hard_only = ibm01.macro_positions[: ibm01.num_hard_macros].clone()
    placement = prepare_warm_start(ibm01, build_warm_start_provider(hard_only))
    assert placement.shape == (ibm01.num_macros, 2)
    assert compute_overlap_metrics(placement, ibm01)["overlap_count"] == 0


def test_placer_accepts_warm_start_method(ibm01):
    class DummyWarmStart:
        def __init__(self):
            self.called = False

        def generate(self, benchmark):
            self.called = True
            return benchmark.macro_positions[: benchmark.num_hard_macros].clone()

    warm_start = DummyWarmStart()
    placer = SAGPUPlacer(
        seeds=(1,),
        iters=0,
        candidate_batch=1,
        top_k_final_candidates=2,
        local_refine_max_trials=0,
        warm_start=warm_start,
        device="cpu",
    )
    placement = placer.place(ibm01)
    valid, violations = validate_placement(placement, ibm01)
    assert warm_start.called
    assert valid, violations
    assert compute_overlap_metrics(placement, ibm01)["overlap_count"] == 0


def test_tiny_run_returns_valid_zero_overlap_placement(ibm01):
    placer = SAGPUPlacer(
        seeds=(1,),
        iters=1,
        candidate_batch=1,
        top_k_final_candidates=4,
        local_refine_max_trials=10,
        device="cpu",
    )
    placement = placer.place(ibm01)
    valid, violations = validate_placement(placement, ibm01)
    assert valid, violations
    assert compute_overlap_metrics(placement, ibm01)["overlap_count"] == 0
