from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types


ROOT = Path(__file__).resolve().parents[3]
DREAMPLACE_GPU = ROOT / "team_trash_Workspace" / "dreamplace_gpu"
if str(DREAMPLACE_GPU) not in sys.path:
    sys.path.insert(0, str(DREAMPLACE_GPU))


def _load_optimizer_module():
    benchmark_context = types.ModuleType("benchmark_context")
    benchmark_context.build_benchmark_context = lambda *args, **kwargs: object()
    benchmark_context.load_plc_for_benchmark = lambda *args, **kwargs: object()
    legalize = types.ModuleType("legalize")
    legalize.clamp_placement = lambda pos, benchmark: pos
    legalize.legalize_initial = lambda benchmark: None
    legalize.legalize_placement = lambda pos, benchmark, gap=0.001: pos
    trace_utils = types.ModuleType("trace_utils")
    trace_utils.PlacementTraceRecorder = type("PlacementTraceRecorder", (), {"from_env": staticmethod(lambda *args, **kwargs: None)})
    torch_objective = types.ModuleType("torch_objective")

    class FakeEvaluator:
        def __init__(self, *args, **kwargs):
            pass

    torch_objective.TorchProxyCostEvaluator = FakeEvaluator
    macro_obj = types.ModuleType("macro_place.objective")
    macro_obj.compute_overlap_metrics = lambda *args, **kwargs: {"overlap_count": 0}
    macro_obj.compute_proxy_cost = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("official should not be called"))
    sys.modules["benchmark_context"] = benchmark_context
    sys.modules["legalize"] = legalize
    sys.modules["trace_utils"] = trace_utils
    sys.modules["torch_objective"] = torch_objective
    sys.modules["macro_place.objective"] = macro_obj
    spec = importlib.util.spec_from_file_location("optimizer_test", DREAMPLACE_GPU / "optimizer.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_internal_official_switch_disables_official_ranking():
    module = _load_optimizer_module()
    config = module.DreamPlaceConfig(official_final_only=True)
    opt = module.DreamPlaceHybridOptimizer(config=config, device="cpu")
    assert opt._internal_official_enabled() is False
