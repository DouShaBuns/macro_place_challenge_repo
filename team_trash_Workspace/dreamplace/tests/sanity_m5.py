"""M5 sanity：hill-climb detail placement 是否能进一步改善 proxy。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_DREAM = Path(__file__).resolve().parents[1]
if str(_DREAM) not in sys.path:
    sys.path.insert(0, str(_DREAM))

from macro_place.loader import load_benchmark_from_dir  # noqa: E402
from macro_place.objective import compute_proxy_cost  # noqa: E402

from detail import hill_climb  # noqa: E402
from global_place import GlobalConfig, run_global_place  # noqa: E402
from legalize import legalize  # noqa: E402


def main():
    bench, plc = load_benchmark_from_dir("external/MacroPlacement/Testcases/ICCAD04/ibm01")
    print("== ibm01 detail placement sanity ==")

    # global + legalize
    cfg = GlobalConfig(iters=1500, lambda_init_ratio=1.0, lambda_mult=1.20, lambda_step=50,
                       log_every=500, diagnostic_history=False)
    t0 = time.time()
    gr = run_global_place(bench, cfg, device="cpu")
    t_global = time.time() - t0

    t0 = time.time()
    legal = legalize(gr.placement.clone(), bench)
    t_legal = time.time() - t0
    p_legal = compute_proxy_cost(legal, bench, plc)
    print(f"\n[after global+legalize]  ({t_global+t_legal:.1f}s)")
    print(f"  proxy = {p_legal['proxy_cost']:.4f}  "
          f"(wl={p_legal['wirelength_cost']:.4f} "
          f"den={p_legal['density_cost']:.4f} "
          f"cong={p_legal['congestion_cost']:.4f}) "
          f"overlaps={p_legal['overlap_count']}")

    # detail
    t0 = time.time()
    refined, diag = hill_climb(legal, bench, plc, max_trials=1000, time_budget=120.0)
    t_detail = time.time() - t0
    p_refined = compute_proxy_cost(refined, bench, plc)
    print(f"\n[after hill-climb]  ({t_detail:.1f}s)")
    print(f"  proxy = {p_refined['proxy_cost']:.4f}  "
          f"(wl={p_refined['wirelength_cost']:.4f} "
          f"den={p_refined['density_cost']:.4f} "
          f"cong={p_refined['congestion_cost']:.4f}) "
          f"overlaps={p_refined['overlap_count']}")
    print(f"  trials={diag['trials']} accepts={diag['accepts']} "
          f"Δproxy={diag['proxy_delta']:+.4f} ({diag['proxy_delta']/diag['proxy_start']:+.2%})")

    print(f"\n== 最终对照 ==")
    print(f"  SA 团队 ibm01          = 1.2935")
    print(f"  RePlAce ibm01 baseline = 0.9976")
    print(f"  DreamPlace v1 (post-D) = {p_refined['proxy_cost']:.4f}")
    vs_replace = (p_refined['proxy_cost'] - 0.9976) / 0.9976
    print(f"  vs RePlAce = {vs_replace:+.2%}")


if __name__ == "__main__":
    main()
