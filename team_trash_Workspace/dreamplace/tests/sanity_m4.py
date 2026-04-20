"""M4 sanity check：spiral search legalization。

流程：跑 M3 global → legalize → 对比 proxy 前后变化。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

_DREAM = Path(__file__).resolve().parents[1]
if str(_DREAM) not in sys.path:
    sys.path.insert(0, str(_DREAM))

from macro_place.loader import load_benchmark_from_dir  # noqa: E402
from macro_place.objective import compute_proxy_cost  # noqa: E402

from global_place import GlobalConfig, run_global_place  # noqa: E402
from legalize import legalize  # noqa: E402


def main():
    bench, plc = load_benchmark_from_dir("external/MacroPlacement/Testcases/ICCAD04/ibm01")
    print(f"== ibm01 legalize sanity ==")

    # Baseline A: initial.plc 直接 legalize
    t0 = time.time()
    legal_init = legalize(bench.macro_positions.clone(), bench)
    t_legal_init = time.time() - t0
    p_init_legal = compute_proxy_cost(legal_init, bench, plc)
    print(f"\n[A] initial.plc → legalize (耗时 {t_legal_init:.2f}s)")
    print(f"    proxy = {p_init_legal['proxy_cost']:.4f}  "
          f"wl={p_init_legal['wirelength_cost']:.4f} "
          f"den={p_init_legal['density_cost']:.4f} "
          f"cong={p_init_legal['congestion_cost']:.4f} "
          f"overlaps={p_init_legal['overlap_count']}")

    # Baseline B: 跑 global 再 legalize
    config = GlobalConfig(iters=1500, lambda_init_ratio=1.0, lambda_mult=1.20,
                          lambda_step=50, log_every=500, diagnostic_history=False)
    print(f"\n[B] global (1500 步) + legalize")
    t0 = time.time()
    gr = run_global_place(bench, config, device="cpu")
    t_global = time.time() - t0
    p_global = compute_proxy_cost(gr.placement.clone(), bench, plc)
    print(f"    global 完成 ({t_global:.2f}s, step {gr.final_step})")
    print(f"    未 legalize proxy = {p_global['proxy_cost']:.4f} "
          f"overlaps={p_global['overlap_count']}")

    t0 = time.time()
    legal_g = legalize(gr.placement.clone(), bench)
    t_legal_g = time.time() - t0
    p_legal_g = compute_proxy_cost(legal_g, bench, plc)
    print(f"    legalize 完成 ({t_legal_g:.2f}s)")
    print(f"    legalize 后 proxy = {p_legal_g['proxy_cost']:.4f}  "
          f"wl={p_legal_g['wirelength_cost']:.4f} "
          f"den={p_legal_g['density_cost']:.4f} "
          f"cong={p_legal_g['congestion_cost']:.4f} "
          f"overlaps={p_legal_g['overlap_count']}")

    # 计算退化
    d_proxy = p_legal_g['proxy_cost'] - p_global['proxy_cost']
    print(f"\n    Δ proxy due to legalize: {d_proxy:+.4f} "
          f"({d_proxy / p_global['proxy_cost']:+.1%})")

    print(f"\n== 对照 ==")
    print(f"  SA 团队 ibm01   = 1.2935")
    print(f"  RePlAce ibm01   = 0.9976")
    print(f"  DreamPlace v1  = {p_legal_g['proxy_cost']:.4f}")

    # 断言
    assert p_legal_g['overlap_count'] == 0, (
        f"legalize 失败：仍有 {p_legal_g['overlap_count']} 个 overlap"
    )
    print(f"\n  ✓ zero overlap after legalize")


if __name__ == "__main__":
    main()
