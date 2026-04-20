"""M3 sanity check：global placement 训练循环跑 ibm01。

跑法：
    srun --account=kcl --partition=interruptible_cpu --time=00:10:00 \
         --mem=8G --cpus-per-task=2 \
         uv run python team_trash_Workspace/dreamplace/tests/sanity_m3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_DREAM = Path(__file__).resolve().parents[1]
if str(_DREAM) not in sys.path:
    sys.path.insert(0, str(_DREAM))

from macro_place.loader import load_benchmark_from_dir  # noqa: E402
from macro_place.objective import compute_proxy_cost  # noqa: E402

from global_place import GlobalConfig, run_global_place  # noqa: E402


def main():
    bench, plc = load_benchmark_from_dir("external/MacroPlacement/Testcases/ICCAD04/ibm01")
    print(f"== ibm01: macros={bench.num_macros} hard={bench.num_hard_macros} "
          f"nets={bench.num_nets} canvas={bench.canvas_width:.2f}x{bench.canvas_height:.2f} ==\n")

    # Baseline: initial.plc 上的 official proxy
    init_proxy = compute_proxy_cost(bench.macro_positions.clone(), bench, plc)
    print(f"initial.plc  official proxy = {init_proxy['proxy_cost']:.4f}  "
          f"(wl={init_proxy['wirelength_cost']:.4f} den={init_proxy['density_cost']:.4f} "
          f"cong={init_proxy['congestion_cost']:.4f}) overlaps={init_proxy['overlap_count']}\n")

    # 跑 300 步看 loss 曲线和最终 official proxy
    config = GlobalConfig(
        iters=1500,
        grid_base=256,
        lr_factor=0.05,
        lambda_init_ratio=1.0,
        lambda_mult=1.20,
        lambda_step=50,
        gamma_start_factor=4.0,
        gamma_end_factor=1.0,
        log_every=100,
        diagnostic_history=True,
    )
    print(f"GlobalConfig: iters={config.iters} lr_factor={config.lr_factor} "
          f"λ_mult={config.lambda_mult}^(every {config.lambda_step}) "
          f"γ:{config.gamma_start_factor}->{config.gamma_end_factor}*grid_w\n")

    result = run_global_place(bench, config, device="cpu")

    # 打印 history
    print("step    wl       density   loss     λ        γ        overflow  rho_max  elapsed")
    for rec in result.history:
        print(f"{rec['step']:>4d}  {rec['wl']:.5f}  {rec['density']:.5f}  "
              f"{rec['loss']:.5f}  {rec['lambda']:.4f}  {rec['gamma']:.3f}  "
              f"{rec['overflow']:.4%}    {rec['rho_max']:.3f}    {rec['elapsed']:.1f}s")
    print(f"\nstopped at step {result.final_step}, reason = {result.stop_reason}, "
          f"elapsed {result.elapsed:.1f}s")

    # 用结果跑 official proxy（global 之后还没 legalize，所以肯定 overlap > 0）
    final_proxy = compute_proxy_cost(result.placement, bench, plc)
    print(f"\nGlobal 完成（未 legalize）official proxy = {final_proxy['proxy_cost']:.4f}  "
          f"(wl={final_proxy['wirelength_cost']:.4f} den={final_proxy['density_cost']:.4f} "
          f"cong={final_proxy['congestion_cost']:.4f}) overlaps={final_proxy['overlap_count']}")


if __name__ == "__main__":
    main()
