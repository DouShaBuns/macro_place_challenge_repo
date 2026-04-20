"""M1 sanity check：WA wirelength vs 官方 HPWL，和梯度方向。

跑法（在本仓库根目录）：
    srun --account=kcl --partition=interruptible_cpu --time=00:05:00 \
         --mem=4G --cpus-per-task=2 \
         uv run python team_trash_Workspace/dreamplace/tests/sanity_m1.py
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

from benchmark_context import build_wl_context  # noqa: E402
from torch_loss import wa_wirelength  # noqa: E402


def main():
    bench, plc = load_benchmark_from_dir("external/MacroPlacement/Testcases/ICCAD04/ibm01")
    ctx = build_wl_context(bench, device="cpu")

    print(f"== ibm01 ==")
    print(f"macros={bench.num_macros} ports={ctx.num_ports} kept_nets={ctx.num_kept_nets}")
    print(f"total pins flattened = {ctx.pin_node_id.numel()}")
    print(f"canvas = {ctx.canvas_w:.2f} x {ctx.canvas_h:.2f}")

    # 官方 proxy 给出的 wirelength（用 initial placement）
    placement = bench.macro_positions.clone()
    official = compute_proxy_cost(placement, bench, plc)
    print(f"\nofficial wirelength = {official['wirelength_cost']:.6f}")

    # 在不同 gamma 下试 WA
    grid_w = ctx.canvas_w / max(bench.grid_cols, 1)
    print(f"grid_w = {grid_w:.4f}")
    for gamma_factor in [10.0, 4.0, 1.0, 0.25]:
        gamma = grid_w * gamma_factor
        wl, _ = wa_wirelength(placement, ctx, gamma=gamma, normalize=True)
        print(f"  gamma={gamma:.3f} (={gamma_factor:.2f}*grid_w)  WA-wl = {wl.item():.6f}")

    # 梯度方向：把一个 movable macro 朝它所有 net 的"其他端点重心"方向应该让 WL 下降
    pos = bench.macro_positions.clone().detach().requires_grad_(True)
    gamma = grid_w * 4.0
    wl, _ = wa_wirelength(pos, ctx, gamma=gamma, normalize=False)
    wl.backward()
    grad = pos.grad
    print(f"\n梯度 stats (gamma={gamma:.3f}, unnormalized):")
    print(f"  grad shape = {tuple(grad.shape)}")
    print(f"  grad norm = {grad.norm().item():.4f}")
    print(f"  grad max abs = {grad.abs().max().item():.4f}")
    print(f"  fraction of zero-grad rows = {(grad.abs().sum(dim=1)==0).float().mean().item():.3%}")

    # 单步 SGD：朝负梯度走一小步，应该让 wl 下降
    # 学习率应让单步移动远小于 canvas（target ~ 1 grid_w）
    target_step = grid_w * 0.5
    lr = target_step / max(grad.abs().max().item(), 1.0e-9)
    new_pos = (pos - lr * grad).detach()
    wl_new, _ = wa_wirelength(new_pos, ctx, gamma=gamma, normalize=False)
    print(f"\n单步 SGD（lr={lr:.6f}, target step ≈ {target_step:.4f}）:")
    print(f"  wl before = {wl.item():.6f}")
    print(f"  wl after  = {wl_new.item():.6f}")
    print(f"  Δ = {(wl_new - wl).item():+.6f}  ({'OK 下降' if wl_new < wl else 'BAD 没降'})")

    # 再用 Adam 跑 20 步，观察单调下降
    pos2 = bench.macro_positions.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([pos2], lr=grid_w * 0.05)
    print(f"\nAdam 20 步（lr={grid_w*0.05:.4f}）:")
    for step in range(20):
        opt.zero_grad()
        wl_step, _ = wa_wirelength(pos2, ctx, gamma=gamma, normalize=False)
        wl_step.backward()
        opt.step()
        if step % 5 == 0 or step == 19:
            print(f"  step {step:2d}: wl = {wl_step.item():.4f}")


if __name__ == "__main__":
    main()
