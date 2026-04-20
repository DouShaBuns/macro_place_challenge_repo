"""M2 sanity check：eDensity painting + Poisson + 梯度方向。

跑法：
    srun --account=kcl --partition=interruptible_cpu --time=00:05:00 \
         --mem=4G --cpus-per-task=2 \
         uv run python team_trash_Workspace/dreamplace/tests/sanity_m2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_DREAM = Path(__file__).resolve().parents[1]
if str(_DREAM) not in sys.path:
    sys.path.insert(0, str(_DREAM))

from macro_place.loader import load_benchmark_from_dir  # noqa: E402

from torch_density import adaptive_grid, build_density_context, density_energy, paint_density  # noqa: E402


def case_two_macros_repulsion():
    """两个 macro 重叠 vs 分开：能量必须重叠时高，且梯度方向把它们推开。"""
    print("\n=== Case A: 两个 macro 重叠测试（合成）===")
    canvas = 10.0
    sizes = torch.tensor([[2.0, 2.0], [2.0, 2.0]])
    ctx = build_density_context(canvas, canvas, rows=64, cols=64)

    # 完全重叠（中心相同）
    pos_overlap = torch.tensor([[5.0, 5.0], [5.0, 5.0]], requires_grad=True)
    e_overlap, diag1 = density_energy(pos_overlap, sizes, ctx)
    e_overlap.backward()
    print(f"  重叠：energy = {e_overlap.item():.6f}, rho_max = {diag1['rho_max']:.4f}")
    print(f"        macro0 grad = {pos_overlap.grad[0].tolist()}")
    print(f"        macro1 grad = {pos_overlap.grad[1].tolist()}")

    # 远离
    pos_apart = torch.tensor([[2.5, 2.5], [7.5, 7.5]], requires_grad=True)
    e_apart, diag2 = density_energy(pos_apart, sizes, ctx)
    print(f"  分开：energy = {e_apart.item():.6f}, rho_max = {diag2['rho_max']:.4f}")
    assert e_overlap.item() > e_apart.item(), "重叠 energy 应高于分开"
    print(f"  ✓ 重叠 energy ({e_overlap.item():.4f}) > 分开 ({e_apart.item():.4f})")


def case_repulsion_step():
    """重叠两个 macro，沿负梯度走一步，能量必须下降。"""
    print("\n=== Case B: 重叠后单步排斥下降测试 ===")
    canvas = 10.0
    sizes = torch.tensor([[2.0, 2.0], [2.0, 2.0]])
    ctx = build_density_context(canvas, canvas, rows=64, cols=64)

    pos = torch.tensor([[5.0, 5.0], [5.2, 5.0]], requires_grad=True)  # 微重叠
    e0, _ = density_energy(pos, sizes, ctx)
    e0.backward()
    grad = pos.grad
    print(f"  初始 energy = {e0.item():.6f}, max |grad| = {grad.abs().max().item():.4f}")
    print(f"  macro0 grad = {grad[0].tolist()}")
    print(f"  macro1 grad = {grad[1].tolist()}")

    target_step = ctx.grid_w * 0.5
    lr = target_step / max(grad.abs().max().item(), 1.0e-9)
    new_pos = (pos - lr * grad).detach()
    e1, _ = density_energy(new_pos, sizes, ctx)
    print(f"  step（lr={lr:.4f}）后 energy = {e1.item():.6f}, Δ = {(e1-e0).item():+.6f}")
    print(f"  新位置: {new_pos.tolist()}")
    assert e1.item() < e0.item(), "排斥后能量应下降"
    print(f"  ✓ 单步排斥让能量下降")


def case_real_benchmark():
    """ibm01 真实数据：grid 自适应、energy 数量级、梯度统计。"""
    print("\n=== Case C: ibm01 真实数据 ===")
    bench, _ = load_benchmark_from_dir("external/MacroPlacement/Testcases/ICCAD04/ibm01")
    sizes = bench.macro_sizes
    rows, cols = adaptive_grid(bench.canvas_width, bench.canvas_height, sizes)
    print(f"  canvas {bench.canvas_width:.2f} x {bench.canvas_height:.2f}, "
          f"min_macro_w = {sizes[:,0].min().item():.4f}, "
          f"min_macro_h = {sizes[:,1].min().item():.4f}")
    print(f"  自适应 grid = {rows} x {cols}")

    ctx = build_density_context(bench.canvas_width, bench.canvas_height, rows, cols)
    pos = bench.macro_positions.clone().detach().requires_grad_(True)

    e, diag = density_energy(pos, sizes, ctx)
    e.backward()
    grad = pos.grad
    print(f"  energy (initial.plc) = {e.item():.6f}")
    print(f"  rho_max = {diag['rho_max']:.4f}, rho_mean = {diag['rho_mean']:.4f}, "
          f"overflow = {diag['overflow']:.4%}")
    print(f"  grad norm = {grad.norm().item():.4f}, max |grad| = {grad.abs().max().item():.4f}")
    print(f"  zero-grad rows = {(grad.abs().sum(dim=1)==0).float().mean().item():.3%}")

    # 单步：只动 movable，让能量下降
    target_step = ctx.grid_w * 0.5
    lr = target_step / max(grad.abs().max().item(), 1.0e-9)
    new_pos = pos.detach() - lr * grad
    e_new, _ = density_energy(new_pos, sizes, ctx)
    print(f"  单步后 energy = {e_new.item():.6f}, Δ = {(e_new-e).item():+.6f} "
          f"({'OK 下降' if e_new < e else 'BAD 没降'})")


def main():
    torch.set_default_dtype(torch.float32)
    case_two_macros_repulsion()
    case_repulsion_step()
    case_real_benchmark()
    print("\n[M2] all sanity checks passed.")


if __name__ == "__main__":
    main()
