from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import torch


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "trace"


class PlacementTraceRecorder:
    def __init__(
        self,
        method: str,
        benchmark,
        every: int = 10,
        out_dir: str | Path = "output/traces",
        fps: int = 8,
        dpi: int = 90,
        keep_frames: bool = False,
    ):
        self.method = _safe_name(method)
        self.benchmark = benchmark
        self.every = max(int(every), 1)
        self.fps = max(int(fps), 1)
        self.dpi = max(int(dpi), 50)
        self.keep_frames = bool(keep_frames)
        self.root = Path(out_dir) / self.method / _safe_name(getattr(benchmark, "name", "benchmark"))
        self.root.mkdir(parents=True, exist_ok=True)
        self.frames_dir = self.root / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.gif_path = self.root / f"{self.method}_{_safe_name(getattr(benchmark, 'name', 'benchmark'))}.gif"
        self._frame_paths: list[Path] = []

    @classmethod
    def from_env(cls, method: str, benchmark) -> Optional["PlacementTraceRecorder"]:
        enabled = _env_bool("TRACE_PLACEMENT", False) or _env_bool("PLACEMENT_TRACE", False)
        method_key = method.upper().replace("-", "_")
        enabled = enabled or _env_bool(f"{method_key}_TRACE", False)
        if not enabled:
            return None
        return cls(
            method=method,
            benchmark=benchmark,
            every=_env_int("TRACE_EVERY", _env_int("PLACEMENT_TRACE_EVERY", 10)),
            out_dir=os.getenv("TRACE_DIR", os.getenv("PLACEMENT_TRACE_DIR", "output/traces")),
            fps=_env_int("TRACE_FPS", _env_int("PLACEMENT_TRACE_FPS", 8)),
            dpi=_env_int("TRACE_DPI", _env_int("PLACEMENT_TRACE_DPI", 90)),
            keep_frames=_env_bool("TRACE_KEEP_FRAMES", _env_bool("PLACEMENT_TRACE_KEEP_FRAMES", False)),
        )

    def should_record_step(self, step: int) -> bool:
        return int(step) % self.every == 0

    def record(self, placement: torch.Tensor, label: str) -> None:
        path = self.frames_dir / f"{len(self._frame_paths):04d}_{_safe_name(label)}.png"
        _render_frame(placement.detach().cpu(), self.benchmark, path, str(label), self.dpi)
        self._frame_paths.append(path)

    def close(self) -> Optional[Path]:
        if not self._frame_paths:
            return None
        try:
            from PIL import Image
        except ImportError:
            return None
        images = [Image.open(path).convert("P", palette=Image.ADAPTIVE) for path in self._frame_paths]
        duration_ms = max(int(1000 / self.fps), 1)
        images[0].save(
            self.gif_path,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
        )
        for image in images:
            image.close()
        if not self.keep_frames:
            for path in self._frame_paths:
                path.unlink(missing_ok=True)
            try:
                self.frames_dir.rmdir()
            except OSError:
                pass
        print(f"Saved placement trace GIF to {self.gif_path}")
        return self.gif_path


def _render_frame(placement: torch.Tensor, benchmark, path: Path, title: str, dpi: int) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.add_patch(
        Rectangle(
            (0, 0),
            float(benchmark.canvas_width),
            float(benchmark.canvas_height),
            fill=False,
            edgecolor="black",
            linewidth=1.5,
        )
    )
    num_hard = int(getattr(benchmark, "num_hard_macros", benchmark.num_macros))
    sizes = benchmark.macro_sizes.detach().cpu()
    fixed = benchmark.macro_fixed.detach().cpu()
    for i in range(num_hard):
        x = float(placement[i, 0].item())
        y = float(placement[i, 1].item())
        w = float(sizes[i, 0].item())
        h = float(sizes[i, 1].item())
        is_fixed = bool(fixed[i].item())
        ax.add_patch(
            Rectangle(
                (x - w / 2, y - h / 2),
                w,
                h,
                fill=True,
                facecolor="tomato" if is_fixed else "royalblue",
                edgecolor="black",
                linewidth=0.35,
                alpha=0.35 if is_fixed else 0.55,
            )
        )
    ax.set_xlim(0, float(benchmark.canvas_width))
    ax.set_ylim(0, float(benchmark.canvas_height))
    ax.set_aspect("equal")
    ax.set_title(f"{benchmark.name} {title}", fontsize=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
