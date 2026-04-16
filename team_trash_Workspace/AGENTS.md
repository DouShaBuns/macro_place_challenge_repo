# Workspace Instructions

- 本地使用 `uv` 来运行 Python，不要直接使用 Python 自带的 `pip` 或裸 `python`。
- 在运行任何 `uv` 命令前，先激活本 workspace 的虚拟环境：

```powershell
..\.venv\Scripts\Activate.ps1
```

从仓库根目录执行时使用：

```powershell
.\.venv\Scripts\Activate.ps1
```

- 激活后再运行命令，例如：

```powershell
uv run evaluate team_trash_Workspace/sa_gpu/placer.py -b ibm01
```
