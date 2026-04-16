# RePlAce Experiment For Partcl Macro Placement Challenge

This directory is parallel to `team_trash_Workspace/sa_gpu`.

It contains two pieces:

- The sparse-checked-out upstream OpenROAD RePlAce core files:
  - `src/`
  - `include/replace/`
  - `cmake/`
  - `doc/`
  - `CMakeLists.txt`, `README.md`, `LICENSE`, `AUTHORS.md`
- A first challenge-native Python prototype:
  - `placer.py`
  - `torch_replace.py`
  - `challenge_legalize.py`

The upstream C++ RePlAce code is built around OpenDB/OpenROAD objects, not this
challenge repository's `Benchmark` tensor API. The initial `placer.py` therefore
implements a RePlAce-style analytical placer in torch:

```text
smooth HPWL + bin density overflow
        optimized by Nesterov-like continuous updates
        followed by hard-macro legalization
```

It is intended as a starting point, not a full RePlAce port.

## Run

Use `uv`, consistent with the workspace instructions:

```powershell
uv run evaluate team_trash_Workspace/RePlAce/placer.py -b ibm01
uv run evaluate team_trash_Workspace/RePlAce/placer.py --all
```

Useful environment knobs:

```powershell
$env:REPLACE_ITERS='600'
$env:REPLACE_LR='0.08'
$env:REPLACE_DENSITY_WEIGHT='0.20'
$env:REPLACE_BIN_GRID_COUNT='32'
$env:REPLACE_DEVICE='cuda'
```

## Next Engineering Steps

1. Calibrate smooth HPWL and density against official proxy components.
2. Add a congestion/RUDY term.
3. Add an incremental/legalized local refinement stage after analytical placement.
4. Use this output as a warm start for the existing GPU SA portfolio.
5. Longer term: either build against OpenROAD/OpenDB, or write a Bookshelf/DEF
   bridge and call OpenROAD `global_placement` externally.
