# DREAMPlace 风格 GPU 解析式 Placer

完整设计与里程碑：见 [`PLAN_zh.md`](PLAN_zh.md)。

## 运行

```bash
# 单 benchmark
uv run evaluate team_trash_Workspace/dreamplace/placer.py -b ibm01

# 全部 17 IBM
uv run evaluate team_trash_Workspace/dreamplace/placer.py --all
```

## 当前状态

v1 已完成（2026-04-20）。

| 里程碑 | 状态 |
|---|---|
| M0 工程骨架 | ✅ |
| M1 WA wirelength | ✅ |
| M2 eDensity | ✅ |
| M3 训练循环 | ✅ |
| M4 Legalization | ✅ |
| M5 Detail | ✅ |
| M6 Pipeline | ✅ |
| M7 Smoke test | ✅ |
| M8 ibm01 调通 | ✅ |
| M9 3-bench 对照 | ✅ |
| M10 17 全跑 | ✅ |

## v1 结果

17 IBM benchmark 平均 proxy = **1.3214**，vs RePlAce baseline (1.4578) **领先 9.4%**，vs SA baseline (2.1251) 领先 37.8%。17/17 VALID（零重叠）。

完整结果表 + v2 TODO：见 [`PLAN_zh.md` §9–10](PLAN_zh.md)。
