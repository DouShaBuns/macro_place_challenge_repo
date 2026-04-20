#!/bin/bash
#SBATCH --job-name=dreamplace-m10
#SBATCH --account=kcl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/users/k2366837/logs/dreamplace-m10-%j.out
#SBATCH --error=/scratch/users/k2366837/logs/dreamplace-m10-%j.err

# M10: 17 IBM benchmarks 全跑
# 评测器 --all 自动遍历并给出 summary 表格（vs SA / vs RePlAce / overlaps）
# 预计 ~45 分钟（大 benchmark 受 detail time_budget=120s 约束）
set -eu

cd /cephfs/volumes/hpc_home/k2366837/df4ff685-4325-4a58-8653-d8244d870233/macro_place_challenge_repo
export PATH="$HOME/.local/bin:$PATH"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

LOG_DIR=team_trash_Workspace/dreamplace/results
mkdir -p "$LOG_DIR"
export DP_LOG_PATH="$LOG_DIR/m10_${SLURM_JOB_ID}.jsonl"
rm -f "$DP_LOG_PATH"

echo "[info] host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
echo

time uv run evaluate team_trash_Workspace/dreamplace/placer.py --all

echo
echo "[done] jsonl log at $DP_LOG_PATH"
