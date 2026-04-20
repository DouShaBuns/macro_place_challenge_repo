#!/bin/bash
#SBATCH --job-name=dreamplace-m9
#SBATCH --account=kcl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/users/k2366837/logs/dreamplace-m9-%j.out
#SBATCH --error=/scratch/users/k2366837/logs/dreamplace-m9-%j.err

# M9: 三 benchmark 对照（ibm01 / ibm04 / ibm09）串行 GPU 跑
# 决定是否进入 M10（17 全跑）
set -eu

cd /cephfs/volumes/hpc_home/k2366837/df4ff685-4325-4a58-8653-d8244d870233/macro_place_challenge_repo
export PATH="$HOME/.local/bin:$PATH"

# Deterministic 要求，必须在 python 启动前设
export CUBLAS_WORKSPACE_CONFIG=:4096:8

LOG_DIR=team_trash_Workspace/dreamplace/results
mkdir -p "$LOG_DIR"
export DP_LOG_PATH="$LOG_DIR/m9_${SLURM_JOB_ID}.jsonl"
rm -f "$DP_LOG_PATH"

echo "[info] host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
echo

for b in ibm01 ibm04 ibm09; do
  echo "============================================================"
  echo "[run] $b"
  echo "============================================================"
  time uv run evaluate team_trash_Workspace/dreamplace/placer.py -b $b
  echo
done

echo "[done] log at $DP_LOG_PATH"
