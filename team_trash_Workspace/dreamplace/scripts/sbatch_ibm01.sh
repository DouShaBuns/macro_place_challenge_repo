#!/bin/bash
#SBATCH --job-name=dreamplace-ibm01
#SBATCH --account=kcl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/users/k2366837/logs/dreamplace-ibm01-%j.out
#SBATCH --error=/scratch/users/k2366837/logs/dreamplace-ibm01-%j.err

# M8: ibm01 GPU 单 benchmark 验证
# - 验证 CUDA 路径跑通
# - 比较 GPU 相对 CPU 的 global placement 加速
# - 确认最终 proxy 与 CPU 一致

set -eu   # 不开 pipefail：nvidia-smi | head 会触发 SIGPIPE 误杀

cd /cephfs/volumes/hpc_home/k2366837/df4ff685-4325-4a58-8653-d8244d870233/macro_place_challenge_repo

export PATH="$HOME/.local/bin:$PATH"

# 日志
LOG_DIR=team_trash_Workspace/dreamplace/results
mkdir -p "$LOG_DIR"
export DP_LOG_PATH="$LOG_DIR/ibm01_gpu_${SLURM_JOB_ID}.jsonl"
rm -f "$DP_LOG_PATH"

echo "[info] host=$(hostname)"
echo "[info] nvidia-smi query:"
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv || true
echo

uv run python -c "import torch; print('cuda available:', torch.cuda.is_available(), 'device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo

echo "[run] evaluate ibm01"
time uv run evaluate team_trash_Workspace/dreamplace/placer.py -b ibm01

echo "[done] log at $DP_LOG_PATH"
