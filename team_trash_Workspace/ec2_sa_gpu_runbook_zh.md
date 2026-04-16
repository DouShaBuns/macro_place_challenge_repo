# EC2 运行 SA GPU 计算流程

本文档记录如何在租用的远端 EC2 实例上绑定 GitHub SSH key、拉取仓库、安装 `uv` 环境，并运行 `team_trash_Workspace/sa_gpu` 里的 SA GPU 计算案例。

本项目本地约定：使用 `uv` 运行 Python，不直接使用系统 `python` 或 `pip`。

## 0. 登录 EC2

在本地终端执行：

```bash
ssh -i /path/to/ec2-key.pem ubuntu@EC2_PUBLIC_IP
```

把下面两个占位符替换掉：

- `/path/to/ec2-key.pem`：你的 EC2 登录私钥路径。
- `EC2_PUBLIC_IP`：EC2 公网 IP 或域名。

如果 AMI 用户名不是 `ubuntu`，常见替代值包括 `ec2-user`。

## 1. 检查 GPU

登录 EC2 后先确认 GPU 和驱动可用：

```bash
nvidia-smi
```

如果这个命令不存在或看不到 GPU，先修 CUDA/driver/AMI 问题。`sa_gpu` 可以回退 CPU，但全量计算会很慢。

## 2. 安装基础工具

```bash
sudo apt update
sudo apt install -y git curl tmux
```

## 3. 在 EC2 上创建 GitHub SSH key

不要把本地私钥复制到 EC2。建议在 EC2 上创建一把专用 GitHub key：

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh

ssh-keygen -t ed25519 -C "你的GitHub邮箱" -f ~/.ssh/id_ed25519_github

eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519_github

cat ~/.ssh/id_ed25519_github.pub
```

执行到 `cat ~/.ssh/id_ed25519_github.pub` 后，复制输出的整行公钥，格式类似：

```text
ssh-ed25519 AAAA... 你的GitHub邮箱
```

## 4. 把公钥绑定到 GitHub

打开 GitHub：

```text
GitHub -> Settings -> SSH and GPG keys -> New SSH key
```

填写：

```text
Title: ec2-partcl-macro-place
Key type: Authentication Key
Key: 粘贴 cat ~/.ssh/id_ed25519_github.pub 打印出来的整行
```

添加完成后，回到 EC2 测试：

```bash
ssh -T git@github.com
```

第一次连接会提示是否信任 GitHub，输入：

```bash
yes
```

成功时会看到类似：

```text
Hi your-username! You've successfully authenticated, but GitHub does not provide shell access.
```

## 5. Clone 仓库

使用 SSH 地址 clone 仓库，并初始化子模块：

```bash
git clone --recurse-submodules git@github.com:DouShaBuns/macro_place_challenge_repo.git
cd macro_place_challenge_repo
git submodule update --init --recursive
```

如果要 clone 上游仓库，使用：

```bash
git clone --recurse-submodules git@github.com:partcleda/partcl-macro-place-challenge.git
cd partcl-macro-place-challenge
git submodule update --init --recursive
```

后续命令假设仓库目录是：

```bash
~/macro_place_challenge_repo
```

## 6. 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv --version
```

## 7. 安装项目依赖

在仓库根目录执行：

```bash
cd ~/macro_place_challenge_repo
export UV_CACHE_DIR="$PWD/.uv-cache"

uv sync --extra dev
```

安装 CUDA 版 PyTorch。下面命令使用 CUDA 12.8 wheel：

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
```

检查 torch 是否能看到 CUDA：

```bash
uv run python - <<'PY'
import torch
print("torch =", torch.__version__)
print("cuda =", torch.version.cuda)
print("cuda_available =", torch.cuda.is_available())
print("gpu =", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")
PY
```

期望看到：

```text
cuda_available = True
```

如果是 `False`，不要继续跑全量，先修 PyTorch/CUDA 环境。

## 8. 跑 smoke test

```bash
uv run --extra dev pytest team_trash_Workspace/sa_gpu/tests
```

期望结果：

```text
3 passed
```

## 9. 先跑单个 benchmark

建议先跑 `ibm01` 验证完整链路：

```bash
uv run python team_trash_Workspace/sa_gpu/parallel_runner.py \
  --benchmarks ibm01 \
  --seeds 1 2 3 4 \
  --candidate-batch 16 \
  --iters 80 \
  --devices cuda:0 \
  --out team_trash_Workspace/sa_gpu/results/ec2_ibm01.jsonl
```

查看结果：

```bash
cat team_trash_Workspace/sa_gpu/results/ec2_ibm01.jsonl
```

## 10. 用 tmux 后台跑全量

创建 tmux 会话：

```bash
tmux new -s sa
```

在 tmux 里执行：

```bash
cd ~/macro_place_challenge_repo
export UV_CACHE_DIR="$PWD/.uv-cache"

uv run python team_trash_Workspace/sa_gpu/parallel_runner.py \
  --all \
  --seeds 1 2 3 4 \
  --candidate-batch 16 \
  --iters 80 \
  --devices cuda:0 \
  --out team_trash_Workspace/sa_gpu/results/ec2_full_sa.jsonl
```

让任务继续运行并退出 tmux：

```text
Ctrl-b
d
```

重新进入 tmux：

```bash
tmux attach -t sa
```

实时查看结果文件：

```bash
tail -f team_trash_Workspace/sa_gpu/results/ec2_full_sa.jsonl
```

## 11. 多 GPU 跑法

如果 EC2 有多张 GPU，比如 `cuda:0` 和 `cuda:1`：

```bash
uv run python team_trash_Workspace/sa_gpu/parallel_runner.py \
  --all \
  --seeds 1 2 3 4 \
  --candidate-batch 16 \
  --iters 80 \
  --workers 2 \
  --devices cuda:0 cuda:1 \
  --out team_trash_Workspace/sa_gpu/results/ec2_full_sa_2gpu.jsonl
```

说明：

- `--workers` 控制 benchmark 级并行。
- `--devices` 按 worker 分配 GPU。
- 单 GPU 通常使用 `--workers 1` 更稳。
- seed 维度已经在每个 worker 内部作为 torch batch 处理，不需要为每个 seed 开一个进程。

## 12. 把结果拉回本地

在本地 Windows PowerShell 执行：

```powershell
scp -i C:\path\to\ec2-key.pem ubuntu@EC2_PUBLIC_IP:~/macro_place_challenge_repo/team_trash_Workspace/sa_gpu/results/ec2_full_sa.jsonl D:\workspace\partcl-macro-place-challenge\team_trash_Workspace\sa_gpu\results\
```

把下面两个占位符替换掉：

- `C:\path\to\ec2-key.pem`：你的 EC2 登录私钥路径。
- `EC2_PUBLIC_IP`：EC2 公网 IP 或域名。

## 13. 最小命令汇总

下面是一套从全新 EC2 到开始运行的最小命令。中间 `cat ~/.ssh/id_ed25519_github.pub` 后需要手动去 GitHub 添加公钥。

```bash
sudo apt update
sudo apt install -y git curl tmux

mkdir -p ~/.ssh
chmod 700 ~/.ssh
ssh-keygen -t ed25519 -C "你的GitHub邮箱" -f ~/.ssh/id_ed25519_github
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519_github
cat ~/.ssh/id_ed25519_github.pub
```

把公钥添加到 GitHub 后继续：

```bash
ssh -T git@github.com

git clone --recurse-submodules git@github.com:DouShaBuns/macro_place_challenge_repo.git
cd macro_place_challenge_repo
git submodule update --init --recursive

curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

export UV_CACHE_DIR="$PWD/.uv-cache"
uv sync --extra dev
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio

uv run python - <<'PY'
import torch
print("torch =", torch.__version__)
print("cuda =", torch.version.cuda)
print("cuda_available =", torch.cuda.is_available())
print("gpu =", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")
PY

uv run --extra dev pytest team_trash_Workspace/sa_gpu/tests

tmux new -s sa
```

进入 tmux 后执行：

```bash
cd ~/macro_place_challenge_repo
export UV_CACHE_DIR="$PWD/.uv-cache"

uv run python team_trash_Workspace/sa_gpu/parallel_runner.py \
  --all \
  --seeds 1 2 3 4 \
  --candidate-batch 16 \
  --iters 80 \
  --devices cuda:0 \
  --out team_trash_Workspace/sa_gpu/results/ec2_full_sa.jsonl
```

