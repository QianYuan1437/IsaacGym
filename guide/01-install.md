# 第一章：完整安装与配置

本章记录在 **Ubuntu 24.04 原生 Linux + RTX 3060** 上从零搭建环境的全过程。

### 1.1 系统环境检查

```bash
# 确认 GPU 可用
nvidia-smi
# 预期输出：NVIDIA GeForce RTX 3060, 驱动 595.84, CUDA 13.2

# 确认 conda 可用
conda --version
```

### 1.2 安装系统依赖

```bash
sudo apt update
sudo apt install -y build-essential cmake mesa-utils mesa-vulkan-drivers
```

> 本机 `libx11-dev`、`libxext-dev` 等库已通过 ROS 2 Jazzy 预先安装，无需重复操作。

### 1.3 创建 Conda 环境

```bash
# 创建环境（必须 Python 3.8，IsaacGym Preview 4 的 .so 绑定了 libpython3.8.so）
conda create -n isaacgym python=3.8 -y
conda activate isaacgym
```

### 1.4 安装 PyTorch + NumPy

```bash
# 安装 PyTorch 2.0.0 + CUDA 11.8（系统 CUDA 13.2 驱动向下兼容）
pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118

# 必须固定 NumPy 1.20，新版 API 不兼容 IsaacGym 的 reward 计算
pip install numpy==1.20.0
```

验证 CUDA：

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# True
# NVIDIA GeForce RTX 3060
```

### 1.5 安装 IsaacGym

1. 前往 [NVIDIA Developer](https://developer.nvidia.com/isaac-gym) 注册并下载 `IsaacGym_Preview_4_Package.tar.gz`（或 `.gz`）。
2. 将文件放到 `~/Download/` 或任意位置，然后解压安装：

```bash
# 解压
cp ~/Download/isaac-gym-preview-4.gz ~/
cd ~/ && tar -xzf isaac-gym-preview-4.gz

# 安装（editable 模式）
conda activate isaacgym
cd ~/isaacgym/python && pip install -e .
```

验证 GPU Physics：

```bash
python -c "
import isaacgym        # 注意：isaacgym 必须在 torch 之前 import！
import torch
from isaacgym import gymapi
gym = gymapi.acquire_gym()
p = gymapi.SimParams()
p.physx.use_gpu = True
p.use_gpu_pipeline = True
s = gym.create_sim(0, 0, gymapi.SIM_PHYSX, p)
gym.destroy_sim(s)
print('GPU Physics OK')
"
# 预期输出：+++ Using GPU PhysX
#          Physics Device: cuda:0
#          GPU Pipeline: enabled
```

### 1.6 安装 RSL-RL

```bash
mkdir -p ~/RL/code/unitree_rl_gym
cd ~/RL/code/unitree_rl_gym

git clone https://github.com/leggedrobotics/rsl_rl.git
cd rsl_rl && git checkout v1.0.2 && pip install -e .
```

### 1.7 安装 Unitree RL Gym

```bash
cd ~/RL/code/unitree_rl_gym

git clone https://github.com/unitreerobotics/unitree_rl_gym.git
cd unitree_rl_gym

# isaacgym 是本地 editable 安装，pip 依赖检测会失败，使用 --no-deps 绕过
pip install --no-deps -e .
```

> `--no-deps` 后需手动补装依赖：`pip install tensorboard matplotlib`

### 1.8 安装 MuJoCo

```bash
pip install mujoco==3.2.3
```

### 1.9 配置环境变量

#### 1.9.1 Conda 环境激活脚本

创建 `~/anaconda3/envs/isaacgym/etc/conda/activate.d/env_vars.sh`：

```bash
mkdir -p ~/anaconda3/envs/isaacgym/etc/conda/activate.d/
cat > ~/anaconda3/envs/isaacgym/etc/conda/activate.d/env_vars.sh << 'EOF'
#!/bin/bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export MUJOCO_GL=glfw
EOF
chmod +x ~/anaconda3/envs/isaacgym/etc/conda/activate.d/env_vars.sh
```

> - `$CONDA_PREFIX/lib` 使 IsaacGym 能找到 `libpython3.8.so`
> - `MUJOCO_GL=glfw` 指定 MuJoCo 渲染后端
> - **原生 Linux 不需要** `/usr/lib/wsl/lib`（WSL2 专属，CUDA 库已在标准路径）

#### 1.9.2 `~/.bashrc` 追加

在 `~/.bashrc` 末尾追加：

```bash
# ============================================================
# Robot RL Environment (Isaac Gym + MuJoCo)
# ============================================================
export MUJOCO_GL=glfw
```

### 1.10 最终验证

```bash
conda activate isaacgym
cd ~/RL/code/unitree_rl_gym/unitree_rl_gym

# 跑 3 轮训练测试（不渲染窗口）
python legged_gym/scripts/train.py --task=go2 --headless --max_iterations=3
```

看到类似输出即配置成功：

```
+++ Using GPU PhysX
Physics Device: cuda:0
GPU Pipeline: enabled
...
Computation: ~95000 steps/s
Learning iteration 3/3
```
