# 机器人强化学习环境配置与训练指南

基于 NVIDIA IsaacGym + RSL-RL + Unitree RL Gym + MuJoCo 的完整操作手册。

---

## 环境信息

| 项目 | 详情 |
|------|------|
| 操作系统 | Ubuntu 24.04 LTS（原生 Linux，非 WSL2） |
| GPU | NVIDIA GeForce RTX 3060 (12GB) |
| 驱动版本 | 595.84 |
| CUDA | 13.2 (系统驱动) / 11.8 (PyTorch，向下兼容) |
| Conda | Anaconda3 (`/home/zhao/anaconda3/`) |
| Conda 环境名 | `isaacgym` |
| Python | 3.8.20 |
| PyTorch | 2.0.0+cu118 |
| IsaacGym | Preview 4 |
| RSL-RL | v1.0.2 |
| MuJoCo | 3.2.3 |
| NumPy | 1.20.0 |

---

## 目录结构

```
~/
├── isaacgym/                                    # IsaacGym Preview 4 解压目录
├── Download/isaac-gym-preview-4.gz              # IsaacGym 安装包（需手动下载）
└── RL/
    ├── README.md                                # 本文件
    └── code/unitree_rl_gym/
        ├── rsl_rl/                              # RSL-RL (PPO 算法实现)
        └── unitree_rl_gym/                      # Unitree RL Gym (宇树机器人训练模板)
              ├── legged_gym/
              │   ├── envs/                      # 机器人配置 (go2/g1/h1/h1_2)
              │   └── scripts/
              │       ├── train.py               # 训练脚本
              │       └── play.py                # IsaacGym 查看器演示
              ├── deploy/
              │   ├── deploy_mujoco/              # MuJoCo 部署与验证
              │   │   ├── deploy_mujoco.py
              │   │   └── configs/               # g1.yaml, h1.yaml, h1_2.yaml
              │   ├── deploy_real/               # 实物部署
              │   └── pre_train/                 # 预训练模型 (g1, h1, h1_2)
              └── resources/robots/              # 机器人模型文件
```

---

## 第一章：完整安装与配置

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

---

## 第二章：训练

### 支持的机器人

| 参数 | 机器人 | 类型 |
|------|--------|------|
| `--task=go2` | Go2 | 四足机器狗 |
| `--task=g1` | G1 | 双足人形机器人 |
| `--task=h1` | H1 | 双足人形机器人 |
| `--task=h1_2` | H1 v2 | 双足人形机器人 |

### 2.1 完整训练

```bash
conda activate isaacgym
cd ~/RL/code/unitree_rl_gym/unitree_rl_gym

# 训练 Go2 机器狗（默认 1500 轮）
python legged_gym/scripts/train.py --task=go2 --headless

# 训练 G1 人形机器人
python legged_gym/scripts/train.py --task=g1 --headless

# 训练 H1 人形机器人
python legged_gym/scripts/train.py --task=h1 --headless

# 训练 H1 v2
python legged_gym/scripts/train.py --task=h1_2 --headless
```

### 2.2 常用参数

```bash
# 限制训练轮数（测试用）
python legged_gym/scripts/train.py --task=go2 --headless --max_iterations=500

# 从上次 checkpoint 继续训练
python legged_gym/scripts/train.py --task=go2 --headless --resume

# 显示渲染窗口（可能不稳定，推荐用 MuJoCo 查看结果）
python legged_gym/scripts/train.py --task=go2
```

### 2.3 后台训练

```bash
conda activate isaacgym
cd ~/RL/code/unitree_rl_gym/unitree_rl_gym

# 使用 setsid 启动，关闭终端后仍运行
setsid python legged_gym/scripts/train.py --task=go2 --headless \
    > ~/train_go2.log 2>&1 < /dev/null &

# 查看日志
tail -f ~/train_go2.log

# 查看进程
ps aux | grep train.py

# 停止训练
kill <PID>
```

### 2.4 查看训练曲线

```bash
tensorboard --logdir=~/RL/code/unitree_rl_gym/unitree_rl_gym/logs/
```

浏览器访问 `http://localhost:6006`。

### 2.5 模型保存位置

训练完成后模型保存在：

```
~/RL/code/unitree_rl_gym/unitree_rl_gym/logs/rough_go2/
```

---

## 第三章：查看训练效果

### 3.1 IsaacGym 查看器 (Go2)

Go2 可使用 IsaacGym 原生查看器演示训练结果：

```bash
conda activate isaacgym
cd ~/RL/code/unitree_rl_gym/unitree_rl_gym

python legged_gym/scripts/play.py --task=go2
```

> 加载最新训练的模型，使用 IsaacGym 渲染窗口展示机器人行走效果。

### 3.2 MuJoCo 部署验证 (G1 / H1)

G1 和 H1 支持 MuJoCo 渲染验证，已提供预训练模型：

```bash
conda activate isaacgym
cd ~/RL/code/unitree_rl_gym/unitree_rl_gym/deploy/deploy_mujoco

# G1 人形机器人
python deploy_mujoco.py g1.yaml

# H1 人形机器人
python deploy_mujoco.py h1.yaml

# H1 v2
python deploy_mujoco.py h1_2.yaml
```

> 如果自己训练了 G1/H1 模型，修改 `configs/*.yaml` 中的 `policy_path` 指向你的 `.pt` 文件。

---

## 第四章：常用快捷操作

```bash
conda activate isaacgym                          # 激活环境
conda deactivate                                 # 退出环境
ps aux | grep train.py                           # 查看训练进程
kill <PID>                                       # 停止训练
tensorboard --logdir=logs/                       # 启动 TensorBoard
```

---

## 第五章：注意事项与原理解释

### 5.1 Python 3.8 必须固定

IsaacGym Preview 4 的预编译 `.so` 文件绑定 `libpython3.8.so.1.0`，使用其他 Python 版本会报：

```
ImportError: libpython3.8.so.1.0: cannot open shared object file
```

### 5.2 import 顺序

`import isaacgym` 必须在 `import torch` 之前，否则报错：

```
ImportError: PyTorch was imported before isaacgym modules
```

Unitree RL Gym 的训练脚本已正确处理此顺序。

### 5.3 NumPy 版本

`numpy==1.20.0` — IsaacGym 的 reward 计算使用了旧版 NumPy API（如 `np.float` 别名），新版会报 `AttributeError`。

### 5.4 WSL2 与本环境（原生 Linux）的配置差异

| 配置项 | WSL2 | 本环境（原生 Linux） |
|--------|------|---------------------|
| `LD_LIBRARY_PATH` | 需含 `/usr/lib/wsl/lib` | 不需要（CUDA 在标准路径） |
| GPU 渲染 | 需 D3D12 翻译层 (`GALLIUM_DRIVER=d3d12`) | 原生 OpenGL，无需额外配置 |
| MuJoCo 渲染 | 默认软渲染 (llvmpipe)，需手动切 GPU | 默认 GPU 渲染 |
| `cuda.so` 位置 | `/usr/lib/wsl/lib/` | `/usr/lib/x86_64-linux-gnu/` 等标准路径 |
| Wayland 兼容 | 需 `unset WAYLAND_DISPLAY` | 本机用 X11，无需此设置 |

### 5.5 训练建议

- 全程使用 `--headless`，渲染窗口可能导致崩溃，且拖慢训练速度
- 查看效果用 MuJoCo（G1/H1）或 IsaacGym play.py（Go2）
- 后台训练用 `setsid` 或 `screen`/`tmux`
