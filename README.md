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

### 2.3 后台训练（SSH 断开后继续运行）

推荐使用 **tmux**——即使关闭 Windows 上的 SSH 终端，训练也不会中断，重新连接后还可以切回查看进度：

```bash
ssh user@host                  # Windows 终端下 SSH 连接

# 1. 创建 tmux 会话
tmux new -s train

# 2. 在会话内启动训练
conda activate isaacgym
cd ~/RL/code/unitree_rl_gym/unitree_rl_gym
python legged_gym/scripts/train.py --task=go2 --headless

# 3. 断开会话（训练继续运行）
#    按 Ctrl+B 然后按 D

# 4. 现在可以安全关闭 SSH 终端了

# 5. 重新 SSH 登录后，重新连接看进度
tmux attach -t train
```

**tmux 常用按键**：

| 操作 | 按键 / 命令 |
|------|------------|
| 新建会话 | `tmux new -s <名称>` |
| 断开会话 | `Ctrl+B` 然后 `D` |
| 重新连接 | `tmux attach -t <名称>` |
| 列出所有会话 | `tmux ls` |
| 关闭会话 | `tmux kill-session -t <名称>` |
| 上下翻页 | `Ctrl+B` 然后 `[`，方向键翻页，`q` 退出 |
| 横向分屏 | `Ctrl+B` 然后 `%` |
| 竖向分屏 | `Ctrl+B` 然后 `"` |
| 切换面板 | `Ctrl+B` 然后方向键 |

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

```bash
# tmux 会话管理
tmux new -s train                                # 新建会话
tmux attach -t train                             # 重新连接会话
tmux ls                                          # 查看所有会话
tmux kill-session -t train                       # 关闭会话
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
- 后台训练用 **tmux**（`tmux new -s train`），SSH 断开不中断

---

## 第六章：PPO 算法详解与参数调优

> 本章对应 `python legged_gym/scripts/train.py --task=go2 --headless` 这条训练命令背后的算法原理、仓库内的具体实现路径，以及参数调节方法。**当前 go2 训练使用的就是 PPO，无需额外切换**——见下文 6.2 的代码链路。

### 6.1 PPO 算法是什么

PPO（Proximal Policy Optimization，近端策略优化）是 OpenA 于 2017 年提出的 **on-policy** 强化学习算法，是目前四足/人形机器人仿真训练（IsaacGym + RSL-RL 生态）的默认算法。核心思想：在**收集数据**和**更新策略**之间反复循环，每次更新时限制新旧策略不要差太远。

**几个关键概念：**

| 概念 | 说明 |
|------|------|
| 策略 π(a\|s) | 给定状态输出动作分布（高斯分布：均值 μ + 标准差 σ），机器人按它采样动作 |
| 轨迹 | 一个 episode 内 (s, a, r, s') 的序列 |
| 优势函数 A(s,a) | 该动作"比平均好多少"。A > 0 鼓励该动作，A < 0 抑制 |
| on-policy | 每次更新用的数据必须由**当前**策略采集，采完即弃（PPO 不用经验回放池） |

**训练循环（每轮迭代）：**

1. **收集（Rollout）**：用当前策略在仿真环境里跑 `num_steps_per_env`（默认 24）步，记录状态、动作、奖励、价值估计；
2. **计算优势（GAE）**：用 Generalized Advantage Estimation 估计每条轨迹每个状态的 A(s,a)；
3. **更新（Update）**：把收集的数据打散成 mini-batch，反复用梯度下降更新策略和值网络；
4. 回到第 1 步，直到跑满 `max_iterations` 轮。

**PPO 的裁剪目标函数（核心公式）：**

```
L = E[ min(  r_t · A_t ,  clip(r_t, 1-ε, 1+ε) · A_t )  -  c1 · (V(s_t) - R_t)²  +  c2 · H ]
     └───────── 策略(policy)损失，clip 就是"近端"约束 ─────────┘  └─值函数损失─┘  └─熵正则─┘
```

- 概率比 `r_t = π_new(a|s) / π_old(a|s)`，衡量新旧策略变化幅度；
- `clip(r_t, 1-ε, 1+ε)` 把比值限制在 [1-ε, 1+ε]（ε = `clip_param`，默认 0.2）。如果旧策略很想做某动作而新策略不想，比值过大，clip 会**截断**其梯度，防止一次更新步子迈太大导致训练崩掉；
- `A_t` 由 GAE 计算：`δ_t = r_t + γ·(1-done)·V(s_{t+1}) - V(s_t)`，`A_t = δ_t + γ·λ·(1-done)·A_{t+1}`（γ 折扣因子，λ 权衡偏差/方差）；
- 熵正则项鼓励探索，系数 `entropy_coef`。

GAE 中 `γ=0.99, λ=0.95` 是机器人领域的常用默认值，能较好平衡估计偏差与方差。

### 6.2 本仓库中 PPO 的具体实现（代码链路）

`go2` 任务的 PPO 实现在 `rsl_rl`（独立的 RSL-RL 库），训练脚本只负责"把环境和配置接进来"。完整调用链：

```
legged_gym/scripts/train.py
  └─ task_registry.make_alg_runner(env, name="go2")        # legged_gym/utils/task_registry.py:75
       └─ 读取 GO2RoughCfgPPO（legged_gym/envs/go2/go2_config.py）
       └─ 创建 OnPolicyRunner(env, cfg, log_dir, device)   # rsl_rl/rsl_rl/runners/on_policy_runner.py
            ├─ runner 配置(runner_class_name)  → OnPolicyRunner
            ├─ policy_class_name='ActorCritic' → rsl_rl/modules/actor_critic.py
            └─ algorithm_class_name='PPO'       → rsl_rl/algorithms/ppo.py
       └─ ppo_runner.learn(max_iterations, init_at_random_ep_len=True)
```

**ActorCritic 网络（`rsl_rl/modules/actor_critic.py`）：**

- Actor：MLP `[512, 256, 128]`，输出动作均值；`self.std` 是**可学习的**初始化为 `init_noise_std=1.0` 的对数标准差参数（`actor_critic.py:85`）；
- Critic：MLP `[512, 256, 128]`，输出状态价值 V(s)；
- 激活函数 `elu`；动作分布为高斯 `Normal(mean, std)`（`actor_critic.py:123`）；
- 推理（部署）时用 `act_inference`，直接取均值（`actor_critic.py:130`）。

**PPO 类（`rsl_rl/algorithms/ppo.py:38`）关键实现：**

| 阶段 | 代码位置 | 作用 |
|------|---------|------|
| 采样动作 | `act()` ppo.py:90 | 用高斯分布采样动作，同时记录 value / log_prob / mu / sigma |
| 存轨迹 | `process_env_step()` ppo.py:104 | 奖励 + 超时(done) 引导 + 存入 RolloutStorage |
| GAE 回报 | `compute_returns()` ppo.py:116 | 调用 storage 计算 GAE 优势并归一化 |
| 自适应 KL 学习率 | ppo.py:139-151 | `schedule='adaptive'` 时按 KL 动态调学习率 |
| 策略裁剪损失 | ppo.py:155-159 | 公式见 6.1 |
| 值函数损失 | ppo.py:161-169 | 裁剪版 MSE（use_clipped_value_loss） |
| 梯度更新 | ppo.py:174-177 | Adam + `max_grad_norm` 梯度裁剪 |

**RolloutStorage（`rsl_rl/storage/rollout_storage.py`）：**

- 缓冲大小 = `num_envs × num_steps_per_env`（4096 × 24）；
- GAE 在 `compute_returns()`（rollout_storage.py:123）里**逆序**逐帧计算，`γ`/`λ` 生效，最后对 advantage 做 z-score 归一化（rollout_storage.py:137）；
- mini-batch 打散由 `mini_batch_generator()` 完成。

**训练主循环（`OnPolicyRunner.learn()`，on_policy_runner.py:80）：**

```
每轮迭代:
  1. Rollout: 24 步 × 4096 环境, 每步 act→env.step→process_env_step
  2. compute_returns: GAE 计算优势/回报
  3. update: 5 epochs × 4 mini-batches = 20 次梯度更新
  4. log() 输出训练曲线到终端 + TensorBoard
  5. 每 save_interval=50 轮保存 model_{it}.pt
```

> `init_at_random_ep_len=True`：训练开始时给每个环境随机的 episode 初始长度，避免 4096 个环境同时被 reset，制造更均匀的数据分布（on_policy_runner.py:82）。

### 6.3 配置文件在哪、如何覆盖

go2 的 PPO 配置 = `GO2RoughCfgPPO`（只改 2 项）+ 继承的 `LeggedRobotCfgPPO` 默认值：

```
legged_gym/envs/go2/go2_config.py               # GO2RoughCfgPPO —— 任务专属，改动这里
  继承于
legged_gym/envs/base/legged_robot_config.py     # LeggedRobotCfgPPO —— 所有任务共用默认值
```

当前 `GO2RoughCfgPPO` 只有：

```python
class GO2RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_go2'
```

其余所有 PPO 参数都取自基类。**想调参只需在 `GO2RoughCfgPPO` 里按同样写法覆盖对应子类**，例如：

```python
class GO2RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        learning_rate = 3.e-4          # 想从 1e-3 降到 3e-4
        num_learning_epochs = 5
        num_mini_batches = 8
        clip_param = 0.2
        gamma = 0.99
        lam = 0.95
        schedule = 'adaptive'          # 或 'fixed'
        desired_kl = 0.01
        max_grad_norm = 1.0
    class policy( LeggedRobotCfgPPO.policy ):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'
    class runner( LeggedRobotCfgPPO.runner ):
        num_steps_per_env = 24
        max_iterations = 1500
        save_interval = 50
        run_name = 'ppo_tuned'          # 改名字 → 新建独立日志目录
        experiment_name = 'rough_go2'
```

### 6.4 参数详解与调参建议

| 参数 | 默认值 | 作用 | 什么时候调 |
|------|-------|------|-----------|
| `clip_param` | 0.2 | 新旧策略概率比的裁剪边界 | 训练震荡/奖励骤降 → 调小(0.1)；学习太慢 → 调大(0.3) |
| `gamma` | 0.99 | 未来奖励折扣因子 | 一般不动；任务时序长可提高 |
| `lam` | 0.95 | GAE 偏差-方差权衡 | 一般不动；噪声大调大(0.99)，环境简单调小(0.9) |
| `entropy_coef` | 0.01 | 熵正则权重，越大探索越强 | 过早收敛/探索不足 → 调大(0.02~0.05)；动作抖动 → 调小 |
| `num_learning_epochs` | 5 | 每批数据重复更新的轮数 | 数据利用率低 → 调大(8~10)；过拟合训练数据 → 调小 |
| `num_mini_batches` | 4 | 每轮更新划分的 mini-batch 数 | 调大(8~16) 更新更平滑；显存紧张可减小 batch |
| `learning_rate` | 1e-3 | Adam 初始学习率 | 不稳定/发散 → 调小(3e-4~5e-4)；收敛太慢 → 调大 |
| `schedule` | 'adaptive' | 是否用 KL 自适应调学习率 | 'adaptive' 通常优于 'fixed'，不追求精确可固定 |
| `desired_kl` | 0.01 | 自适应调 LR 的目标 KL | KL 波动大 → 调大(0.02) |
| `value_loss_coef` | 1.0 | 值函数损失权重 | 值网络不稳定 → 调小(0.5) |
| `use_clipped_value_loss` | True | 值损失是否裁剪 | 保持 True |
| `max_grad_norm` | 1.0 | 梯度裁剪阈值 | 梯度爆炸 → 调小(0.5) |
| `init_noise_std` | 1.0 | 初始策略噪声 | 训练初期原地打转 → 调大(1.5~2.0) |
| `actor/critic_hidden_dims` | [512,256,128] | 网络宽度 | 3060 显存紧张可减半；任务复杂可加深 |
| `activation` | 'elu' | 激活函数 | 换 relu/tanh 观察效果 |
| `num_steps_per_env` | 24 | 每次收集的 rollout 步数 | 调大(48) 更稳定但更慢 |
| `max_iterations` | 1500 | 总训练迭代数 | 不够收敛就加大 |
| `save_interval` | 50 | 每多少轮保存模型 | 默认即可 |

**快速诊断表：**

| 现象 | 优先尝试 |
|------|---------|
| Mean reward 不涨 / 曲线发散 | 调小 `learning_rate` → 调小 `clip_param` → 检查 reward 定义 |
| 训练太慢 / 卡在局部最优 | 调大 `entropy_coef`、`init_noise_std`，调大 `num_learning_epochs` |
| 动作抖动、输出不稳 | 调小 `entropy_coef`，调大 `max_grad_norm` 的裁剪 |
| 显存不足 (OOM) | 减小 `--num_envs`（3060 建议 ≤ 4096）或 `actor_hidden_dims` |
| 训练早期奖励骤降 | 调小 `clip_param` / `learning_rate`，启用/检查 `schedule='adaptive'` |

### 6.5 无需改代码的调参方式

`legged_gym/utils/helpers.py:122` 的 `get_args()` 提供命令行覆盖（**有限**，主要覆盖 runner 部分）：

```bash
cd ~/RL/code/unitree_rl_gym/unitree_rl_gym

# 只改迭代数和随机种子（新 run，不动原日志）
python legged_gym/scripts/train.py --task=go2 --headless --max_iterations=2000 --seed=42

# 改日志实验名/run 名（开独立目录，方便对比不同超参）
python legged_gym/scripts/train.py --task=go2 --headless --experiment_name=rough_go2 --run_name=ppo_try2
```

更细粒度的超参（learning_rate、entropy_coef 等）**不支持命令行**，只能改 `go2_config.py`（见 6.3）。

### 6.6 进阶：LSTM-PPO（循环策略）

如果任务需要记忆历史信息（如复杂地形、历史状态），可切到循环网络 ActorCriticRecurrent（`rsl_rl/modules/actor_critic_recurrent.py`），支持 LSTM/GRU：

```python
class GO2RoughCfgPPO( LeggedRobotCfgPPO ):
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCriticRecurrent'
        algorithm_class_name = 'PPO'
    class policy( LeggedRobotCfgPPO.policy ):
        rnn_type = 'lstm'          # 或 'gru'
        rnn_hidden_size = 512
        rnn_num_layers = 1
```

注意：MLP 策略（ActorCritic）一般已足够，LSTM 训练更慢、对时序数据更敏感，非必要不启用。

### 6.7 续训 / 加载模型

已完成 run 的模型在 `logs/rough_go2/<时间戳>_<run_name>/model_*.pt`。续训：

```bash
# 从指定 checkpoint 继续（注意 max_iterations 是"总迭代数"，续训要写更大的值）
python legged_gym/scripts/train.py --task=go2 --headless --resume --load_run=Aug04_21-40-48_ --checkpoint=1500 --max_iterations=3000

# 或只加 --resume，自动加载该实验目录下最新 run 的最新模型
python legged_gym/scripts/train.py --task=go2 --headless --resume --max_iterations=3000
```

### 6.8 如何监控训练

终端每轮输出关键指标（on_policy_runner.py:157）：

```
Computation: ~95000 steps/s
Value function loss: 0.1234
Surrogate loss: 0.0056
Mean action noise std: 0.48
Mean reward: 12.34
Mean episode length: 12.56
```

| 指标 | 含义 | 预期 |
|------|------|------|
| Mean reward | 每个完成的 episode 平均总奖励 | 训练中应单调上升并趋平 |
| Mean episode length | 平均步长（截断时≈训练轮数内最大） | 应趋于接近 1s/0.005s=200 步（满 episode） |
| Value function loss | 值网络回归误差 | 应逐渐下降、平稳 |
| Mean action noise std | 策略当前噪声标准差（从 1.0 开始衰减） | 应随训练下降，说明策略趋于确定 |
| Surrogate loss | 策略裁剪损失 | 小且平稳即可，无需归零 |

曲线可视化：

```bash
tensorboard --logdir=~/RL/code/unitree_rl_gym/unitree_rl_gym/logs/
```

浏览器访问 `http://localhost:6006`，关注的标签：`Train/mean_reward`、`Train/mean_episode_length`、`Loss/*`、`Policy/mean_noise_std`、`Loss/learning_rate`（adaptive 下应看到自适应变化）。
