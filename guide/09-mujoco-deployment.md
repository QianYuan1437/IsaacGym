# 第九章：MuJoCo 部署运行（sim-to-sim）

> 把 IsaacGym 训练好的策略导出为 TorchScript，放入 MuJoCo 仿真中运行，用于脱离 IsaacGym 环境的快速验证、真实物理对照与后续实物移植。

## 9.1 为什么需要 sim-to-sim

IsaacGym 的渲染与刚体动力学对训练足够，但：

- 依赖 IsaacGym 运行时与 GPU，验证不够轻量；
- 与真实机器人/其他仿真（MuJoCo、真实硬件）之间仍有**动力学与解算器差异**，需要提前暴露 sim-to-sim 差距；
- MuJoCo 生态（`mujoco` Python 包、URDF/MJCF 转换）便于后续接仿真器测试或实物部署。

本项目 G1 的 IsaacGym 训练模型（`g1_12dof.urdf`）与 MuJoCo 模型（`g1_12dof.xml`）共用同一运动学/关节定义，仅需保证**关节顺序、默认姿态、归一化尺度、观测构造**一致即可无缝迁移。

## 9.2 部署链路总览

```
训练 (train.py) → 检查点 model_XXXX.pt
    → play.py 导出 TorchScript（policy_lstm_1.pt / policy_1.pt）
    → deploy_mujoco.py 加载 .pt + scene.xml，PD 控制 + 策略推理
    → MuJoCo 仿真窗口 / headless 冒烟
```

涉及文件：

```
deploy/deploy_mujoco/
├── deploy_mujoco.py            # MuJoCo 运行主脚本（支持 --headless）
└── configs/
    ├── g1.yaml                 # 官方 G1 部署配置
    ├── h1.yaml / h1_2.yaml     # 官方 H1 系列
    └── g1_sc.yaml              # 本项目自碰撞策略部署配置（新增）
```

## 9.3 导出策略

`play.py` 在加载最新检查点后会调用 `export_policy_as_jit()`：

- **MLP 策略** → `logs/<exp>/exported/policies/policy_1.pt`
- **LSTM 策略**（g1 / g1_sc 均为 `ActorCriticRecurrent`）→ `logs/<exp>/exported/policies/policy_lstm_1.pt`，由 `PolicyExporterLSTM` 封装 actor + LSTM 记忆（内部维护 hidden/cell state，无需外部管理）

```bash
conda activate isaacgym
cd ~/RL/code/unitree_rl_gym/unitree_rl_gym

# 导出 g1_sc 策略（--headless 无窗口；--num_envs=1 只加载 1 个机器人）
python legged_gym/scripts/play.py --task=g1_sc --headless --num_envs=1
# → logs/g1_sc/exported/policies/policy_lstm_1.pt
```

## 9.4 在 MuJoCo 中运行

```bash
# 图形窗口（需显示器）
python deploy/deploy_mujoco/deploy_mujoco.py g1_sc.yaml

# 无显示器 / CI 冒烟测试
python deploy/deploy_mujoco/deploy_mujoco.py g1_sc.yaml --headless
```

`deploy_mujoco.py` 流程：

1. 读配置（`policy_path`、`xml_path`、PD 增益、归一化尺度、命令等）；
2. `MjModel.from_xml_path(scene.xml)` 加载 G1 模型（19 qpos / 18 qvel / 12 关节驱动）；
3. `torch.jit.load(policy_lstm_1.pt)` 加载策略；
4. 每个控制周期（`control_decimation=10` × `simulation_dt=0.002s`，即 50 Hz）：
   - 构造 47 维观测：`[ω·0.25 (3), 投影重力 (3), cmd·[2,2,0.25] (3), (q-default)·1.0 (12), q̇·0.05 (12), 上一动作 (12), sin/cos 相位 (2)]`（相位周期 0.8 s，与训练 `g1_env.py` 一致）；
   - 策略推理得到 12 维动作，映射为目标关节角 `target = action·0.25 + default_angles`；
   - PD 控制器输出力矩：`τ = kp·(target-q) + kd·(0-q̇)`。

## 9.5 g1_sc 部署配置说明

`configs/g1_sc.yaml` 与官方 `g1.yaml` 唯一区别是 `policy_path` 指向自碰撞策略的导出文件，其余参数（PD 增益、`default_angles`、`action_scale=0.25`、`cmd_scale=[2,2,0.25]`、`num_obs=47`、`num_actions=12`）与训练配置完全一致：

```yaml
policy_path: "{LEGGED_GYM_ROOT_DIR}/logs/g1_sc/exported/policies/policy_lstm_1.pt"
xml_path:    "{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/scene.xml"
simulation_duration: 60.0
simulation_dt: 0.002
control_decimation: 10
kps: [100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40]
kds: [2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2]
default_angles: [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0]
ang_vel_scale: 0.25
dof_pos_scale: 1.0
dof_vel_scale: 0.05
action_scale: 0.25
cmd_scale: [2.0, 2.0, 0.25]
num_actions: 12
num_obs: 47
cmd_init: [0.5, 0, 0]   # 初始指令：x 向 0.5 m/s 前进
```

> 注：MuJoCo 关节顺序（左/右各 hip_pitch → hip_roll → hip_yaw → knee → ankle_pitch → ankle_roll）与训练 URDF 一致，`d.qpos[7:]` / `d.qvel[6:]` 的切片假设与此对应，勿随意改序。

## 9.6 冒烟验证结果（g1_sc，headless 5 s）

在无显示器环境运行 `deploy_mujoco.py --headless`（把 `simulation_duration` 临时改为 5 s）逐秒采样：

| 时刻 | 基座高度 | 前进速度 | 动作 NaN | 状态 NaN |
|------|---------|---------|---------|---------|
| 1.0 s | 0.775 m | 0.48 m/s | 否 | 否 |
| 2.0 s | 0.751 m | 0.63 m/s | 否 | 否 |
| 3.0 s | 0.769 m | 0.46 m/s | 否 | 否 |
| 4.0 s | 0.774 m | 0.41 m/s | 否 | 否 |
| 5.0 s | 0.772 m | 0.46 m/s | 否 | 否 |

结论：策略在 MuJoCo 中稳定行走（高度 ~0.77 m、速度 ~0.5 m/s 与指令一致），无 NaN，说明模型正确加载且 obs 构造/归一化与训练对齐。

## 9.7 常见问题

| 现象 | 原因 / 解决 |
|------|------------|
| `FileNotFoundError ... configs/...` | `deploy_mujoco.py` 会把配置路径拼到 `configs/` 目录下，传配置文件名而非绝对路径 |
| MuJoCo 报模型加载失败 | 检查 `xml_path` 中的相对 include 与 mesh 路径；`scene.xml` include 的 `g1_12dof.xml` 需在其同目录 |
| 无显示器报 viewer 错误 | 加 `--headless` 参数运行 |
| 机器人直接摔倒 / 原地抖动 | 检查 `default_angles`、`action_scale`、归一化尺度是否与训练配置一致；确认 `policy_path` 指向正确的导出文件 |
