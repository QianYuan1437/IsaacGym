# 附录：G1 自碰撞避免复现详细指南

> 本章由 `legged_gym/README_SELF_COLLISION.md` 迁入，作为 GitBook 附录章节。

---

---


本指南在当前的 IsaacGym + RSL-RL + legged_gym 环境上，用 **G1 人形机器人** 复现该论文的核心思想：**在行走训练中避免机器人自身部件之间的碰撞**，并把论文基于控制障碍函数（CBF）的安全保证思想转化为 PPO 强化学习训练中的奖励塑形。

---

## 一、论文核心思想

论文面向 MIT Humanoid（18 关节）提出 **CBF-WBC**（基于控制障碍函数的全身控制器）：

1. **几何近似**：用 4 个球 + 14 个胶囊近似机器人几何，共考虑 **15 个碰撞体对**；
2. **有符号距离（SDF）**：对每个碰撞体对 AB 定义
   `h_AB(q) = ‖pA - pB‖ - (ρA + ρB)`（球/胶囊中心距减去两者半径之和）；
3. **CBF 约束**：把"永不碰撞"编码为前向不变性约束 `ḣ + α·h ≥ 0`，作为 QP 求解全身控制量的硬约束；
4. **对比基线（APF）**：人工势场法，当距离 `d < d0` 时施加 `K·(d0-d)²` 型软惩罚，仅在阈值内生效。

结论：CBF 相比 APF 无需精确调节 `(K, d0)` 两个参数，且在高动态场景（如受外力推搡后恢复、摆腿轨迹与支撑腿交叉）下仍能保证无碰撞。

## 二、论文 → RL 的映射

论文是**控制器**（QP 求解），本仓库是 **PPO 强化学习**。我们把安全保证翻译成奖励塑形：

| 论文要素 | 论文实现 | 本复现（RL）实现 |
|---------|---------|----------------|
| 安全集 `h(q) ≥ 0` | CBF 硬约束 | 碰撞体对中心距 `d` 与安全距离 `safe` 之差 |
| CBF 约束 `ḣ+αh≥0` | QP 不等式 | 指数势垒惩罚 + 接近速率项（见 `_reward_self_collision_cbf`） |
| APF 软惩罚（Eq.23） | QP 软约束 | 位置型二次惩罚（见 `_reward_self_collision_apf`） |
| 关节限位 CBF | QP 约束 | legged_gym 自带 `dof_pos_limits` / `soft_dof_pos_limit` |

核心公式：

```
CBF 势垒惩罚（论文方法）:
    penalty_pair = exp(-barrier_alpha * (d - safe))  *  (1 + approach_beta * max(0, -d_dot))
    d_dot = (d - d_prev) / dt      # 接近速率为负
    -> d == safe 时惩罚≈1；d → 0 时指数增长；两部件快速接近时额外加重

APF 基线惩罚（论文 Eq.23）:
    penalty_pair = max(0, safe - d)²     # 仅在 d < safe 时生效
```

`d_dot` 项正是论文 CBF 中 `ḣ + αh ≥ 0` 的动力学含义：**不仅惩罚距离过近，还惩罚"正在高速接近"**，这是 CBF 优于纯位置 APF 的关键，也是我们复现的重点。

## 三、新增文件

| 文件 | 作用 |
|------|------|
| `legged_gym/envs/g1/g1_self_collision.py` | 环境类 `G1SelfCollisionRobot`：解析碰撞体对、计算中心距、实现 CBF/APF 两个奖励函数 |
| `legged_gym/envs/g1/g1_self_collision_config.py` | 配置类 `G1SelfCollisionRoughCfg` / `G1SelfCollisionRoughCfgPPO`：碰撞体对、安全距离、奖励权重、PPO 参数 |
| `legged_gym/envs/__init__.py` | 注册新任务 `g1_sc` |

## 四、代码说明

### 4.1 环境类 `G1SelfCollisionRobot`（`g1_self_collision.py`）

继承 `G1Robot`，只增加自碰撞相关逻辑，其余（观测、动作、地形、奖励框架）完全复用：

```python
def _init_self_collision(self):
    # 从 cfg.self_collision.pairs 逐个解析：
    #   gym.find_actor_rigid_body_handle(env, actor, body_name)
    # 找不到的体（被 collapse_fixed_joints 合并掉）自动跳过
    ...
    self.sc_idx_a / self.sc_idx_b   # 碰撞体对 a、b 的 rigid body 索引
    self.sc_safe                    # 每对的安全距离 [m]
    self.sc_dist_prev               # 上一步距离，用于有限差分接近速率

def _sc_distances(self):
    # 用 rigid_body_states_view（每步由 update_feet_state 刷新）取体位置
    pos = self.rigid_body_states_view[:, :, :3]
    return torch.norm(pos[:, self.sc_idx_a, :] - pos[:, self.sc_idx_b, :], dim=2)
```

奖励函数（名字与 config 中 `rewards.scales` 的 key 一一对应）：

```python
def _reward_self_collision_cbf(self):     # 论文方法（CBF 势垒）
    d = self._sc_distances()
    d_dot = clamp((d - self.sc_dist_prev) / self.dt, ±max_approach_rate)
    self.sc_dist_prev = d
    barrier = exp(-barrier_alpha * (d - self.sc_safe))
    approach = 1.0 + approach_beta * relu(-d_dot)
    return sum(barrier * approach, dim=1)

def _reward_self_collision_apf(self):     # 基线（论文 Eq.23）
    d = self._sc_distances()
    return sum(relu(self.sc_safe - d) ** 2, dim=1)
```

`reset_idx` 中会把 `sc_dist_prev` 重置为当前距离，避免环境复位导致的距离跳变污染接近速率项。

### 4.2 配置类 `G1SelfCollisionRoughCfg`（`g1_self_collision_config.py`）

关键设置：

```python
class rewards(G1RoughCfg.rewards):
    only_positive_rewards = False          # ★ 让负的惩罚项真正生效
    class scales(G1RoughCfg.rewards.scales):
        collision = -0.5                   # 原有接触力惩罚
        self_collision_cbf = -0.5          # CBF 势垒惩罚（论文方法）
        self_collision_apf = 0.0           # APF 基线（对比时再开启）

class self_collision:
    safe_distance    = 0.10    # 默认安全距离 [m]
    barrier_alpha    = 8.0     # 势垒陡峭度
    approach_beta    = 1.0     # 接近速率项权重
    max_approach_rate = 2.0    # 接近速率上限 [m/s]
    pairs = [
        ('left_hip_pitch_link',  'right_hip_pitch_link',  0.12),  # 大腿-大腿
        ('left_knee_link',       'right_knee_link',       0.10),  # 小腿-小腿
        ('left_hip_pitch_link',  'right_knee_link',       0.10),  # 大腿-小腿（摆腿交叉）
        ('right_hip_pitch_link', 'left_knee_link',        0.10),  # 大腿-小腿
        # 以下臂部对在 12 关节 URDF 中不存在，会被自动跳过；
        # 若换成带臂的 g1_29dof 会自动激活
        ('left_elbow_link', 'torso_link', 0.10), ...
    ]
```

> **为什么必须 `only_positive_rewards = False`？**
> legged_gym 基类默认 `only_positive_rewards = True`，`compute_reward()` 会把总奖励小于 0 的部分直接裁剪为 0（legged_robot.py:174-175）。在 G1 原始配置下，全部负权重惩罚项都会因此被"剪掉"。开启自碰撞负惩罚后，必须关闭该裁剪，负梯度才能真正反向传播。
> 副作用：所有负惩罚项（`lin_vel_z`、`orientation`、`base_height` 等）也开始真实生效——这本来就是它们的语义，训练会稍更保守、更稳。

## 五、如何运行

### 5.1 训练（论文方法 CBF）

```bash
conda activate isaacgym
cd ~/RL/code/unitree_rl_gym/unitree_rl_gym

# 完整训练（默认 3000 轮，4096 环境，RTX 3060 约 1 小时）
python legged_gym/scripts/train.py --task=g1_sc --headless

# 后台训练（SSH 断开不中断）
tmux new -s train
conda activate isaacgym
cd ~/RL/code/unitree_rl_gym/unitree_rl_gym
python legged_gym/scripts/train.py --task=g1_sc --headless
# Ctrl+B D 断开；tmux attach -t train 重新查看
```

### 5.2 切换 CBF / APF 对比实验

编辑 `g1_self_collision_config.py` 中的 `rewards.scales`，每次只开一个：

```python
# 实验 A（论文方法）：CBF 势垒
self_collision_cbf = -0.5
self_collision_apf = 0.0

# 实验 B（基线）：APF 二次惩罚
self_collision_cbf = 0.0
self_collision_apf = -0.5
```

配合不同 `run_name`（`--run_name=cbf` / `--run_name=apf`）即可在 TensorBoard 里对比两条曲线，对应论文 Fig.4 的对比结论。

### 5.3 查看训练曲线

```bash
tensorboard --logdir=~/RL/code/unitree_rl_gym/unitree_rl_gym/logs/
# 关注指标：Train/mean_reward、Episode/rew_self_collision_cbf（应趋近 0）、
#          Policy/mean_noise_std、Loss/*
```

### 5.4 可视化验证

```bash
cd ~/RL/code/unitree_rl_gym/unitree_rl_gym
# 加 --num_envs=1 只加载 1 个机器人，避免默认 100 个并行环境导致负载过大
python legged_gym/scripts/play.py --task=g1_sc --num_envs=1
```

会加载 `logs/g1_sc/` 下最新模型并用 IsaacGym 窗口演示。重点观察：
- 行走时双腿是否出现交叉/刮蹭（对应论文 Fig.5 摆腿自碰撞场景）；
- 受 `domain_rand.push_robots` 随机推力后恢复过程中是否有自碰撞。

### 5.5 MuJoCo 部署运行（sim-to-sim）

将 IsaacGym 训练出的策略导出并在 **MuJoCo** 中运行，可脱离 IsaacGym 直接验证真实物理效果（同一 `g1_12dof` 模型，保证 sim-to-sim 一致性）。

```bash
cd ~/RL/code/unitree_rl_gym/unitree_rl_gym

# 1) 导出策略为 TorchScript（自动生成 logs/g1_sc/exported/policies/policy_lstm_1.pt）
python legged_gym/scripts/play.py --task=g1_sc --headless --num_envs=1

# 2) MuJoCo 中运行（图形窗口；--headless 可无显示器运行，用于 CI/冒烟测试）
python deploy/deploy_mujoco/deploy_mujoco.py g1_sc.yaml
python deploy/deploy_mujoco/deploy_mujoco.py g1_sc.yaml --headless
```

- 配置文件：`deploy/deploy_mujoco/configs/g1_sc.yaml`，`policy_path` 指向导出的 LSTM 策略；其余参数与 g1.yaml 一致（PD 增益、`default_angles`、归一化尺度、`cmd_init=[0.5,0,0]` 即 0.5 m/s 前进）；
- 观测构造与训练完全一致：47 维 = 角速度(3) + 投影重力(3) + 命令(3) + dof_pos(12) + dof_vel(12) + 上一动作(12) + 相位 sin/cos(2)，相位周期 0.8 s；
- LSTM 记忆由导出模块内部维护（`PolicyExporterLSTM`），无需手动管理 hidden state；
- 冒烟验证（headless 5 s）：基座高度稳定 ~0.77 m、前进速度 ~0.5 m/s、动作与状态无 NaN。

## 六、参数调优指南

| 参数 | 默认 | 含义 | 调节建议 |
|------|------|------|---------|
| `safe_distance` | 0.10 | 安全距离（等价论文的 `ρA+ρB`） | 调大 → 更保守、更早躲避；调小 → 允许更紧凑动作 |
| `barrier_alpha` | 8.0 | 势垒陡峭度 | 调大 → 惩罚更集中在贴脸瞬间，远处几乎无惩罚 |
| `approach_beta` | 1.0 | 接近速率项权重（CBF 特色） | 调大 → 更强"提前刹车"；设 0 退化为纯位置势垒 |
| `max_approach_rate` | 2.0 | 速率项上限 [m/s] | 防止复位/瞬间跳变产生极大惩罚 |
| `self_collision_cbf` | -0.5 | 奖励权重 | 与 `tracking_lin_vel`(=1.0) 等正奖励权衡；权重过大可能不愿动腿 |
| `self_collision_apf` | 0.0 | APF 权重 | 同上，仅在 APF 实验时开启 |
| `collision` | -0.5 | 原有接触力惩罚 | 保留，作为接触层的兜底 |
| 关节限位 | — | legged_gym 自带 | `soft_dof_pos_limit=0.9`、`dof_pos_limits=-5.0` 已覆盖论文的关节限位 CBF |

经验：若训练中 `rew_self_collision_cbf` 长期不为 0，先调大 `barrier_alpha`（更聚焦），再考虑调大权重；若机器人"不敢走路"（速度跟踪奖励低），适当调小权重或调大 `safe_distance` 与 `barrier_alpha` 的组合，让惩罚更局部。

## 七、PPO 训练参数（复用 G1 基线）

`G1SelfCollisionRoughCfgPPO` 继承 `G1RoughCfgPPO`：

| 项 | 值 |
|----|----|
| 策略网络 | ActorCriticRecurrent（LSTM, hidden 64） |
| 迭代数 `max_iterations` | 3000（可加大） |
| 学习率 / 裁剪 | 1e-3 / 0.2（自适应 KL） |
| 实验名 | `g1_sc` |

详细 PPO 原理与调参见仓库根目录 `README.md` 第六章。

## 八、复现进度

- [x] 新增 `g1_sc` 任务（环境 + 配置 + 注册）
- [x] 冒烟测试通过（`self_collision_cbf` 奖励正常参与计算）
- [ ] 完整训练（3000 轮，tmux `train` 会话中运行）
- [ ] 用 `play.py --task=g1_sc` 可视化验证无自碰撞行走
- [ ] 可选：`g1_sc_apf` 对照组训练，与 CBF 对比
