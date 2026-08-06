# 第七章：G1 人形机器人自碰撞避免论文复现（g1_sc）

> 复现论文：*Humanoid Self-Collision Avoidance Using Whole-Body Control with Control Barrier Functions*（Khazoom 等，IEEE-RAS Humanoids 2022，MIT）
> 论文 PDF：`~/RL/docx/`
> 完整独立指南：[附录：G1 自碰撞复现详细指南](08-appendix-g1-sc-detailed.md)
> 训练命令：`python legged_gym/scripts/train.py --task=g1_sc --headless`

### 7.1 论文与复现思路

论文为 MIT Humanoid 提出 **CBF-WBC**（控制障碍函数 + 全身控制器）：用球/胶囊近似机器人几何，对有符号距离 `h(q) = ‖pA-pB‖-(ρA+ρB)` 施加 CBF 硬约束 `ḣ+αh≥0`，在 QP 中保证"永不自碰撞"，并用人工势场（APF，Eq.23）作基线对比。

本复现将这套安全思想翻译进 **PPO 奖励塑形**：

| 论文要素 | 本复现实现 |
|---------|-----------|
| CBF 约束 `ḣ+αh≥0` | `_reward_self_collision_cbf`：`exp(-α·(d-safe))·(1+β·max(0,-ḋ))`，指数势垒 + 接近速率项 |
| APF 软惩罚（Eq.23） | `_reward_self_collision_apf`：`max(0, safe-d)²`，仅阈值内生效 |
| 关节限位 CBF | legged_gym 自带 `dof_pos_limits` / `soft_dof_pos_limit` |

### 7.2 新增代码

```
legged_gym/envs/g1/g1_self_collision.py         # G1SelfCollisionRobot 环境类
legged_gym/envs/g1/g1_self_collision_config.py  # 配置（碰撞对、奖励、PPO）
legged_gym/envs/__init__.py                     # 注册任务 g1_sc
```

核心：`_sc_distances()` 用 `rigid_body_states_view`（每步刷新）计算各碰撞体对中心距；CBF 奖励里的 `ḋ` 有限差分项对应论文的动力学约束，是区别于纯位置 APF 的关键。

### 7.3 关键配置

- **`only_positive_rewards = False`**：必须关闭基类默认的奖励裁剪（legged_robot.py:174），否则负惩罚项全部被剪掉、无法产生梯度；
- 碰撞对在 `cfg.self_collision.pairs` 定义，12 关节 G1 只有腿部 4 对生效，臂部对自动跳过；换 `g1_29dof` 会自动激活臂部对；
- 对比实验：`rewards.scales` 中每次只开 `self_collision_cbf` 或 `self_collision_apf` 之一，配不同 `--run_name` 即可在 TensorBoard 对比。

### 7.4 训练 / 验证

```bash
conda activate isaacgym
cd ~/RL/code/unitree_rl_gym/unitree_rl_gym

# 训练（默认 3000 轮，RTX 3060 约 1 小时）
python legged_gym/scripts/train.py --task=g1_sc --headless

# 可视化验证（加载 logs/g1_sc/ 最新模型；加 --num_envs=1 只加载 1 个机器人以降低负载）
python legged_gym/scripts/play.py --task=g1_sc --num_envs=1

# 监控
tensorboard --logdir=~/RL/code/unitree_rl_gym/unitree_rl_gym/logs/
# 关键指标：Episode/rew_self_collision_cbf（应趋近 0，即不再自碰撞）
```

> 训练当前在 tmux 会话 `train` 中运行：`tmux attach -t train` 查看进度。

### 7.5 MuJoCo 部署运行（sim-to-sim）

策略训练完成后可脱离 IsaacGym 在 MuJoCo 中运行（同一 `g1_12dof` 模型）：

```bash
cd ~/RL/code/unitree_rl_gym/unitree_rl_gym

# 1) 导出策略为 TorchScript（生成 logs/g1_sc/exported/policies/policy_lstm_1.pt）
python legged_gym/scripts/play.py --task=g1_sc --headless --num_envs=1

# 2) MuJoCo 中运行（--headless 可无显示器运行）
python deploy/deploy_mujoco/deploy_mujoco.py g1_sc.yaml
```

配置与原理详见 [第九章：MuJoCo 部署运行（sim-to-sim）](09-mujoco-deployment.md)。冒烟验证结果：机器人稳定行走（高度 ~0.77 m、速度 ~0.5 m/s），无 NaN。
