# 第六章：PPO 算法详解与参数调优

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
