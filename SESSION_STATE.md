# 会话存档：G1 自碰撞避免论文复现（g1_sc）

> 本文件用于跨会话续接任务。下次对话请先读本文件，再 `tmux attach -t train` 查看训练实时状态。
> 存档时间：2026-08-06（约 00:26）

---

## 1. 任务背景

在 IsaacGym + RSL-RL + legged_gym 环境（conda: `isaacgym`，Python 3.8，RTX 3060 12G）上用 **G1 人形机器人** 复现论文：
*Humanoid Self-Collision Avoidance Using Whole-Body Control with Control Barrier Functions*（Khazoom 等，IEEE-RAS Humanoids 2022，MIT）。
论文 PDF 在 `~/RL/docx/`（未纳入 git）。

核心思路：论文是 CBF 硬约束控制器，本复现将其翻译为 PPO **奖励塑形**：
- CBF 势垒奖励 `_reward_self_collision_cbf`：`exp(-alpha·(d-safe))·(1+beta·max(0,-ḋ))`（含接近速率项，对应论文 `ḣ+αh≥0`）
- APF 基线奖励 `_reward_self_collision_apf`：`max(0, safe-d)²`（论文 Eq.23，供对比）
- 关键配置：`only_positive_rewards = False`（否则负惩罚被裁剪，无梯度）

## 2. 已完成的代码与文档（commit `0c27a6f`，已推送 origin/main）

| 文件 | 状态 |
|------|------|
| `code/unitree_rl_gym/unitree_rl_gym/legged_gym/envs/g1/g1_self_collision.py` | 已提交推送 |
| `code/unitree_rl_gym/unitree_rl_gym/legged_gym/envs/g1/g1_self_collision_config.py` | 已提交推送 |
| `code/unitree_rl_gym/unitree_rl_gym/legged_gym/envs/__init__.py`（注册 `g1_sc`） | 已提交推送 |
| `code/unitree_rl_gym/unitree_rl_gym/legged_gym/README_SELF_COLLISION.md`（独立复现指南） | 已提交推送 |
| `README.md`（新增第六章 PPO 详解 + 第七章 g1_sc 复现） | 已提交推送 |

git 分支 `main`，最近提交：`0c27a6f`（g1_sc）、`37bac0d`（第六章 PPO）。工作区仅剩未跟踪的 `docx/`（论文 PDF，有意不提交）。

## 3. 当前训练状态（进行中）

- 会话：tmux **`train`**（`tmux attach -t train` 查看，`Ctrl+B D` 脱离）
- 命令：`python legged_gym/scripts/train.py --task=g1_sc --headless`（4096 环境，`max_iterations=3000`）
- PID：**38058**（确认存活：`ps aux | grep train.py`）
- 进度：约 **372 / 3000** 轮（存档时刻），ETA 约 51 分钟，单轮约 1.2s
- 日志目录：`logs/g1_sc/Aug06_00-17-07_/`（每 50 轮存 `model_{it}.pt`）
- 冒烟测试目录：`logs/g1_sc/Aug06_00-16-25_/`（2 轮，可忽略）
- 训练正常指标：`Episode/rew_self_collision_cbf` 应随训练趋近 0；`Policy/mean_noise_std` 从 ~0.8 下降

## 4. 剩余任务

- [ ] 等训练跑到 3000 轮自动结束（模型 `model_3000.pt`）
- [ ] 可视化验证：`python legged_gym/scripts/play.py --task=g1_sc`，重点观察行走时双腿是否交叉/刮蹭（对应论文 Fig.5 摆腿自碰撞场景）
- [ ] 可选：APF 对照组训练（把 `self_collision_apf` 设为 -0.5、`self_collision_cbf` 设为 0，配 `--run_name=apf`），与 CBF 对比，对应论文 Fig.4
- [ ] 可选：调参（`safe_distance`、`barrier_alpha`、`approach_beta`、权重），详见 `README_SELF_COLLISION.md` 第六节
- [ ] 可选：训练完成后提交新模型产物说明 / 更新进度勾选（README_SELF_COLLISION.md 第八节）

## 5. GitBook 重构（2026-08-06 新增）

仓库 README 已重构为 GitBook 形式：
- `README.md` → 首页（环境信息 + 章节导航链接）
- `SUMMARY.md` → GitBook 目录导航
- `guide/` → 各章节独立文件（01~07 + 附录 08）
- 查看方式：本地 `honkit serve`（或任意 Markdown 阅读器）；远程直接看 GitHub 渲染
- 后续新增章节：在 `guide/` 新建文件 + 更新 `SUMMARY.md` 与首页导航链接

## 6. 常用命令速查

```bash
# 环境
conda activate isaacgym
cd ~/RL/code/unitree_rl_gym/unitree_rl_gym

# 训练 / 可视化 / 监控
python legged_gym/scripts/train.py --task=g1_sc --headless
python legged_gym/scripts/play.py --task=g1_sc
tensorboard --logdir=~/RL/code/unitree_rl_gym/unitree_rl_gym/logs/

# 远程同步（如需更新 README 章节后）
git -C ~/RL add -A && git -C ~/RL commit -m "..." && git -C ~/RL push origin main
```

## 7. 注意事项 / 坑

- `import isaacgym` 必须在 `import torch` 前；NumPy 固定 1.20；Python 必须 3.8（IsaacGym Preview 4）
- `only_positive_rewards = False` 会让所有负惩罚项真实生效，训练偏保守属正常
- 12 关节 G1 URDF 无臂部，臂部碰撞对自动跳过；若换 `g1_29dof` 会自动激活
- 训练用 `--headless`；后台用 tmux，SSH 断开不中断
- 论文 PDF（`~/RL/docx/`）体积大且涉及版权，未纳入 git
