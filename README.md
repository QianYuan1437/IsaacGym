# 机器人强化学习环境配置与训练指南

基于 NVIDIA IsaacGym + RSL-RL + Unitree RL Gym + MuJoCo 的完整操作手册。

> 本书已按 **GitBook** 形式组织：左侧目录导航由 `SUMMARY.md` 定义，各章节为 `guide/` 下的独立文件，可配合 GitBook / HonKit / 任意 Markdown 阅读器直观阅读。

---

## 章节导航

- [第一章：完整安装与配置](guide/01-install.md)
- [第二章：训练](guide/02-training.md)
- [第三章：查看训练效果](guide/03-viewing-results.md)
- [第四章：常用快捷操作](guide/04-quick-reference.md)
- [第五章：注意事项与原理解释](guide/05-notes-and-principles.md)
- [第六章：PPO 算法详解与参数调优](guide/06-ppo-details.md)
- [第七章：G1 人形机器人自碰撞避免论文复现（g1_sc）](guide/07-g1-self-collision.md)
- [附录：G1 自碰撞复现详细指南](guide/08-appendix-g1-sc-detailed.md)

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
    ├── README.md                                # GitBook 首页
    ├── SUMMARY.md                               # GitBook 目录导航
    ├── guide/                                   # GitBook 各章节文件（第一章~附录）
    ├── SESSION_STATE.md                         # 会话存档（续接任务用）
    └── code/unitree_rl_gym/
        ├── rsl_rl/                              # RSL-RL (PPO 算法实现)
        └── unitree_rl_gym/                      # Unitree RL Gym (宇树机器人训练模板)
              ├── legged_gym/
              │   ├── envs/                      # 机器人配置 (go2/g1/h1/h1_2/g1_sc)
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

## 快速开始

```bash
conda activate isaacgym
cd ~/RL/code/unitree_rl_gym/unitree_rl_gym

# 训练（详见第二章 / 第七章）
python legged_gym/scripts/train.py --task=go2 --headless

# 可视化验证（详见第三章；加 --num_envs=1 只加载 1 个机器人以降低负载）
python legged_gym/scripts/play.py --task=go2 --num_envs=1

# 监控
tensorboard --logdir=logs/
```
