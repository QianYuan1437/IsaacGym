# 第三章：查看训练效果

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
