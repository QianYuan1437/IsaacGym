# 第二章：训练

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
