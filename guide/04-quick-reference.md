# 第四章：常用快捷操作

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
