# 第五章：注意事项与原理解释

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
