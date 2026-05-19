# SJZ-dunbingmiaozhun

GPU-accelerated AI 瞄准检测系统。选择任意窗口或进程，系统将持续捕获画面并使用CNN进行实时瞄准状态检测。

---

## 功能特性

- **窗口选择器** — 可搜索的所有打开窗口列表，显示应用名称、标题、PID和尺寸
- **实时预览** — 带注释的画面流，每一帧上绘制检测结果
- **检测日志** — 时间戳日志面板，显示检测状态、置信度和坐标；限制500行以保持内存稳定
- **窗口跟踪** — 每帧跟踪位置变化；即使窗口移动或被遮挡，捕获仍继续
- **GPU推理** — 使用TensorFlow进行CNN推理；支持GPU加速
- **可配置间隔** — 默认100ms（约10 FPS）；可通过工具栏调整（10-5000ms）
- **会话日志文件** — 每次运行将时间戳日志写入 `etc/logs/<YYYYMMDD_HHMMSS>.log`，便于后续调试
- **自动截图** — 支持手动截图和定时自动截图功能
- **瞄准检测** — 使用CNN模型实时检测准心瞄准状态

---

## 需求条件

- Python 3.10+
- NVIDIA GPU（推荐，支持CUDA；支持CPU回退）
- [uv](https://docs.astral.sh/uv/)（推荐）**或** pip
- TensorFlow 2.15+

---

## 安装步骤

### 使用 uv（推荐）

```bash
# 1. 克隆项目
git clone <repository-url>
cd SJZ-dunbingmiaozhun

# 2. 同步环境（uv自动创建.venv）
uv sync

# 3. 运行
uv run python main.py
```

### 使用 pip

```bash
# 1. 克隆项目
git clone <repository-url>
cd SJZ-dunbingmiaozhun

# 2. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. 安装TensorFlow（根据您的系统选择合适的版本）
# GPU版本（需要NVIDIA CUDA）:
pip install tensorflow[and-cuda]
# CPU版本:
# pip install tensorflow

# 4. 安装剩余依赖
pip install -r requirements.txt
```

---

## 使用方法

```bash
python main.py
```

### 基础操作步骤

1. 点击工具栏中的 **选择窗口**。
2. 找到并双击要监控的窗口。
3. 点击 **开始** 开始捕获和检测。
4. 实时预览显示瞄准状态；右侧面板记录每次检测。
5. 随时点击 **取消聚焦** 释放目标窗口。

### 手机投屏检测配置（IQOO 手机）

1. **准备工作**：
   - 将 IQOO 手机通过 USB 连接到电脑
   - 打开 `vivo 手机管家` 软件
   - 在手机管家中找到投屏功能并启用
   - 将投屏窗口最大化

2. **软件配置**：
   - 打开 UI 窗口后，在设置中将裁剪大小从 640 调整为 64
   - 点击 **选择窗口** 选择投屏窗口
   - 点击 **开始** 开始运行

3. **开始检测**：
   - 系统会自动捕获投屏画面并进行瞄准状态检测
   - 检测结果会实时显示在预览窗口和日志面板中

### 截图功能

- **F9 / 截图按钮** — 手动截取当前帧，保存到瞄准状态目录
- **F10 / 自动截图按钮** — 启用/停止定时自动截图，保存到未瞄准状态目录

---

## 项目结构

```
SJZ-dunbingmiaozhun/
├── main.py                        # 入口文件
├── pyproject.toml                 # 项目元数据 + uv/pip配置
├── uv.lock                        # 锁定的依赖图
├── requirements.txt               # pip兼容的依赖列表
├── augment_data.py                # 数据增强脚本
├── click_at_position.py           # 点击位置脚本
├── train_cnn.py                   # CNN训练脚本
├── 准心数据集/
│   ├── 已瞄准状态/                 # 瞄准状态图片数据集
│   └── 未瞄准状态/                 # 未瞄准状态图片数据集
├── etc/
│   └── logs/                      # 会话日志文件（自动创建，gitignored）
└── panopticon/
    ├── app.py                     # QApplication引导 + 深色主题
    ├── logging_setup.py           # 根日志配置（文件 + 控制台）
    ├── ui/
    │   ├── main_window.py         # 主窗口（预览 + 日志）
    │   └── window_selector.py     # 窗口/进程选择对话框
    ├── capture/
    │   ├── manager.py             # QThread捕获循环
    │   ├── screenshot.py          # 跨平台截图
    │   ├── screenshot_capture.py  # 截图捕获
    │   ├── classifier.py          # CNN分类器
    │   └── __init__.py
    ├── utils/
    │   └── platform.py            # 操作系统特定的窗口枚举
    └── __init__.py
```

---

## CNN模型说明

项目使用自定义CNN模型进行瞄准状态检测：

### 模型架构

```python
Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])
```

### 输入要求

- 图片尺寸：64x64像素
- 颜色格式：RGB
- 归一化：像素值除以255

### 输出

- `status`: 'aimed'（瞄准中）或 'not_aimed'（未瞄准）
- `confidence`: 置信度（0-1）

模型文件需命名为 `准心状态分类器.h5` 放在项目根目录。

---

## 平台支持

| 平台 | 窗口枚举 | 捕获方法 | 状态 |
|---|---|---|---|
| Linux (X11) | `python-xlib` | `mss` | 支持 |
| Linux (KDE Wayland) | KWin D-Bus | `spectacle` 全屏裁剪 | 支持 |
| Windows | `pywin32` | `PrintWindow` + `mss` 回退 | 支持 |
| macOS | `pyobjc-framework-Quartz` | `mss` | 支持 |

### 已知限制

- **Wayland (非KDE):** GNOME、Sway、Hyprland和其他Wayland合成器不支持。非KDE Wayland上的窗口枚举和屏幕捕获需要合成器特定的门户，尚未实现。
- **Wayland (KDE):** 捕获需要安装 `spectacle`（随KDE Plasma附带）并在 `$PATH` 中可用。每一帧触发全屏抓取然后裁剪，比X11/mss路径慢。
- **最小化窗口 (Windows):** 使用 `PrintWindow` 捕获最小化或其他窗口后面的窗口。某些通过DirectX/Vulkan渲染的应用可能返回空白帧。
- **macOS:** 捕获需要在 **系统设置 → 隐私与安全 → 屏幕录制** 中授予终端或应用屏幕录制权限。
- **GPU要求:** GPU加速需要NVIDIA GPU和兼容的CUDA工具包。AMD和Apple Silicon GPU支持需安装对应版本的TensorFlow。
- **遮挡/屏幕外窗口:** 如果目标窗口完全在屏幕外或其几何形状无法解析，捕获会静默不产生帧，直到窗口重新定位。

---

## 数据集

项目包含瞄准检测数据集：
- `准心数据集/已瞄准状态/` — 包含瞄准状态的图片
- `准心数据集/未瞄准状态/` — 包含未瞄准状态的图片

数据增强脚本 `augment_data.py` 可用于扩展数据集，支持以下增强操作：
- 模糊处理
- 亮度调整
- 水平翻转
- 垂直翻转
- 旋转（90°, 180°, 270°）

---

## 许可证

MIT