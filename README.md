# GoogLeNet 猫狗图像分类

在同一「猫 / 狗」二分类任务上，对比两条技术路线：

- **从零手写 GoogLeNet**：自己实现 Inception 模块与辅助分类器，从随机初始化开始训练；
- **ImageNet 迁移学习**：复用 `torchvision` 预训练 GoogLeNet 特征，仅替换分类头并微调。

项目目标在于理解 GoogLeNet 网络结构、训练技巧，并通过对照实验直观认识「小数据下迁移学习 vs 从零训练」的差距。

---

## 功能特性

- 两条训练路线：从零训练（`model_train.py`）与迁移学习（`train_transfer.py`）；
- 测试阶段输出**混淆矩阵 + ROC + PR** 三子图，并弹窗显示（图片同时落盘到 `result/`）；
- 各优化阶段训练曲线可视化，保存在 `result/`；
- 附带两个独立辅助工具：数据集划分与归一化统计（见 `tools/`）；
- CUDA 可用时自动启用 GPU，否则回退 CPU。

---

## 目录结构

```
GoogLeNet_with_my_data/
├── model.py                 # 手写从零 GoogLeNet（Inception + 辅助分类器）
├── model_train.py           # 从零版训练脚本
├── model_test.py            # 从零版测试 + 可视化
├── model_transfer.py        # 迁移学习网络构建（torchvision 预训练 GoogLeNet）
├── train_transfer.py        # 迁移学习训练脚本
├── model_test_transfer.py   # 迁移版测试 + 可视化
├── result/                  # 训练曲线与测试可视化图片
├── tools/                   # 数据准备与统计辅助工具（各带 README）
│   ├── split_dataset.py             # 数据集分层划分
│   ├── split_dataset_README.md      # 划分工具说明
│   ├── compute_mean_std.py          # 数据集 mean/std 计算
│   └── compute_mean_std_README.md   # 统计工具说明
├── data/                    # 数据集（ImageFolder 结构，见下文；由 .gitignore 排除）
├── 完整实验报告.md           # 面向老师的实验报告（图文结合）
└── 模型演进报告.md           # 按 Git 提交逐阶段说明代码演进
```

---

## 环境依赖

- Python 3（建议使用 Anaconda 环境 `pytorch`）
- `torch` / `torchvision` / `matplotlib` / `numpy`

如使用 Anaconda，可创建并激活环境后安装依赖：

```bash
conda create -n pytorch python=3.9
conda activate pytorch
conda install pytorch torchvision torchaudio cpuonly   # 或按 CUDA 版本安装
pip install matplotlib numpy
```

> `tools/` 下的两个工具不依赖 torch：`split_dataset.py` 仅用 Python 标准库，`compute_mean_std.py` 仅需 `numpy` + `Pillow`。

---

## 数据准备

数据按 PyTorch `ImageFolder` 结构组织（本项目实测规模：训练集 1721 张、测试集 304 张）：

```
data/
├── train/
│   ├── cats/   *.jpg
│   └── dogs/   *.jpg
└── test/
    ├── cats/   *.jpg
    └── dogs/   *.jpg
```

若手头是 `data/cats`、`data/dogs` 这类「按类别分文件夹」的原始目录，可用 `tools/split_dataset.py` 一键划分为上面的 `train/test` 结构（详见「辅助工具」章节）。

> **注意**：`data/` 目录与训练得到的权重文件（`best_model.pth`、`best_model_transfer.pth`）由 `.gitignore` 排除，**不纳入版本库**。请自行准备猫狗图片并放入上述目录结构后运行。

---

## 辅助工具

`tools/` 下提供两个可独立使用的工具，也可 `from split_dataset import ...` 在代码中复用。

### split_dataset.py — 数据集划分

把「按类别分文件夹」的图片按**分层抽样**重新组织为 `data/{train,test}/<class>/`，默认 **train 85% / test 15%**，固定随机种子可复现。

```powershell
# 在项目根目录执行
python tools/split_dataset.py --data_dir ./data            # 默认 85%/15%，移动模式（原地重组织）
python tools/split_dataset.py --data_dir ./data --copy     # 复制模式，保留原始目录
python tools/split_dataset.py --data_dir ./data --ratios 0.8 0.2 --seed 42
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data_dir` | `./data` | 数据集根目录，其每个子文件夹视为一个类别。 |
| `--ratios` | `0.85 0.15` | 训练/测试比例，两值之和须为 1。 |
| `--seed` | `42` | 随机种子，固定后划分可复现。 |
| `--copy` | 关闭 | 复制而非移动，保留原图。 |

> 若 `data/` 下已存在 `train` / `test` 目录，脚本会报错退出，避免重复划分破坏数据。

### compute_mean_std.py — 归一化参数统计

用 **IQR 四分位距**自动识别并剔除异常图片后，计算数据集逐通道 mean / std，可直接作为 `transforms.Normalize(mean, std)` 参数。**从零版使用的归一化常数即由此工具计算得到**。

```powershell
python tools/compute_mean_std.py --data_dir ./data
python tools/compute_mean_std.py --data_dir ./data --image_size 224 --iqr_k 1.5
```

输出示例：

```
==== 数据集均值/标准差统计 ====
图片总数    : 2023
异常剔除    : 42
  通道 0: mean = 0.481451   std = 0.256604
  通道 1: mean = 0.447649   std = 0.247922
  通道 2: mean = 0.407904   std = 0.250483

Normalize 可用参数：
  mean = [0.481451, 0.447649, 0.407904]
  std  = [0.256604, 0.247922, 0.250483]
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data_dir` | 脚本同级 `data/` | 图片根目录（可含类别子目录）。 |
| `--iqr_k` | `1.5` | IQR 倍数，`k` 越大越宽松、剔除越少。 |
| `--image_size` | `224` | Resize 边长，需与训练 `transforms.Resize` 一致；`0` 用原图。 |
| `--force_channels` | 自动 | 强制通道数：`1` 灰度 / `3` RGB。 |
| `--no_recursive` | 关闭 | 只扫描 `data_dir` 下一级。 |

> 迁移学习版**不使用**该统计值，而是采用 **ImageNet 常数**归一化，以匹配预训练权重的输入分布。

两工具的完整参数、原理与常见问题，见 `tools/split_dataset_README.md` 与 `tools/compute_mean_std_README.md`。

---

## 使用说明

在 `GoogLeNet_with_my_data/` 目录下执行：

| 目的 | 命令 | 产物 |
|------|------|------|
| 从零训练 | `python model_train.py` | 权重 `best_model.pth` |
| 迁移训练 | `python train_transfer.py` | 权重 `best_model_transfer.pth` |
| 从零测试 | `python model_test.py` | `result/model_test.png`（弹窗 + 落盘） |
| 迁移测试 | `python model_test_transfer.py` | `result/model_transfer_test.png`（弹窗 + 落盘） |

> 测试脚本运行结束会**直接弹窗**显示可视化结果；在无图形界面的环境（服务器 / 容器）运行时，可临时设置 `MPLBACKEND=Agg` 跳过弹窗、仅保存图片（无需改代码）：
> ```bash
> MPLBACKEND=Agg python model_test.py
> ```

---

## 模型与结果

在测试集（共 304 张，每类 152 张）上的整体准确率：

| 方案 | 测试准确率 |
|------|-----------|
| 从零手写 GoogLeNet（最终版） | **0.8224** |
| ImageNet 迁移学习 | **0.9704** |

结果表明：在小规模数据下，迁移学习显著优于从零训练；从零版的精度上限更多由数据规模决定。详细的训练曲线、混淆矩阵、ROC/PR 分析与方法对比，见 **`完整实验报告.md`**。

---

## 文档索引

- **`完整实验报告.md`** — 面向老师的实验报告，含数据集说明、方法演进、7 张图表与图文分析、讨论与结论；
- **`模型演进报告.md`** — 按 6 个 Git 提交逐阶段讲解代码改动与优化动机；
- **`tools/split_dataset_README.md`** — 数据集划分工具的参数、分层抽样原理与常见问题；
- **`tools/compute_mean_std_README.md`** — 归一化统计工具的参数、IQR 异常判定原理与常见问题。

---

## 备注

- 训练脚本仅保存验证集最优权重，未留存逐 epoch 快照；如需复现中间版本，可自行在训练脚本中增加 checkpoint 保存逻辑。
- 测试集样本量有限（每类 152 张），绝对准确率存在一定统计波动，对比结论以两方案相对差距为准。
