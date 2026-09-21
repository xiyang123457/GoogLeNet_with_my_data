# 图像数据集均值/标准差计算工具 (compute_mean_std.py)

读取图片文件夹，逐通道（RGB 或灰度）计算像素的**均值 (mean)** 与**标准差 (std)**，
用于训练时 `transforms.Normalize(mean, std)` 的归一化参数；并通过 **IQR 四分位距**
自动识别并剔除异常图片，使异常数据不计入统计量。

## 特性

- **逐通道统计**：自动识别灰度(1 通道) / 彩色 RGB(3 通道)，统一处理。
- **异常图剔除**：用 IQR 法（默认 1.5 倍）判定异常图片并排除，结果更稳健、无需假设分布。
- **与训练管线对齐**：图像先 `Resize(image_size)`（默认 224，等价于 `transforms.Resize`）
  再 `/255` 转为 `[0,1]`，算出的 mean/std 可直接喂给 `Normalize`。
- **内存友好**：只常驻「每张图均值」向量，像素级 sum/sumsq/count 增量累加，不一次性载入全部像素。
- **解码容错**：单张损坏图片跳过并记录，不影响整体统计。
- **双入口**：既可用命令行直接运行，也可 `from compute_mean_std import compute_dataset_mean_std` 在代码中复用。
- **轻量依赖**：仅 `numpy` + `Pillow`，不依赖 `torch` / `torchvision`。

## 异常判定原理

1. 对每张图计算逐通道像素均值（标量向量）。
2. 对「所有图的逐通道均值」做 IQR：区间 `[Q1 - k·IQR, Q3 + k·IQR]`。
3. **某张图任意通道均值落在区间之外 → 整张图被剔除**（按图为单位，而非单像素）。
4. 用剔除后剩余图片的**全部像素**重算逐通道均值与标准差（总体标准差，ddof=0）。

> 单位选「整张图的均值」而非「单个像素」：单像素 IQR 在连续的图像像素分布上几乎无区分度，
> 而「某张图整体偏亮/偏暗/损坏」才是真正异常，按图剔除更符合数据清洗目的。

## 目录约定

输入目录（可含类别子目录，递归扫描 jpg/png/bmp/webp/tif/gif）：

```
data/
├── cats/
│   ├── cat.0001.jpg
│   └── ...
└── dogs/
    ├── dog.0001.jpg
    └── ...
```

## 命令行用法

```powershell
# 默认：读脚本同级 data/ 文件夹，224 尺寸，IQR 倍数 1.5
python compute_mean_std.py

# 指定数据目录
python compute_mean_std.py --data_dir "D:\pytorch_test\GoogLeNet_with_my_data\data"

# 调整异常判定强度（k 越大越宽松，剔除越少）
python compute_mean_std.py --data_dir ./data --iqr_k 2.0

# 用原图尺寸（不做 Resize，速度更快但需与训练管线一致）
python compute_mean_std.py --data_dir ./data --image_size 0

# 强制按灰度(1 通道)或 RGB(3 通道)处理
python compute_mean_std.py --data_dir ./data --force_channels 3

# 只扫描 data_dir 下一级（不递归子目录）
python compute_mean_std.py --data_dir ./data --no_recursive
```

运行输出示例：

```
==== 数据集均值/标准差统计 ====
图片目录    : D:\pytorch_test\GoogLeNet_with_my_data\data
通道数      : 3
图片总数    : 2025
成功解码    : 2025
异常剔除    : 42
  通道 0: mean = 0.481451   std = 0.256604
  通道 1: mean = 0.447649   std = 0.247922
  通道 2: mean = 0.407904   std = 0.250483

Normalize 可用参数：
  mean = [0.481451, 0.447649, 0.407904]
  std  = [0.256604, 0.247922, 0.250483]
```

> 注：示例中的 `图片总数` 已按当前数据集更正为 **2025** 张（= `data/train` 1721 + `data/test` 304，与 `README.md`「数据准备」一致）。`异常剔除` 与 mean/std 数值为当时的统计结果——其中 mean/std 与训练脚本中硬编码的常数一致，因此保留原值；若需最新数值，请重跑本工具。
>
> 另需注意：该示例是在 `data/` **全量（train + test）** 上递归统计得到的，因此从零版训练所用的归一化常数含有一点测试集信息（低维统计量，影响很小），详见 `README.md`「已知限制」第 3 条。

### 命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data_dir` | 脚本同级 `data/` | 图片根目录（可含类别子目录）。 |
| `--iqr_k` | `1.5` | IQR 倍数；判定区间 `[Q1 - k·IQR, Q3 + k·IQR]`。 |
| `--image_size` | `224` | 预处理 Resize 边长（与训练管线对齐）；`0` 表示用原图尺寸。 |
| `--force_channels` | 自动 | 强制通道数：`1`=灰度，`3`=RGB；默认按首张有效图自动判定。 |
| `--no_recursive` | 关闭 | 加上后不递归子目录，只扫描 `data_dir` 下一级。 |

## 作为库复用

```python
from compute_mean_std import compute_dataset_mean_std

res = compute_dataset_mean_std(
    image_dir="path/to/data",   # 图片根目录
    iqr_k=1.5,                  # IQR 倍数
    image_size=224,             # Resize 边长，None/0=原图
    recursive=True,             # 是否递归子目录
    force_channels=None,        # None=自动, 1=灰度, 3=RGB
)

print(res["mean"])      # [R, G, B] 均值
print(res["std"])       # [R, G, B] 标准差
print(res["num_images"])    # 图片总数
print(res["num_valid"])     # 成功解码数
print(res["num_excluded"])  # 被 IQR 剔除的异常图数量
print(res["excluded"])      # 被剔除图片路径列表
print(res["errored"])       # 解码失败图片路径列表
print(res["channels"])      # 通道数

# 直接用于训练归一化
# transforms.Normalize(mean=res["mean"], std=res["std"])
```

### `compute_dataset_mean_std` 参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `image_dir` | `str` | 必填 | 图片根目录。 |
| `iqr_k` | `float` | `1.5` | IQR 倍数。 |
| `image_size` | `int \| None` | `224` | Resize 边长；`None`/`0` 用原图。 |
| `recursive` | `bool` | `True` | 是否递归子目录。 |
| `force_channels` | `int \| None` | `None` | 通道数：`1`/`3`/自动。 |

返回值为字典，字段见上方代码示例。

## 常见问题

- **被剔除的图片太多？** 调大 `--iqr_k`（如 2.0~3.0）放宽异常判定；或消失异常图本身确实是错误数据。
- **mean/std 想和训练完全一致？** 让 `--image_size` 与训练 `transforms.Resize` 的尺寸一致（本工具默认 224，匹配 GoogLeNet）。
- **解码失败的文件？** 会列在 `errored` 中并跳过，不参与任何统计；建议单独检查这些文件是否损坏。
- **全部被判定异常？** 脚本会打印警告并返回全 0 的 mean/std，避免除零崩溃。
- **灰度与彩色混存？** 默认按首张图自动决定通道数；如需统一，用 `--force_channels` 强制。
